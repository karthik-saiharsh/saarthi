"""Post-processing of detector output. Pure functions with no model.

Order of the steps (see `select_candidates`):
1. Map each label to the phrases of the vocabulary (a label can name several).
2. Keep the best boxes of each phrase.
3. Clip the boxes and drop those with a bad size.
4. Merge duplicate boxes (class-agnostic NMS) and keep the union of labels.
5. Mark objects that lie inside a screen, a picture frame or a mirror.
6. Flag objects that a hand covers.
(Step 5 marks the object as not good. It does not drop it.)
"""

import dataclasses
import re
from collections.abc import Callable, Sequence

from PIL import Image

from ai.geometry import clip_box, iou, nms, overlap_fraction, touches_edge
from ai.quality import blur_score
from ai.settings import DetectionConfig
from ai.types import Box, Candidate, RawDetection, Source

# Maximum length of a text prompt for the detector, in tokens.
MAX_PROMPT_TOKENS: int = 256

_ARTICLES: frozenset[str] = frozenset({"a", "an", "the", "some"})
_FILLERS: frozenset[str] = frozenset({"of"})


def _words(text: str) -> list[str]:
    """Return the lowercase words of a text, without articles."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _ARTICLES]


def _head(words: list[str]) -> str:
    """Return the head noun: the last word that is not a filler."""
    content: list[str] = [w for w in words if w not in _FILLERS]
    return content[-1] if content else ""


def match_phrase(label: str, vocabulary: Sequence[str]) -> str:
    """Return the phrase of the vocabulary that a detector label stands for.

    The detector can return a part of a phrase ("bottle" for "a water bottle").
    Match in this order: same words, same head noun, label words inside the
    phrase. If nothing matches, return the label as it is.
    """
    assert isinstance(label, str), "label must be a string"
    label_words: list[str] = _words(label)
    if not label_words:
        return label.strip()
    entries: list[tuple[str, list[str]]] = [(p, _words(p)) for p in vocabulary]

    for phrase, words in entries:
        if words == label_words:
            return phrase

    label_head: str = _head(label_words)
    same_head: list[tuple[str, list[str]]] = [
        (p, w) for p, w in entries if _head(w) == label_head
    ]
    if same_head:
        return max(same_head, key=lambda e: len(set(e[1]) & set(label_words)))[0]

    for phrase, words in entries:
        if set(label_words) <= set(words):
            return phrase
    return label.strip()


def match_phrases(label: str, vocabulary: Sequence[str]) -> list[str]:
    """Return every vocabulary phrase that a detector label stands for.

    The detector joins all phrases that it links to a box in one label, for
    example "a mobile phone a mirror". The articles mark where one phrase
    ends and the next one starts. Phrases that match are returned once each,
    in the order of the label. A part that matches nothing stays as it is.
    """
    assert isinstance(label, str), "label must be a string"
    parts: list[list[str]] = [[]]
    for word in label.lower().split():
        if word in _ARTICLES:
            if parts[-1]:
                parts.append([])
        else:
            parts[-1].append(word)
    phrases: list[str] = []
    for words in parts:
        if not words:
            continue
        phrase: str = match_phrase(" ".join(words), vocabulary)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def plan_prompts(
    object_phrases: Sequence[str],
    extra_phrases: Sequence[str],
    chunk_size: int,
    count_tokens: Callable[[str], int],
) -> list[list[str]]:
    """Split the vocabulary into prompts for separate detector passes.

    Each prompt holds the extra phrases (hand, depictions) and as many object
    phrases as `chunk_size` and the token limit allow.
    """
    assert object_phrases, "object_phrases must not be empty"
    assert chunk_size > len(extra_phrases), "chunk_size must leave room for objects"

    def tokens(phrases: list[str]) -> int:
        # Two special tokens, and one period after each phrase.
        return 2 + sum(count_tokens(p) + 1 for p in phrases)

    if tokens(list(extra_phrases)) >= MAX_PROMPT_TOKENS:
        raise ValueError("The extra phrases alone fill the prompt")

    prompts: list[list[str]] = []
    current: list[str] = []
    for phrase in object_phrases:
        trial: list[str] = [*current, phrase]
        full: bool = len(trial) + len(extra_phrases) > chunk_size or (
            tokens([*trial, *extra_phrases]) > MAX_PROMPT_TOKENS
        )
        if full and current:
            prompts.append([*current, *extra_phrases])
            trial = [phrase]
        current = trial
    prompts.append([*current, *extra_phrases])
    return prompts


@dataclasses.dataclass(frozen=True)
class _Item:
    box: Box
    score: float
    source: Source
    phrases: tuple[str, ...]


def _cap_per_phrase(
    labelled: list[tuple[RawDetection, list[str]]], limit: int
) -> list[tuple[RawDetection, list[str]]]:
    """Keep the `limit` best boxes of each main phrase."""
    counts: dict[str, int] = {}
    kept: list[tuple[RawDetection, list[str]]] = []
    for detection, phrases in sorted(labelled, key=lambda e: -e[0].score):
        key: str = phrases[0] if phrases else ""
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= limit:
            kept.append((detection, phrases))
    return kept


def select_candidates(
    detections: list[RawDetection],
    width: int,
    height: int,
    vocabulary: Sequence[str],
    config: DetectionConfig,
) -> list[Candidate]:
    """Turn raw detections into candidates. The best score comes first.

    A box whose label names an object phrase is an object, even if the label
    also names a hand or a screen: the detector often adds weak extra phrases.
    A box that names no object is a hand or a depiction. `quality_ok` is False
    for an object that lies inside a depiction, and True for the others.
    `apply_crop_quality` can only lower it.
    """
    assert width > 0 and height > 0, "image size must be positive"
    known: list[str] = [
        *vocabulary,
        config.hand_phrase,
        *config.depiction_phrases,
        config.generic_phrase,
    ]
    depiction_phrases: frozenset[str] = frozenset(config.depiction_phrases)
    labelled: list[tuple[RawDetection, list[str]]] = [
        (d, match_phrases(d.label, known)) for d in detections
    ]

    hands: list[Box] = []
    depictions: list[Box] = []
    objects: list[_Item] = []
    for detection, phrases in _cap_per_phrase(labelled, config.max_per_phrase):
        box: Box | None = clip_box(detection.box, width, height)
        if box is None or min(box.width, box.height) < config.min_box_side:
            continue
        # A box that covers most of the frame is a detector error. Dropping it
        # is safer than letting it hide or flag every object in the frame.
        if box.area > config.max_frame_fraction * width * height:
            continue
        object_phrases: list[str] = [
            p for p in phrases if p != config.hand_phrase and p not in depiction_phrases
        ]
        if object_phrases or not phrases:
            objects.append(
                _Item(box, detection.score, detection.source, tuple(object_phrases))
            )
        elif config.hand_phrase in phrases:
            hands.append(box)
        else:
            depictions.append(box)

    kept: list[int] = nms(
        [o.box for o in objects], [o.score for o in objects], config.nms_iou
    )
    candidates: list[Candidate] = []
    for index in kept:
        item: _Item = objects[index]
        # The detector can box a book cover as a screen, so a real object on it
        # must not be lost. The candidate stays in the archive, but it is marked
        # not good, so that it cannot update the last-seen record.
        inside_depiction: bool = any(
            overlap_fraction(item.box, d) >= config.depiction_inside for d in depictions
        )
        phrases_out: list[str] = []
        for other in objects:
            if iou(item.box, other.box) >= config.nms_iou:
                phrases_out += [p for p in other.phrases if p not in phrases_out]
        in_hand: bool = any(
            overlap_fraction(item.box, h) > config.hand_overlap for h in hands
        )
        candidates.append(
            Candidate(
                item.box,
                item.score,
                item.source,
                tuple(phrases_out),
                in_hand,
                not inside_depiction,
            )
        )
    return candidates


def apply_crop_quality(
    candidates: list[Candidate],
    image: Image.Image,
    min_crop_side: int,
    crop_blur_min: float,
) -> list[Candidate]:
    """Set `quality_ok`. A candidate is good if its crop is big enough, sharp
    enough, and not cut by the image edge. A candidate that is already not good
    (for example, inside a depiction) stays not good.
    """
    assert isinstance(image, Image.Image), "image must be a PIL image"
    width, height = image.size
    result: list[Candidate] = []
    for candidate in candidates:
        box: Box = candidate.box
        good: bool = (
            candidate.quality_ok
            and min(box.width, box.height) >= min_crop_side
            and not touches_edge(box, width, height)
            and blur_score(image.crop((box.x0, box.y0, box.x1, box.y1)))
            >= crop_blur_min
        )
        result.append(dataclasses.replace(candidate, quality_ok=good))
    return result
