"""The search for objects that nobody registered (the open search, see `docs/backend/Open Search.md`).

It runs only when no enrolled object fits the words of the user. The broad detector stored the boxes
of each frame with a name and a SigLIP vector. This service finds the newest box that fits the words:

1. **Names.** The words become a phrase. The sentence encoder compares the phrase with the names that the
   detector gave to the stored boxes. A name with a high score fits. The newest box among the best names wins.
2. **Pictures.** If no name fits, the SigLIP text vector of the phrase is compared with the vector of each box.
   The best box must reach a probability, and must beat a few neutral phrases ("a wall", "an object").
3. **Rewrite.** If both fail, and the words are a description (3 words or more), a small language model rewrites them
   into two or three common names. Step 1 runs again with these names, and a name must fit almost exactly.
   The names only feed the search. They never reach the voice.

The answer is always weak: the voice says "I am not sure". The words of the user are private. The log shows counts only.
"""

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import AiError
from backend.search import Answer, AnswerSource
from backend.settings import BackendConfig, OpenSearchConfig
from db.schema import Image, OpenRegion, utc_now

logger: logging.Logger = logging.getLogger(__name__)

# Words that start a question, and words that point to a thing of the user. They are not the object.
_QUESTION_START = re.compile(
    r"^(?:(?:please|hey|hello|ok|okay|um|uh)\s+)*"
    r"(?:where(?:'s| is| are| did i (?:put|leave|keep|place|see)| can i find| could i find)"
    r"|i (?:can'?t|cannot|can not|couldn'?t|could not|don'?t know where) (?:find|see|locate)"
    r"|i (?:lost|misplaced|forgot where i (?:put|left|kept)|need)"
    r"|i(?:'m| am) looking for"
    r"|have you seen|did you see|help me find|help me look for|can you find|could you find"
    r"|find|show me|look for|looking for|locate)\s*",
    re.IGNORECASE,
)
_OWNER_WORDS = re.compile(
    r"^(?:my|the|a|an|our|your|some|any|that|this)(?:\s+|$)", re.IGNORECASE
)
_FILLER_END = re.compile(
    r"\s+(?:now|right now|please|again|anywhere|around|today|somewhere)$", re.IGNORECASE
)


def phrase_from_words(words: str) -> str | None:
    """Remove the question words and the pointing words. Return None if nothing is left.

    "where did I leave my laptop?" gives "laptop". A description ("that thing I check my mails on")
    keeps its words: the encoders read it as it is, and the small language model, if there is one, rewrites it.
    """
    assert isinstance(words, str), "words must be a string"
    text: str = " ".join(words.replace("’", "'").lower().split())
    text = text.strip(" ?.!,;:")
    text = _QUESTION_START.sub("", text, count=1)
    text = _FILLER_END.sub("", text.strip(" ?.!,;:"))
    text = _OWNER_WORDS.sub("", text.strip(), count=1)
    text = text.strip(" ?.!,;:")
    return text or None


@dataclass(frozen=True)
class OpenAnswer:
    """The newest box that fits the words."""

    label: str  # The name that the detector gave the box, for example "laptop"
    phrase: str  # What was searched for
    via: str  # "label" (step 1), "visual" (step 2) or "rewrite" (step 3)
    score: float  # The label score (cosine), or the SigLIP probability
    answer: Answer  # For the description: `source` is `open`, `strong` is False


class OpenSearchService:
    """Finds objects that nobody registered. It is safe to call from several threads."""

    def __init__(self, config: BackendConfig, engine: AiEngine) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: OpenSearchConfig = config.open_search
        self._crop_padding: float = config.search.crop_padding
        self._engine: AiEngine = engine
        self._lock: threading.Lock = threading.Lock()
        self._label_vectors: dict[str, np.ndarray] = {}
        self._neutral: np.ndarray | None = None

    @property
    def enabled(self) -> bool:
        """True if boxes can be stored and searched: the broad detector and the scene model are loaded."""
        return self._engine.open_detection_enabled and self._engine.scene_enabled

    def find(self, words: str, now: datetime | None = None) -> OpenAnswer | None:
        """Find the newest box that fits the words, or None."""
        assert isinstance(words, str), "words must be a string"
        assert now is None or isinstance(now, datetime), "now must be a datetime"
        if not self.enabled:
            return None
        phrase: str | None = phrase_from_words(words)
        if phrase is None:
            return None
        moment: datetime = now if now is not None else utc_now()
        rows: list[OpenRegion] = db.open_regions.list_open_regions(
            moment - timedelta(hours=self._config.max_scan_hours),
            moment,
            self._engine.open_vector_model,
            self._config.max_boxes,
        )
        if not rows:
            logger.info("Open search: no stored box")
            return None
        try:
            hit: tuple[OpenRegion, str, float] | None = self._by_label(phrase, rows)
            if hit is None:
                hit = self._by_picture(phrase, rows)
            if hit is None:
                hit = self._by_rewrite(phrase, rows)
        except AiError:
            logger.exception("Open search failed in a model")
            return None
        if hit is None:
            logger.info("Open search: no box fits (%d box(es) compared)", len(rows))
            return None
        row, via, score = hit
        logger.info("Open search: a box fits through the %s", via)
        return OpenAnswer(
            row.label, phrase, via, score, self._answer(row, score, moment)
        )

    def find_with_picture(
        self, words: str, max_side: int = 512, now: datetime | None = None
    ) -> tuple[OpenAnswer | None, bytes | None]:
        """Find the box and cut the picture of it (the box with `search.crop_padding` around it).

        The picture is None if the frame cannot be read. The answer is still true.
        """
        assert isinstance(max_side, int) and max_side > 0, "max_side must be positive"
        found: OpenAnswer | None = self.find(words, now)
        if found is None or found.answer.image_id is None or found.answer.box is None:
            return found, None
        try:
            image: Image | None = db.get_image(found.answer.image_id)
            if image is None:
                return found, None
            x0, y0, x1, y1 = found.answer.box
            pad_x: int = round((x1 - x0) * self._crop_padding)
            pad_y: int = round((y1 - y0) * self._crop_padding)
            box = (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(image.width, x1 + pad_x),
                min(image.height, y1 + pad_y),
            )
            return found, storage.read_crop(image.file_path, box, max_side)
        except storage.StorageError:
            logger.warning("Open search: no picture of the frame")
            return found, None

    # ---- Step 1: the names ----

    def _vectors_of_labels(self, labels: list[str]) -> np.ndarray:
        """Return the sentence vector of each name. A name is encoded one time."""
        with self._lock:
            missing: list[str] = [l for l in labels if l not in self._label_vectors]
            if missing:
                for label, vector in zip(
                    missing, self._engine.encode_texts(missing), strict=True
                ):
                    self._label_vectors[label] = vector
            return np.stack([self._label_vectors[l] for l in labels])

    def _by_label(
        self, phrase: str, rows: list[OpenRegion], min_score: float | None = None
    ) -> tuple[OpenRegion, str, float] | None:
        labels: list[str] = sorted({row.label for row in rows})
        scores: np.ndarray = (
            self._vectors_of_labels(labels) @ self._engine.encode_texts([phrase])[0]
        )
        best: float = float(scores.max())
        needed: float = self._config.label_min_score if min_score is None else min_score
        if best < needed:
            return None
        floor: float = max(needed, best - self._config.label_tier)
        fits: dict[str, float] = {
            label: float(score)
            for label, score in zip(labels, scores, strict=True)
            if score >= floor
        }
        for row in rows:  # Newest frame first
            if row.label in fits:
                return row, "label", fits[row.label]
        return None

    # ---- Step 3: the rewrite ----

    def _by_rewrite(
        self, phrase: str, rows: list[OpenRegion]
    ) -> tuple[OpenRegion, str, float] | None:
        """Rewrite the words into names with the language model, and look for each name (step 1).

        The best score wins. A name from the model is never shown to the user: only the name of a stored box is.
        """
        if (
            not self._engine.language_enabled
            or len(phrase.split()) < self._config.rewrite_min_words
        ):
            return None  # A name is not a description. The model would give a neighbour of the name.
        best: tuple[OpenRegion, str, float] | None = None
        for name in self._engine.rewrite_query(phrase):
            hit = self._by_label(name, rows, self._config.rewrite_min_score)
            if hit is not None and (best is None or hit[2] > best[2]):
                best = hit
        return None if best is None else (best[0], "rewrite", best[2])

    # ---- Step 2: the pictures ----

    def _neutral_vectors(self) -> np.ndarray:
        with self._lock:
            if self._neutral is None:
                self._neutral = self._engine.embed_open_phrases(
                    list(self._config.neutral_phrases)
                )
            return self._neutral

    def _by_picture(
        self, phrase: str, rows: list[OpenRegion]
    ) -> tuple[OpenRegion, str, float] | None:
        crops: np.ndarray = db.open_regions.get_vectors([row.id for row in rows])
        asked: np.ndarray = self._engine.embed_open_phrases([phrase])
        probability: np.ndarray = self._engine.open_pair_scores(crops, asked)[:, 0]
        neutral: np.ndarray = self._engine.open_pair_scores(
            crops, self._neutral_vectors()
        ).max(axis=1)
        margin: np.ndarray = probability - neutral
        fits: np.ndarray = (probability >= self._config.visual_min_probability) & (
            margin >= self._config.visual_margin
        )
        if not fits.any():
            return None
        best: int = int(np.argmax(np.where(fits, probability, -1.0)))
        return rows[best], "visual", float(probability[best])

    def _answer(self, row: OpenRegion, score: float, moment: datetime) -> Answer:
        seen_at: datetime = row.image.captured_at
        return Answer(
            object_id=0,
            found=True,
            source=AnswerSource.OPEN,
            seen_at=seen_at,
            age_seconds=max(0, int((moment - seen_at).total_seconds())),
            image_id=row.image_id,
            box=(row.x0, row.y0, row.x1, row.y1),
            similarity=score,
            strong=False,
            in_hand=False,
            stale=False,
            complete=True,
        )
