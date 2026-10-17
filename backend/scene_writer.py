"""The description of a frame: one sentence about the place of an object, written in the background.

The language model gets the picture and the facts that the system checked (the object, the surface, the room, and the
names of the broad detector that are sure enough). It writes one sentence. The model can invent details or repeat a wrong
name, so the backend checks **every word** of the sentence: a word must be a function word, a word of the facts, or a safe
word. A sentence with another word is refused, and the answer uses the template sentence of `backend/describe.py`.

The sentence has no time. The answer adds the time when the user asks. The writer runs when the worker has no frame to
process, because one sentence takes 10 to 17 s on a CPU. The text is stored for each frame and object
(`db/scenes.py`, table `frame_description`). The log shows ids and counts, never the sentence.
"""

import logging
import re

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import AiError
from backend.describe import DescribeService, description_model
from backend.settings import BackendConfig, DescribeConfig
from db.scenes import FrameToDescribe
from db.schema import Image, Region

logger: logging.Logger = logging.getLogger(__name__)

# Words without a claim about the scene. A sentence may use them with no support from the facts.
FUNCTION_WORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "with",
        "without",
        "on",
        "in",
        "at",
        "by",
        "of",
        "to",
        "from",
        "near",
        "next",
        "beside",
        "above",
        "below",
        "under",
        "over",
        "behind",
        "front",
        "between",
        "among",
        "around",
        "inside",
        "outside",
        "onto",
        "into",
        "against",
        "along",
        "across",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "it",
        "its",
        "it's",
        "there",
        "this",
        "that",
        "these",
        "those",
        "which",
        "has",
        "have",
        "had",
        "held",
        "holding",
        "hold",
        "holds",
        "sits",
        "sitting",
        "stands",
        "standing",
        "lies",
        "lying",
        "rests",
        "resting",
        "placed",
        "put",
        "kept",
        "left",
        "leaning",
        "hanging",
        "shown",
        "seen",
        "visible",
        "also",
        "not",
        "no",
        "some",
        "other",
        "another",
        "one",
        "two",
        "few",
        "several",
        "many",
        "nearby",
        "close",
        "together",
        "here",
        "very",
        "just",
        "only",
        "still",
        "all",
        "both",
        "each",
        "any",
        "more",
        "most",
        "as",
        "like",
        "looks",
        "look",
        "appears",
        "appear",
    ]
)
# The most names of the broad detector in the facts.
MAX_NAMES: int = 6


def words_of(text: str) -> list[str]:
    """Return the lower-case words of a text."""
    return re.findall(r"[a-z']+", text.lower())


def _known(word: str, allowed: set[str]) -> bool:
    """True if the word is allowed. A plural or a singular of an allowed word counts."""
    variants: set[str] = {word, word.rstrip("s"), word + "s", word + "es"}
    if word.endswith("es"):
        variants.add(word[:-2])
    return bool(variants & allowed)


def check_sentence(text: str, allowed: set[str], max_words: int) -> str | None:
    """Return None if the sentence passes, else the reason why it is refused (a fixed text, never a word of the sentence).

    `allowed` has the words of the facts and the safe words. The function words are always allowed.
    """
    assert isinstance(text, str), "text must be a string"
    if not text.strip():
        return "empty"
    tokens: list[str] = words_of(text)
    if not tokens or len(text.split()) > max_words:
        return "too long"
    if re.search(r"\d", text):
        return "a digit"
    permitted: set[str] = allowed | FUNCTION_WORDS
    if any(not _known(token, permitted) for token in tokens):
        return "a word that is not in the facts"
    return None


class SceneWriter:
    """Writes the sentence for the next last-seen frame that has none. It is safe to call from one thread."""

    def __init__(
        self, config: BackendConfig, engine: AiEngine, describe: DescribeService
    ) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        assert isinstance(
            describe, DescribeService
        ), "describe must be a DescribeService"
        self._config: DescribeConfig = config.describe
        self._engine: AiEngine = engine
        self._describe: DescribeService = describe
        # Frames that failed in this run. They are not tried again until the next start.
        self._skipped: set[tuple[int, int]] = set()

    @property
    def enabled(self) -> bool:
        return self._engine.language_enabled and self._config.write

    def write_next(self) -> bool:
        """Write the sentence for one frame. Return True if a frame was handled (the sentence can be refused).

        Return False if there is nothing to do, or if the work failed. An error is logged and never raised.
        """
        if not self.enabled:
            return False
        model: str = description_model(self._config, self._engine)
        try:
            todo: list[FrameToDescribe] = [
                item
                for item in db.scenes.list_frames_to_describe(model, 20)
                if (item.image_id, item.object_id) not in self._skipped
            ]
            if not todo:
                return False
            item: FrameToDescribe = todo[0]
            try:
                self._write(item, model)
            except (AiError, storage.StorageError, db.DbError) as error:
                logger.warning(
                    "The description of image %d failed: %s",
                    item.image_id,
                    type(error).__name__,
                )
                self._skipped.add((item.image_id, item.object_id))
                return False
            return True
        except db.DbError:
            logger.exception("The list of frames to describe could not be read")
            return False

    def _write(self, item: FrameToDescribe, model: str) -> None:
        frame: Image | None = db.get_image(item.image_id)
        if frame is None:
            return
        picture = self._engine.open_image(storage.read_image(frame.file_path))
        surface, room = self._describe.place(item.image_id, item.box, item.in_hand)
        names: list[str] = self._sure_names(frame.id, item.object_name)
        facts: list[str] = self._facts(item, surface, room, names)
        allowed: set[str] = set(self._config.safe_words)
        for fact in [item.object_name, surface or "", room or "", *names]:
            allowed.update(words_of(fact))
        if item.in_hand:
            allowed.update({"hand", "hands"})
        text: str = self._engine.describe_scene(picture, facts)
        reason: str | None = check_sentence(text, allowed, self._config.max_words)
        db.scenes.set_description(
            item.image_id, item.object_id, model, text if reason is None else "", facts
        )
        logger.info(
            "Description of image %d: %s",
            item.image_id,
            "accepted" if reason is None else f"refused ({reason})",
        )

    def _sure_names(self, image_id: int, object_name: str) -> list[str]:
        """Return the names of the broad detector that are sure enough, without a repeat of the object."""
        boxes: list[Region] = db.get_broad_regions(image_id)
        names: list[str] = []
        known: set[str] = set(words_of(object_name))
        for box in boxes:
            if box.det_score < self._config.name_min_confidence:
                continue
            label: str = box.label.strip().lower()
            if label in names or set(words_of(label)) <= known:
                continue
            names.append(label)
        return names[:MAX_NAMES]

    @staticmethod
    def _facts(
        item: FrameToDescribe, surface: str | None, room: str | None, names: list[str]
    ) -> list[str]:
        facts: list[str] = [f"Object: {item.object_name}"]
        if item.in_hand:
            facts.append("A hand holds the object")
        if surface:
            facts.append(f"It is on: {surface}")
        if room:
            facts.append(f"The room looks like: {room}")
        if names:
            facts.append(
                f"Other things in the photo (names from a detector, they can be wrong): {', '.join(names)}"
            )
        return facts
