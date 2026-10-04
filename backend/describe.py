"""The description of the place of an object, for the voice (stage Q2).

Given an answer of the search, `DescribeService.describe` makes a short sentence. The sentence says
when the object was seen, what it was on, and what kind of room it was in (see `docs/backend/Voice.md`).

The words come from fixed lists of the scene model (`ai/config.toml`, `[scene]`). The room is chosen
by a rule: the scores of the frame of the answer and of the frames near it in time are averaged.
The room is named only if enough frames agree and the score is high enough. Otherwise the sentence
says nothing about the room. A single frame can mislead: a close-up of a floor does not show the room.

The description is an extra. An error of the model or of a file never fails an answer: the sentence
then has the time only. The log shows counts and ids. It never shows a name or a word of the lists.
"""

import logging
from dataclasses import dataclass

from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import AiError
from ai.types import Box
from backend.search import Answer
from backend.settings import BackendConfig, DescribeConfig
from db.schema import Image

logger: logging.Logger = logging.getLogger(__name__)

_MINUTE: int = 60
_HOUR: int = 60 * _MINUTE
_DAY: int = 24 * _HOUR


@dataclass(frozen=True)
class RoomChoice:
    """The result of the room rule. `room` is None when the rule says nothing."""

    room: str | None
    score: (
        float  # The average score of the best room. Also set when the rule says nothing
    )
    agreeing: int  # The number of frames that have the best room as their best room
    frames: int  # The number of frames that the rule used


@dataclass(frozen=True)
class Description:
    """The sentence for the voice, and the words that it uses."""

    text: str
    surface: str | None  # For example "a table". None if the sentence does not name it
    room: str | None  # For example "a bedroom". None if the sentence does not name it


def time_phrase(age_seconds: int) -> str:
    """Return a plain time such as `12 minutes ago`."""
    assert isinstance(age_seconds, int) and age_seconds >= 0, "age must be 0 or more"
    if age_seconds < _MINUTE:
        return "just now"
    for unit, name in ((_DAY, "day"), (_HOUR, "hour"), (_MINUTE, "minute")):
        if age_seconds >= unit:
            count: int = age_seconds // unit
            return f"{count} {name}{'' if count == 1 else 's'} ago"
    raise AssertionError("unreachable")  # pragma: no cover


def choose_room(frames: list[dict[str, float]], config: DescribeConfig) -> RoomChoice:
    """Choose the room from the room scores of some frames, or say nothing.

    Each frame has the probability of each room. The rule has three parts. All must hold:
    enough frames, enough of them have the best room as their best room, and its average score is high enough.
    """
    assert isinstance(frames, list), "frames must be a list"
    if len(frames) < config.room_min_frames or not frames:
        return RoomChoice(None, 0.0, 0, len(frames))
    names: list[str] = list(frames[0])
    average: dict[str, float] = {
        room: sum(frame.get(room, 0.0) for frame in frames) / len(frames)
        for room in names
    }
    best: str = max(average, key=average.get)  # type: ignore[arg-type]
    agreeing: int = sum(max(frame, key=frame.get) == best for frame in frames)  # type: ignore[arg-type]
    named: bool = (
        agreeing / len(frames) >= config.room_agree_fraction
        and average[best] >= config.room_min_score
    )
    return RoomChoice(best if named else None, average[best], agreeing, len(frames))


def build_sentence(
    name: str, answer: Answer, surface: str | None, room: str | None
) -> str:
    """Make the sentence for the voice. A weak answer says that the system is not sure.

    `surface` and `room` have an article, for example "a table" and "a bedroom". None leaves the part out.
    """
    assert isinstance(name, str) and name.strip(), "name must not be empty"
    who: str = f"your {name.strip()}"
    if not answer.found or answer.age_seconds is None:
        text: str = f"I have not seen {who} lately."
        return text + (
            "" if answer.complete else " I could not look at all the pictures."
        )

    when: str = time_phrase(answer.age_seconds)
    sure: bool = bool(answer.strong)
    parts: list[str] = []
    if answer.in_hand:  # The place is the person, so the surface is not named
        parts.append(
            f"You were holding {who} {when}."
            if sure
            else f"I am not sure, but you may have been holding {who} {when}."
        )
    else:
        if answer.stale:
            parts.append(
                f"The last time that I saw {who} was {when}. That was a long time ago."
            )
        elif sure:
            parts.append(f"{who[0].upper()}{who[1:]} was last seen {when}.")
        else:
            parts.append(f"I am not sure, but I may have seen {who} {when}.")
        if surface:
            parts.append(f"It {'was' if sure else 'may have been'} on {surface}.")
    if room:
        parts.append(f"The room {'looks' if sure else 'may look'} like {room}.")
    if not sure:
        parts.append("Please look at the picture.")
    return " ".join(parts)


class DescribeService:
    """Describes the place of an answer. It is safe to call from several threads."""

    def __init__(self, config: BackendConfig, engine: AiEngine) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: DescribeConfig = config.describe
        self._engine: AiEngine = engine

    def describe(self, object_name: str, answer: Answer) -> Description:
        """Make the sentence for an answer. An error of the scene model, of a file or of the database gives a sentence without the place."""
        assert isinstance(answer, Answer), "answer must be an Answer"
        surface: str | None = None
        room: str | None = None
        if (
            answer.found
            and answer.image_id is not None
            and answer.box is not None
            and self._engine.scene_enabled
        ):
            try:
                surface, room = self._place(
                    answer.image_id, answer.box, bool(answer.in_hand)
                )
            except (AiError, storage.StorageError, db.DbError):
                logger.warning(
                    "The place of image %d could not be described", answer.image_id
                )
        return Description(
            build_sentence(object_name, answer, surface, room), surface, room
        )

    # ---- The place ----

    def _place(
        self, image_id: int, box: tuple[int, int, int, int], in_hand: bool
    ) -> tuple[str | None, str | None]:
        """Return the surface and the room of the frame of the answer. Each can be None."""
        frame: Image | None = db.get_image(image_id)
        if frame is None:
            return None, None
        picture: PilImage.Image = self._open(frame)
        surface: str | None = None
        if not in_hand:
            best_name, best_score = self._engine.score_surfaces(picture, Box(*box))[0]
            if best_score >= self._config.surface_min_score:
                surface = best_name
        return surface, self._room(frame, picture)

    def _open(self, frame: Image) -> PilImage.Image:
        return self._engine.open_image(storage.read_image(frame.file_path))

    def _room(self, frame: Image, picture: PilImage.Image) -> str | None:
        """Choose the room from the frame of the answer and the frames near it in time."""
        neighbours: list[Image] = db.scenes.list_neighbour_images(
            frame.id,
            self._config.neighbour_window_seconds,
            self._config.neighbour_max_frames,
        )
        frames: list[Image] = [frame, *neighbours]
        model: str = self._engine.scene_model
        saved: dict[int, dict[str, float]] = db.scenes.get_room_scores(
            [f.id for f in frames], model
        )
        scores: list[dict[str, float]] = []
        for candidate in frames:
            if candidate.id in saved:
                scores.append(saved[candidate.id])
                continue
            try:
                own: PilImage.Image = (
                    picture if candidate.id == frame.id else self._open(candidate)
                )
                rooms: dict[str, float] = self._engine.score_rooms(own)
            except (AiError, storage.StorageError):
                if candidate.id == frame.id:
                    raise  # Without the frame of the answer, there is no room
                logger.warning(
                    "The frame %d was left out of the room rule", candidate.id
                )
                continue
            self._save(candidate.id, model, rooms)
            scores.append(rooms)
        choice: RoomChoice = choose_room(scores, self._config)
        logger.info(
            "Room rule: %d frame(s), %d agree, %s",
            choice.frames,
            choice.agreeing,
            "room named" if choice.room else "no room named",
        )
        return choice.room

    @staticmethod
    def _save(image_id: int, model: str, rooms: dict[str, float]) -> None:
        """Keep the scores of a frame. The cache is an extra: a failure is logged and does not stop the description."""
        try:
            db.scenes.set_room_scores(image_id, model, rooms)
        except db.DbError:
            logger.warning("The room scores of image %d could not be saved", image_id)
