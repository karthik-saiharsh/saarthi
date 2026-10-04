"""The scene cache and the neighbouring frames, for the description of a place.

The description of the place of an object takes the room from the frames around the
frame of the answer (see `docs/backend/Voice.md`). The scene model makes the room
scores of a frame. These helpers keep the scores, so a frame is scored one time.
They change rows only. They do not run a model and do not read files.
"""

import logging
import math
from datetime import timedelta

from db.errors import InvalidRecordError
from db.schema import FrameScene, Image, ImageStatus, database_proxy, utc_now
from db.support import translate_errors

logger: logging.Logger = logging.getLogger(__name__)

# The most scores that one call reads. A description needs a few frames.
MAX_IDS: int = 500
# The most frames that the search of neighbours reads before it sorts them by their distance in time.
MAX_SCANNED: int = 2000
# How far the sum of the scores may be from 1. The scores are probabilities.
SUM_TOLERANCE: float = 0.01


def _check_id(value: int, name: str) -> None:
    assert isinstance(value, int) and value > 0, f"{name} must be a positive integer"


def get_room_scores(image_ids: list[int], model: str) -> dict[int, dict[str, float]]:
    """Return the saved room scores of these frames, made by `model`.

    A frame without scores, or with scores of another model, is not in the result.
    """
    assert isinstance(image_ids, list), "image_ids must be a list"
    assert len(image_ids) <= MAX_IDS, f"at most {MAX_IDS} ids"
    for image_id in image_ids:
        _check_id(image_id, "image id")
    assert isinstance(model, str) and model, "model name needed"
    if not image_ids:
        return {}
    with translate_errors("get_room_scores"):
        rows = FrameScene.select().where(
            FrameScene.image.in_(set(image_ids)), FrameScene.model == model
        )
        return {row.image_id: dict(row.rooms) for row in rows}


def set_room_scores(image_id: int, model: str, rooms: dict[str, float]) -> None:
    """Save the room scores of a frame. Scores that exist are replaced.

    `rooms` has the probability of each room. The values must be finite, not negative,
    and add up to 1. Raise `InvalidRecordError` for a wrong value, or if the frame does not exist.
    """
    _check_id(image_id, "image_id")
    assert isinstance(model, str) and model, "model name needed"
    assert isinstance(rooms, dict) and rooms, "rooms must be a non-empty dict"
    assert all(isinstance(k, str) and k for k in rooms), "the rooms must be named"
    assert all(
        isinstance(v, float) for v in rooms.values()
    ), "the scores must be floats"
    if not all(math.isfinite(v) and v >= 0.0 for v in rooms.values()):
        raise InvalidRecordError("A room score is not finite or is negative")
    if abs(sum(rooms.values()) - 1.0) > SUM_TOLERANCE:
        raise InvalidRecordError("The room scores must add up to 1")
    with translate_errors("set_room_scores"), database_proxy.atomic():
        FrameScene.insert(
            image=image_id, model=model, rooms=dict(rooms), computed_at=utc_now()
        ).on_conflict_replace().execute()


def list_neighbour_images(
    image_id: int, window_seconds: int, limit: int
) -> list[Image]:
    """Return the `done` frames that are near a frame in time, the nearest first.

    A frame is near if it was captured at most `window_seconds` before or after the
    frame (both ends included). The frame itself is not in the result. A frame at the
    same distance before and after: the older id comes first. An unknown frame gives an empty list.
    """
    _check_id(image_id, "image_id")
    assert isinstance(window_seconds, int) and window_seconds > 0, "window must be > 0"
    assert isinstance(limit, int) and limit > 0, "limit must be positive"
    with translate_errors("list_neighbour_images"):
        centre: Image | None = Image.get_or_none(Image.id == image_id)
        if centre is None:
            return []
        window: timedelta = timedelta(seconds=window_seconds)
        near: list[Image] = list(
            Image.select()
            .where(
                Image.status == ImageStatus.DONE.value,
                Image.id != image_id,
                Image.captured_at >= centre.captured_at - window,
                Image.captured_at <= centre.captured_at + window,
            )
            .order_by(Image.captured_at, Image.id)
            .limit(MAX_SCANNED)
        )
    near.sort(key=lambda image: (abs(image.captured_at - centre.captured_at), image.id))
    return near[:limit]
