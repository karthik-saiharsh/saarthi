"""The sighting log and the last-seen record of each enrolled object.

These helpers change rows only. The rule that decides when a sighting changes
the last-seen record is in the `backend` module (`backend/memory.py`).
"""

import logging
from datetime import datetime

from peewee import fn

from db.schema import LastSeen, Region, Sighting, database_proxy, utc_now
from db.support import translate_errors

logger: logging.Logger = logging.getLogger(__name__)


def _check_id(value: int, name: str) -> None:
    assert isinstance(value, int) and value > 0, f"{name} must be a positive integer"


def add_sighting(
    object_id: int,
    image_id: int,
    region_id: int,
    seen_at: datetime,
    similarity: float,
    margin: float,
    strong: bool,
    in_hand: bool,
) -> Sighting:
    """Add one accepted match to the log.

    Raise `InvalidRecordError` if the object, the image or the region does not exist.
    """
    for value, name in (
        (object_id, "object_id"),
        (image_id, "image_id"),
        (region_id, "region_id"),
    ):
        _check_id(value, name)
    assert isinstance(seen_at, datetime), "seen_at must be a datetime"
    assert isinstance(similarity, float) and isinstance(margin, float), "floats needed"
    assert isinstance(strong, bool) and isinstance(in_hand, bool), "bools needed"
    with translate_errors("add_sighting"):
        return Sighting.create(
            enrolled_object=object_id,
            image=image_id,
            region=region_id,
            seen_at=seen_at,
            similarity=similarity,
            margin=margin,
            strong=strong,
            in_hand=in_hand,
        )


def count_recent(
    object_id: int, since: datetime, until: datetime, strong: bool | None = None
) -> int:
    """Count the frames with a sighting of the object from `since` to `until` (both included).

    `strong` True or False counts only the strong or only the weak sightings. None counts all.

    A frame counts one time, also if it has more than one row.
    """
    _check_id(object_id, "object_id")
    assert isinstance(since, datetime) and isinstance(
        until, datetime
    ), "datetimes needed"
    assert since <= until, "since must not be after until"
    assert strong is None or isinstance(strong, bool), "strong must be bool or None"
    with translate_errors("count_recent"):
        query = Sighting.select(fn.COUNT(fn.DISTINCT(Sighting.image))).where(
            Sighting.enrolled_object == object_id,
            Sighting.seen_at >= since,
            Sighting.seen_at <= until,
        )
        if strong is not None:
            query = query.where(Sighting.strong == strong)
        return int(query.scalar() or 0)


def newest_weak(object_id: int, after: datetime, until: datetime) -> Sighting | None:
    """Return the newest weak sighting of the object with `after` < `seen_at` <= `until`, or None.

    The row has its region in `sighting.region` (the box). The search names it as a second, "maybe" place.
    """
    _check_id(object_id, "object_id")
    assert isinstance(after, datetime) and isinstance(until, datetime), "datetimes"
    with translate_errors("newest_weak"):
        return (
            Sighting.select(Sighting, Region)
            .join(Region)
            .where(
                Sighting.enrolled_object == object_id,
                Sighting.strong == 0,
                Sighting.seen_at > after,
                Sighting.seen_at <= until,
            )
            .order_by(Sighting.seen_at.desc(), Sighting.id.desc())
            .first()
        )


def get_last_seen(object_id: int) -> LastSeen | None:
    """Return the last-seen record of an object, or None."""
    _check_id(object_id, "object_id")
    with translate_errors("get_last_seen"):
        return LastSeen.get_or_none(LastSeen.enrolled_object == object_id)


def list_last_seen() -> list[LastSeen]:
    """Return the last-seen records of all objects, ordered by object id."""
    with translate_errors("list_last_seen"):
        return list(LastSeen.select().order_by(LastSeen.enrolled_object))


def set_last_seen(
    object_id: int,
    image_id: int,
    box: tuple[int, int, int, int],
    seen_at: datetime,
    similarity: float,
    margin: float,
    strong: bool,
    in_hand: bool,
) -> bool:
    """Make a sighting the last-seen record of the object.

    An older sighting does not replace a newer record. Return True if the record
    changed. Raise `InvalidRecordError` if the object or the image does not exist.
    """
    _check_id(object_id, "object_id")
    _check_id(image_id, "image_id")
    assert len(box) == 4 and all(isinstance(v, int) for v in box), "box needs 4 ints"
    assert isinstance(seen_at, datetime), "seen_at must be a datetime"
    assert isinstance(similarity, float) and isinstance(margin, float), "floats needed"
    assert isinstance(strong, bool) and isinstance(in_hand, bool), "bools needed"
    with translate_errors("set_last_seen"), database_proxy.atomic("IMMEDIATE"):
        current: LastSeen | None = LastSeen.get_or_none(
            LastSeen.enrolled_object == object_id
        )
        if current is not None and current.seen_at > seen_at:
            return False
        LastSeen.insert(
            enrolled_object=object_id,
            image=image_id,
            x0=box[0],
            y0=box[1],
            x1=box[2],
            y1=box[3],
            seen_at=seen_at,
            similarity=similarity,
            margin=margin,
            strong=strong,
            in_hand=in_hand,
            updated_at=utc_now(),
        ).on_conflict_replace().execute()
        return True
