"""The search: where did the system last see an object of the owner? (stage Q1)

`SearchService.find` answers in this order:

1. The record `last_seen` of the object, if it is fresh (younger than
   `search.stale_after_hours`).
2. The archive of frames. The search goes back in time, one window at a time,
   newest first. It matches the stored regions of each frame with the same
   rule as the ingestion (`AiEngine.match_regions`). The first frame with an
   accepted match is the answer. A match that is not strong stays marked as
   weak, so the caller can say "maybe".
3. The old record, marked `stale`, if the archive has nothing newer.
4. No answer: the object was not seen.

The search never writes data. The log shows ids and counts, never names.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from itertools import groupby

import db.api as db
from ai.api import AiEngine
from ai.types import MatchObject, MatchRegion, Sighting
from backend.errors import NotFoundError, ObjectNotReadyError
from backend.memory import load_match_objects
from backend.settings import BackendConfig, SearchConfig
from db.schema import EnrolledObject, LastSeen, ObjectStatus, Region, utc_now

logger: logging.Logger = logging.getLogger(__name__)


class AnswerSource(StrEnum):
    """Where an answer comes from."""

    LAST_SEEN = "last_seen"  # The record of the memory
    ARCHIVE = "archive"  # A search in the stored regions of the frames


@dataclass(frozen=True)
class Answer:
    """The result of a search. `found` is False when the object was not seen."""

    object_id: int
    found: bool
    source: AnswerSource | None
    seen_at: datetime | None  # Naive UTC time at which the server received the frame
    age_seconds: int | None  # The time between the frame and the search
    image_id: int | None
    box: tuple[int, int, int, int] | None  # Where the object is in the frame
    similarity: float | None
    strong: bool | None  # False: the match is weak. Say "maybe"
    in_hand: bool | None  # True: the person held the object
    stale: bool  # True: an old record, and the archive has nothing newer
    complete: bool  # False: the search stopped at `search.max_frames`


@dataclass(frozen=True)
class _Hit:
    image_id: int
    seen_at: datetime
    box: tuple[int, int, int, int]
    sighting: Sighting
    in_hand: bool


class SearchService:
    """Answers the question "where is my object?". It is safe to call from several threads."""

    def __init__(self, config: BackendConfig, engine: AiEngine) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: SearchConfig = config.search
        self._engine: AiEngine = engine

    def find(self, object_id: int, now: datetime | None = None) -> Answer:
        """Find the last sighting of an object.

        `now` is a naive UTC time. Raise `NotFoundError` if the object does not
        exist, and `ObjectNotReadyError` if it is a draft or a look-alike.
        """
        assert isinstance(object_id, int) and object_id > 0, "object_id must be > 0"
        assert now is None or isinstance(now, datetime), "now must be a datetime"
        moment: datetime = now if now is not None else utc_now()
        self._require_findable(object_id)

        record: LastSeen | None = db.sightings.get_last_seen(object_id)
        stale_after: timedelta = timedelta(hours=self._config.stale_after_hours)
        if record is not None and moment - record.seen_at <= stale_after:
            logger.info("Search for object %d: fresh record", object_id)
            return self._from_record(record, moment, stale=False, complete=True)

        lower: datetime = moment - timedelta(hours=self._config.max_scan_hours)
        if record is not None:
            lower = max(lower, record.seen_at)  # Only a newer sighting is of interest
        hit, complete = self._scan_archive(object_id, lower, moment)
        if hit is not None:
            logger.info("Search for object %d: found in the archive", object_id)
            return self._from_hit(object_id, hit, moment, complete)
        if record is not None:
            logger.info("Search for object %d: stale record", object_id)
            return self._from_record(record, moment, stale=True, complete=complete)
        logger.info(
            "Search for object %d: not seen (complete: %s)", object_id, complete
        )
        return Answer(
            object_id, False, None, None, None, None, None, None, None, None,
            False, complete,
        )  # fmt: skip

    # ---- Helpers ----

    @staticmethod
    def _require_findable(object_id: int) -> EnrolledObject:
        obj: EnrolledObject | None = db.gallery.get_object(object_id)
        if obj is None:
            raise NotFoundError("The object does not exist")
        if obj.status != ObjectStatus.READY.value or not obj.tracked:
            raise ObjectNotReadyError("The object cannot be searched")
        return obj

    def _scan_archive(
        self, object_id: int, lower: datetime, upper: datetime
    ) -> tuple[_Hit | None, bool]:
        """Look for the newest frame in (`lower`, `upper`] with a match for the object.

        Return the hit (or None) and False if the search stopped at `max_frames`.
        """
        objects: list[MatchObject] = load_match_objects(self._engine)
        if all(o.object_id != object_id for o in objects):
            return None, True  # No views of the current model: nothing to compare
        model: str = self._engine.embedding_model
        window: timedelta = timedelta(minutes=self._config.window_minutes)
        checked: int = 0
        until: datetime = upper
        while until > lower:
            since: datetime = max(until - window, lower)
            regions: list[Region] = db.list_searchable_regions(
                since, until, model, self._config.max_regions_per_window
            )
            for _, group in groupby(regions, key=lambda r: r.image_id):
                if checked >= self._config.max_frames:
                    logger.warning("The search stopped after %d frame(s)", checked)
                    return None, False
                checked += 1
                hit: _Hit | None = self._match_frame(object_id, objects, list(group))
                if hit is not None:
                    return hit, True
            until = since
        logger.info("The search compared %d frame(s) and found nothing", checked)
        return None, True

    def _match_frame(
        self, object_id: int, objects: list[MatchObject], regions: list[Region]
    ) -> _Hit | None:
        """Match the regions of one frame. Return the hit for the object, or None."""
        vectors = db.get_region_vectors([r.id for r in regions])
        sightings: list[Sighting] = self._engine.match_regions(
            vectors, [MatchRegion(tuple(r.phrases), True) for r in regions], objects
        )
        for sighting in sightings:
            if sighting.object_id != object_id:
                continue
            region: Region = regions[sighting.region_index]
            return _Hit(
                region.image_id,
                region.image.captured_at,
                (region.x0, region.y0, region.x1, region.y1),
                sighting,
                region.in_hand,
            )
        return None

    @staticmethod
    def _age(seen_at: datetime, now: datetime) -> int:
        return max(0, int((now - seen_at).total_seconds()))

    def _from_record(
        self, record: LastSeen, now: datetime, stale: bool, complete: bool
    ) -> Answer:
        return Answer(
            object_id=record.enrolled_object_id,
            found=True,
            source=AnswerSource.LAST_SEEN,
            seen_at=record.seen_at,
            age_seconds=self._age(record.seen_at, now),
            image_id=record.image_id,
            box=(record.x0, record.y0, record.x1, record.y1),
            similarity=record.similarity,
            strong=record.strong,
            in_hand=record.in_hand,
            stale=stale,
            complete=complete,
        )

    def _from_hit(
        self, object_id: int, hit: _Hit, now: datetime, complete: bool
    ) -> Answer:
        return Answer(
            object_id=object_id,
            found=True,
            source=AnswerSource.ARCHIVE,
            seen_at=hit.seen_at,
            age_seconds=self._age(hit.seen_at, now),
            image_id=hit.image_id,
            box=hit.box,
            similarity=hit.sighting.similarity,
            strong=hit.sighting.strong,
            in_hand=hit.in_hand,
            stale=False,
            complete=complete,
        )
