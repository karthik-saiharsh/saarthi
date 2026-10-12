"""The search: where did the system last see an object of the owner? (stage Q1)

`SearchService.find` answers in this order:

1. The record `last_seen` of the object, if it is fresh (younger than
   `search.stale_after_hours`).
2. The archive of frames. The search goes back in time, one window at a time,
   newest first. It matches the stored regions of each frame with the same
   rule as the ingestion (the mode of `[matching]` in `ai/config.toml`). In the mode `evidence`,
   a frame is read and scored only if one of its regions is a candidate for the object, and at most
   `search.evidence_max_frames` frames are scored. The first frame with an
   accepted match is the answer. A match that is not strong is checked against
   the reference photos of the object (stage I5). If the check passes, the match counts as
   strong. If not, it stays weak, so the caller can say "maybe".
3. The old record, marked `stale`, if the archive has nothing newer.
4. No answer: the object was not seen.

The search never writes data. The log shows ids and counts, never names.
"""

import dataclasses
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from itertools import groupby

import numpy as np

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import AiError
from ai.types import Box, MatchObject, MatchRegion, Sighting, VerifyResult
from backend.errors import NotFoundError, ObjectNotReadyError
from backend.memory import load_match_objects
from backend.references import ReferenceVectors
from backend.settings import BackendConfig, SearchConfig
from backend.verification import ReferenceVerifier
from db.schema import (
    EnrolledObject,
    Image,
    LastSeen,
    ObjectStatus,
    Region,
    utc_now,
)

logger: logging.Logger = logging.getLogger(__name__)


PICTURE_VIEWS: tuple[str, str] = ("crop", "frame")


class AnswerSource(StrEnum):
    """Where an answer comes from."""

    LAST_SEEN = "last_seen"  # The record of the memory
    ARCHIVE = "archive"  # A search in the stored regions of the frames
    OPEN = "open"  # A box of the broad detector. The object is not registered


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
    vector: np.ndarray  # The vector of the region. It orders the reference photos


class SearchService:
    """Answers the question "where is my object?". It is safe to call from several threads."""

    def __init__(
        self,
        config: BackendConfig,
        engine: AiEngine,
        references: ReferenceVectors | None = None,
    ) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: SearchConfig = config.search
        self._engine: AiEngine = engine
        self._references: ReferenceVectors = references or ReferenceVectors(engine)
        self._verifier: ReferenceVerifier = ReferenceVerifier(
            engine, self._config.verify_max_references
        )

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

    def get_picture(
        self,
        object_id: int,
        view: str = "crop",
        max_side: int = 512,
        now: datetime | None = None,
    ) -> bytes:
        """Return a JPEG of the answer: the object with space around it (`crop`) or the whole frame (`frame`).

        The longest side is at most `max_side`. Raise `NotFoundError` if the
        object was not seen, or its frame is gone. The other errors are those of `find`.
        """
        assert view in PICTURE_VIEWS, f"view must be one of {PICTURE_VIEWS}"
        assert isinstance(max_side, int) and max_side > 0, "max_side must be positive"
        picture: bytes | None = self._render(self.find(object_id, now), view, max_side)
        if picture is None:
            raise NotFoundError("The object was not seen")
        return picture

    def find_with_picture(
        self, object_id: int, max_side: int = 512, now: datetime | None = None
    ) -> tuple[Answer, bytes | None]:
        """Find the object and make the `crop` picture of the answer with one search.

        The picture is None if the object was not seen, or if the frame cannot be
        read: the answer is still true, so the caller can show it without a picture.
        """
        assert isinstance(max_side, int) and max_side > 0, "max_side must be positive"
        answer: Answer = self.find(object_id, now)
        try:
            return answer, self._render(answer, "crop", max_side)
        except (storage.StorageError, NotFoundError):
            logger.warning("Search for object %d: no picture of the frame", object_id)
            return answer, None

    def _render(self, answer: Answer, view: str, max_side: int) -> bytes | None:
        """Cut the picture of an answer from its frame. Return None if there is no answer."""
        if not answer.found or answer.image_id is None or answer.box is None:
            return None
        image: Image | None = db.get_image(answer.image_id)
        if image is None:
            raise NotFoundError("The frame of the answer is gone")
        box: tuple[int, int, int, int] = (0, 0, image.width, image.height)
        if view == "crop":
            box = self._padded(answer.box, image.width, image.height)
        return storage.read_crop(image.file_path, box, max_side)

    # ---- Helpers ----

    def _padded(
        self, box: tuple[int, int, int, int], width: int, height: int
    ) -> tuple[int, int, int, int]:
        """Grow the box on each side by `search.crop_padding` of its size. Keep it inside the frame."""
        x0, y0, x1, y1 = box
        pad_x: int = round((x1 - x0) * self._config.crop_padding)
        pad_y: int = round((y1 - y0) * self._config.crop_padding)
        return (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(width, x1 + pad_x),
            min(height, y1 + pad_y),
        )

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

        Return the hit (or None) and False if the search stopped at `max_frames` or `evidence_max_frames`.
        """
        evidence: bool = self._engine.matching_mode == "evidence"
        objects: list[MatchObject] = load_match_objects(
            self._engine, self._references if evidence else None
        )
        if all(o.object_id != object_id for o in objects):
            return None, True  # No views of the current model: nothing to compare
        model: str = self._engine.embedding_model
        window: timedelta = timedelta(minutes=self._config.window_minutes)
        checked: int = 0
        scored: int = 0
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
                frame_regions: list[Region] = list(group)
                hit: _Hit | None
                if evidence:
                    chosen: list[int] = self._candidates(
                        object_id, objects, frame_regions
                    )
                    if not chosen:
                        continue
                    if scored >= self._config.evidence_max_frames:
                        logger.warning(
                            "The search scored %d frame(s) and stopped", scored
                        )
                        return None, False
                    scored += 1
                    hit = self._score_frame(object_id, objects, frame_regions, chosen)
                else:
                    hit = self._match_frame(object_id, objects, frame_regions)
                if hit is not None:
                    return self._checked(object_id, hit), True
            until = since
        logger.info("The search compared %d frame(s) and found nothing", checked)
        return None, True

    def _checked(self, object_id: int, hit: _Hit) -> _Hit:
        """Check a weak hit against the reference photos (stage I5). A hit that passes counts as strong.

        A strong hit, a switched-off check, a frame that cannot be read, or an error of the
        check gives the hit back as it is. The check is an extra, so it never fails a search.
        """
        if (
            hit.sighting.strong
            or not self._config.verify_weak
            or not self._verifier.enabled
        ):
            return hit
        image: Image | None = db.get_image(hit.image_id)
        if image is None:
            return hit
        try:
            picture = self._engine.open_image(storage.read_image(image.file_path))
        except (AiError, storage.StorageError):
            logger.warning("Search for object %d: the frame cannot be read", object_id)
            return hit
        result: VerifyResult | None = self._verifier.verify(
            object_id, hit.vector, picture, Box(*hit.box)
        )
        if result is None or not result.verified:
            return hit
        logger.info(
            "Search for object %d: a weak hit passed the check (%d matching keypoints)",
            object_id,
            result.matches,
        )
        return dataclasses.replace(
            hit, sighting=dataclasses.replace(hit.sighting, strong=True)
        )

    def _candidates(
        self, object_id: int, objects: list[MatchObject], regions: list[Region]
    ) -> list[int]:
        """Return the regions of a frame that the evidence mode must score for the object (no frame is read)."""
        target: list[MatchObject] = [o for o in objects if o.object_id == object_id]
        vectors: np.ndarray = db.get_region_vectors([r.id for r in regions])
        return self._engine.evidence_regions(
            vectors, [MatchRegion(tuple(r.phrases), True) for r in regions], target
        )

    def _score_frame(
        self,
        object_id: int,
        objects: list[MatchObject],
        regions: list[Region],
        chosen: list[int],
    ) -> _Hit | None:
        """Score the chosen regions of one frame (mode `evidence`). Return the hit for the object, or None.

        A frame that cannot be read gives no hit. The search goes on with the next frame.
        """
        vectors: np.ndarray = db.get_region_vectors([r.id for r in regions])
        try:
            picture = self._engine.open_image(
                storage.read_image(regions[0].image.file_path)
            )
            crops: np.ndarray = self._engine.embed_evidence_crops(
                picture,
                [Box(r.x0, r.y0, r.x1, r.y1) for r in (regions[i] for i in chosen)],
            )
            sightings: list[Sighting] = self._engine.score_evidence(
                chosen, vectors, crops, objects
            )
        except (AiError, storage.StorageError):
            logger.warning(
                "Search for object %d: frame %d cannot be scored",
                object_id,
                regions[0].image_id,
            )
            return None
        return self._hit_for(object_id, sightings, regions, vectors)

    def _match_frame(
        self, object_id: int, objects: list[MatchObject], regions: list[Region]
    ) -> _Hit | None:
        """Match the regions of one frame. Return the hit for the object, or None."""
        vectors = db.get_region_vectors([r.id for r in regions])
        sightings: list[Sighting] = self._engine.match_regions(
            vectors, [MatchRegion(tuple(r.phrases), True) for r in regions], objects
        )
        return self._hit_for(object_id, sightings, regions, vectors)

    @staticmethod
    def _hit_for(
        object_id: int,
        sightings: list[Sighting],
        regions: list[Region],
        vectors: np.ndarray,
    ) -> _Hit | None:
        """Return the hit of the object among the sightings of one frame, or None."""
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
                vectors[sighting.region_index],
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
