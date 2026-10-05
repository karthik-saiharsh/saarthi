"""The memory of the system: matching, the sighting log and the last-seen record (stages I4 and I6).

For each frame, the worker gives this service the regions and their vectors.
The service does these steps:

1. It loads the `ready` objects of the gallery with their views.
2. It matches the regions with the objects (`AiEngine.match_regions`).
3. It checks each weak match against the reference photos of the object (stage I5).
4. It writes each accepted match to the sighting log.
5. It changes the last-seen record of an object only by the rule below.

The rule (hysteresis) stops a single false match from replacing a correct answer:

- A **strong** sighting changes the record at once. A weak sighting that passes the check (stage I5) also counts as strong.
- A **weak** sighting changes the record only if the object has `hysteresis_hits`
  weak sightings in different frames within `hysteresis_window_seconds`. The new sighting counts.
  A strong sighting does not count as a hit for a later weak one. So a weak false match that follows a true strong
  sighting cannot replace the record alone.

The check can only promote a weak sighting. A sighting that fails stays weak, and an error of the check changes nothing.
The count reads the sighting log. A restart loses nothing.
The time of a sighting is the time at which the server received the frame.
Names and phrases are never written to the log.
"""

import dataclasses
import logging
from collections.abc import Sequence
from datetime import timedelta

import numpy as np
from PIL import Image as PilImage

import db.api as db
from ai.api import AiEngine
from ai.types import (
    Candidate,
    MatchObject,
    MatchRegion,
    Sighting,
    VerifyResult,
)
from backend.settings import MemoryConfig
from backend.verification import ReferenceVerifier
from db.schema import EnrolledObject, Image, ObjectStatus, Region

logger: logging.Logger = logging.getLogger(__name__)


def _to_match_object(obj: EnrolledObject, views: np.ndarray) -> MatchObject:
    return MatchObject(
        object_id=obj.id,
        tracked=obj.tracked,
        phrases=(obj.category_phrase, *obj.synonyms),
        views=views,
        tau=float(obj.tau),
        tau_strong=float(obj.tau_strong),
        margin_min=float(obj.margin_min),
    )


def load_match_objects(engine: AiEngine) -> list[MatchObject]:
    """Return the `ready` objects with the views of the current model.

    The memory and the search use it. An object with no view of the current
    model is left out. Its calibration is from another model, so its
    thresholds do not fit.
    """
    assert isinstance(engine, AiEngine), "engine must be an AiEngine"
    model: str = engine.embedding_model
    objects: list[MatchObject] = []
    for obj in db.gallery.list_objects():
        if (
            obj.status != ObjectStatus.READY.value
            or obj.tau is None
            or obj.tau_strong is None
            or obj.margin_min is None
        ):
            continue
        views, _ = db.gallery.load_views(obj.id, model)
        if views.shape[0] == 0:
            logger.debug("Object %d has no views of the current model", obj.id)
            continue
        objects.append(_to_match_object(obj, views))
    return objects


class MemoryService:
    """Matches the regions of a frame and keeps the last-seen records."""

    def __init__(self, config: MemoryConfig, engine: AiEngine) -> None:
        assert isinstance(config, MemoryConfig), "config must be a MemoryConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: MemoryConfig = config
        self._engine: AiEngine = engine
        self._verifier: ReferenceVerifier = ReferenceVerifier(
            engine, config.verify_max_references
        )

    def remember(
        self,
        image: Image,
        candidates: Sequence[Candidate],
        regions: Sequence[Region],
        vectors: np.ndarray,
        picture: PilImage.Image | None = None,
    ) -> int:
        """Match the regions of one frame and update the memory. Return the number of sightings.

        `candidates`, `regions` and `vectors` have the same order. The regions
        must be stored, with their vectors, before this call. `picture` is the
        decoded frame. Without it, no weak sighting is checked.
        Raise `AiError` or `DbError` for a problem of the data. The worker marks the image `failed`.
        """
        assert len(candidates) == len(regions) == len(vectors), "one row per region"
        if not regions:
            return 0
        objects: list[MatchObject] = self.load_objects()
        if not objects:
            return 0
        found: list[Sighting] = self._engine.match_regions(
            vectors,
            [MatchRegion(tuple(c.phrases), c.quality_ok) for c in candidates],
            objects,
        )
        recorded: int = 0
        for sighting in found:
            sighting = self._verified(sighting, candidates, vectors, picture)
            if self._record(image, sighting, candidates, regions):
                recorded += 1
        return recorded

    def _verified(
        self,
        sighting: Sighting,
        candidates: Sequence[Candidate],
        vectors: np.ndarray,
        picture: PilImage.Image | None,
    ) -> Sighting:
        """Check a weak sighting (stage I5). Return it as a strong sighting if the check passes.

        A strong sighting, or a frame without picture, or a switched-off check
        gives the sighting back as it is. An error of the check does the same:
        the check is an extra, so it must not fail the image.
        """
        if (
            sighting.strong
            or picture is None
            or not self._config.verify_weak
            or not self._verifier.enabled
        ):
            return sighting
        result: VerifyResult | None = self._verifier.verify(
            sighting.object_id,
            vectors[sighting.region_index],
            picture,
            candidates[sighting.region_index].box,
        )
        if result is None or not result.verified:
            return sighting
        logger.info(
            "A weak sighting of object %d passed the check (%d matching keypoints)",
            sighting.object_id,
            result.matches,
        )
        return dataclasses.replace(sighting, strong=True)

    def load_objects(self) -> list[MatchObject]:
        """Return the `ready` objects with the views of the current model."""
        return load_match_objects(self._engine)

    def _record(
        self,
        image: Image,
        sighting: Sighting,
        candidates: Sequence[Candidate],
        regions: Sequence[Region],
    ) -> bool:
        """Write one sighting. Return False if the object was deleted in the meantime."""
        candidate: Candidate = candidates[sighting.region_index]
        region: Region = regions[sighting.region_index]
        try:
            db.sightings.add_sighting(
                sighting.object_id,
                image.id,
                region.id,
                image.captured_at,
                sighting.similarity,
                sighting.margin,
                sighting.strong,
                candidate.in_hand,
            )
            if not self._may_change_record(image, sighting):
                return True
            box = candidate.box
            changed: bool = db.sightings.set_last_seen(
                sighting.object_id,
                image.id,
                (box.x0, box.y0, box.x1, box.y1),
                image.captured_at,
                sighting.similarity,
                sighting.margin,
                sighting.strong,
                candidate.in_hand,
            )
        except db.InvalidRecordError:
            # A carer can delete an object while the worker handles a frame.
            logger.warning(
                "Object %d no longer exists. Its sighting is dropped",
                sighting.object_id,
            )
            return False
        logger.info(
            "Sighting of object %d in image %d (%s). Last-seen record %s",
            sighting.object_id,
            image.id,
            "strong" if sighting.strong else "weak",
            "changed" if changed else "kept",
        )
        return True

    def _may_change_record(self, image: Image, sighting: Sighting) -> bool:
        """Apply the rule: a strong sighting, or enough weak ones in the window."""
        if sighting.strong:
            return True
        window = timedelta(seconds=self._config.hysteresis_window_seconds)
        # Only weak sightings count. A strong sighting that came just before must not help a weak false match.
        hits: int = db.sightings.count_recent(
            sighting.object_id,
            image.captured_at - window,
            image.captured_at,
            strong=False,
        )
        return hits >= self._config.hysteresis_hits
