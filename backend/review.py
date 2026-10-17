"""Is this your object? The carer checks boxes of recent frames (plan part D2).

The reference photos decide more than any signal: one view from the place of use found a headset that two photos
on a bed could not find. So the carer sees the boxes of the last `enrollment.review_hours` that look most like an
object, and answers:

- **Yes.** The frame is copied into the storage and becomes a confirmed reference photo of the object, with the box
  of the frame. All objects are calibrated again. The carer can remove the photo on the page, as any other photo.
- **No.** The box becomes a negative of the object (`db.reviews`). The matching mode `keys` refuses a box that is
  closer to a negative than to the object, and the box is not asked again.

The boxes come from the stored crop vectors (`db.crop_vectors`). No model runs to choose them. One box is shown for
each frame. A box that was answered, or that is a strong sighting of the object, is not shown.
The log shows ids and counts only.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.types import Box, MatchObject
from backend.enrollment import EnrollmentService
from backend.errors import InvalidInputError, LimitReachedError, NotFoundError
from backend.memory import load_match_objects
from backend.references import ReferenceVectors
from backend.settings import BackendConfig, EnrollmentConfig
from db.schema import Image, Region, utc_now

logger: logging.Logger = logging.getLogger(__name__)

MESSAGE_NOT_READY: str = (
    "The object must be ready (enough confirmed photos) before it can be checked."
)
MESSAGE_BOX: str = "This box cannot be checked any more."
# The most stored boxes that one list reads.
MAX_BOXES: int = 20000


@dataclass(frozen=True)
class ReviewCandidate:
    """A box that may show the object."""

    region_id: int
    image_id: int
    box: tuple[int, int, int, int]
    age_seconds: int
    score: float  # The SigLIP agreement with the reference photos


class ReviewService:
    """Chooses the boxes to check and keeps the answers. It is safe to call from several threads."""

    def __init__(
        self,
        config: BackendConfig,
        engine: AiEngine,
        enrollment: EnrollmentService,
        references: ReferenceVectors | None = None,
    ) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        assert isinstance(enrollment, EnrollmentService), "enrollment needed"
        self._config: EnrollmentConfig = config.enrollment
        self._padding: float = (
            config.search.crop_padding
        )  # The space around a picture, as on the find page
        self._engine: AiEngine = engine
        self._enrollment: EnrollmentService = enrollment
        self._references: ReferenceVectors = references or ReferenceVectors(engine)

    @property
    def enabled(self) -> bool:
        """True if the boxes have crop vectors to compare: the scene model is loaded."""
        return self._engine.scene_enabled

    def candidates(
        self, object_id: int, now: datetime | None = None
    ) -> list[ReviewCandidate]:
        """Return the boxes to check for an object, the most similar first. Raise `NotFoundError` or `InvalidInputError`."""
        assert isinstance(object_id, int) and object_id > 0, "object_id must be > 0"
        if not self.enabled:
            return []
        target: MatchObject = self._target(object_id)
        moment: datetime = now if now is not None else utc_now()
        since: datetime = moment - timedelta(hours=self._config.review_hours)
        regions: list[Region] = db.crop_vectors.list_keyed_regions(
            since,
            moment,
            self._engine.open_vector_model,
            self._engine.key_model,
            MAX_BOXES,
        )
        skip: set[int] = db.reviews.answered_region_ids(
            object_id
        ) | db.sightings.strong_region_ids(object_id, since)
        regions = [r for r in regions if r.id not in skip]
        if not regions:
            return []
        ids: list[int] = [r.id for r in regions]
        scores: np.ndarray = self._engine.review_scores(
            db.crop_vectors.get_crop_vectors(ids),
            db.crop_vectors.get_key_vectors(ids),
            target,
        )
        best: dict[int, tuple[float, Region]] = {}  # One box for each frame
        for region, score in zip(regions, scores, strict=True):
            if score < self._config.review_min:
                continue
            kept = best.get(region.image_id)
            if kept is None or score > kept[0]:
                best[region.image_id] = (float(score), region)
        chosen = sorted(best.values(), key=lambda item: -item[0])[
            : self._config.review_max
        ]
        logger.info(
            "Review of object %d: %d box(es) of %d compared",
            object_id,
            len(chosen),
            len(ids),
        )
        return [
            ReviewCandidate(
                r.id,
                r.image_id,
                (r.x0, r.y0, r.x1, r.y1),
                max(0, int((moment - r.image.captured_at).total_seconds())),
                score,
            )
            for score, r in chosen
        ]

    def picture(self, region_id: int, max_side: int = 512) -> bytes:
        """Return a JPEG of the box with a little space around it. Raise `NotFoundError` if the box or the frame is gone."""
        region, image = self._region(region_id)
        pad_x: int = round((region.x1 - region.x0) * self._padding)
        pad_y: int = round((region.y1 - region.y0) * self._padding)
        box: tuple[int, int, int, int] = (
            max(0, region.x0 - pad_x),
            max(0, region.y0 - pad_y),
            min(image.width, region.x1 + pad_x),
            min(image.height, region.y1 + pad_y),
        )
        try:
            return storage.read_crop(image.file_path, box, max_side)
        except storage.StorageError as error:
            raise NotFoundError(MESSAGE_BOX) from error

    def answer(self, object_id: int, region_id: int, yes: bool) -> None:
        """Keep the answer of the carer. Raise `NotFoundError`, `InvalidInputError` or `LimitReachedError`."""
        assert isinstance(yes, bool), "yes must be a bool"
        self._target(object_id)
        region, image = self._region(region_id)
        if region.crop_model is None or region.key_model is None:
            raise NotFoundError(MESSAGE_BOX)
        crop: np.ndarray = db.crop_vectors.get_crop_vectors([region.id])[0]
        key: np.ndarray = db.crop_vectors.get_key_vectors([region.id])[0]
        if yes:
            self._add_photo(object_id, region, image)
        db.reviews.add_answer(
            object_id,
            region.id,
            "yes" if yes else "no",
            crop,
            region.crop_model,
            key,
            region.key_model,
        )

    # ---- Helpers ----

    def _add_photo(self, object_id: int, region: Region, image: Image) -> None:
        """Copy the frame into the storage as a confirmed reference photo with the box. Then calibrate again."""
        if len(db.gallery.list_photos(object_id)) >= self._config.max_photos_per_object:
            raise LimitReachedError("The object has the most photos already")
        data: bytes = storage.read_image(image.file_path)
        picture = self._engine.open_image(data)
        box: Box = Box(region.x0, region.y0, region.x1, region.y1)
        views = self._engine.embed_views(picture, box)
        prepared: storage.PreparedImage = storage.prepare_image(data)
        path: str = storage.save_image(prepared)
        try:
            db.gallery.add_confirmed_photo(
                object_id,
                path,
                prepared.sha256,
                prepared.width,
                prepared.height,
                (box.x0, box.y0, box.x1, box.y1),
                views,
                self._engine.embedding_model,
            )
        except db.DbError:
            storage.delete_images([path])
            raise
        self._enrollment.recalibrate_all()
        logger.info("Object %d has a new reference photo from a frame", object_id)

    def _target(self, object_id: int) -> MatchObject:
        if db.gallery.get_object(object_id) is None:
            raise NotFoundError("The object does not exist")
        for obj in load_match_objects(
            self._engine, self._references, look_alikes_need_references=True
        ):
            if obj.object_id == object_id and obj.tracked:
                return obj
        raise InvalidInputError(MESSAGE_NOT_READY)

    @staticmethod
    def _region(region_id: int) -> tuple[Region, Image]:
        assert isinstance(region_id, int) and region_id > 0, "region_id must be > 0"
        region: Region | None = db.get_region(region_id)
        if region is None:
            raise NotFoundError(MESSAGE_BOX)
        return region, region.image
