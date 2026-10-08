"""The worker: takes `pending` reference photos and images and runs the ai stages on them.

One item at a time. A reference photo of an enrolled object comes first: the
worker finds the object (stage E2) and makes its views (stage E3).
For each frame the worker does these steps:

1. Read the stored file and decode it.
2. Quality check. Save the blur score and the hash.
3. Reject a blurry image and a near duplicate of a recent image.
4. Detect candidate boxes.
5. Make one vector for each box. This comes before any region row is written,
   so a model error leaves no half-written regions.
6. Add the regions and store the vectors.
7. Match the regions with the objects and update the memory (stages I4 and I6).
8. Broad detection: find the objects that nobody registered and store them (see `docs/backend/Open Search.md`).
   This step is an extra. Its errors are logged and never fail the image.
9. Mark the image `done`.

Any error marks the image `failed`. The image is not tried again. The worker
goes on with the next image.
"""

import logging
import threading
from collections.abc import Callable
from datetime import timedelta

import numpy as np
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import AiError
from ai.types import Box, Candidate
from backend.memory import MemoryService
from backend.settings import BackendConfig, WorkerConfig
from db.schema import EnrolledObject, Image, ReferencePhoto, RegionSource, utc_now

logger: logging.Logger = logging.getLogger(__name__)

REASON_BLURRY: str = "too blurry"
REASON_DUPLICATE: str = "near duplicate of a recent image"
# The reason of a reference photo where the detector finds nothing. The UI shows a text for it.
REASON_OBJECT_NOT_FOUND: str = "object not found"


def _iou(a: Box, b: Box) -> float:
    """Overlap of two boxes: the area of the overlap divided by the area of both together."""
    width: int = min(a.x1, b.x1) - max(a.x0, b.x0)
    height: int = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0 or height <= 0:
        return 0.0
    inside: int = width * height
    return inside / (a.area + b.area - inside)


def _to_new_region(candidate: Candidate) -> db.NewRegion:
    box = candidate.box
    return db.NewRegion(
        box.x0,
        box.y0,
        box.x1,
        box.y1,
        candidate.det_score,
        RegionSource(candidate.source),
        list(candidate.phrases),
        candidate.in_hand,
        candidate.quality_ok,
    )


class Worker:
    """Runs the ai stages on `pending` images. Use `start` for a background thread."""

    def __init__(
        self,
        config: BackendConfig,
        engine: AiEngine,
        idle_task: Callable[[], bool] | None = None,
    ) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: WorkerConfig = config.worker
        self._engine: AiEngine = engine
        # Work for the time when no frame waits (for example, writing the description of a frame).
        # It returns True if it did something. An error inside it must not stop the worker.
        self._idle_task: Callable[[], bool] | None = idle_task
        self._memory: MemoryService = MemoryService(config.memory, engine)
        self._analysed: int = (
            0  # Images that reached the detector. Used for the generic pass
        )
        self._stop: threading.Event = threading.Event()
        self._wake: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- Control of the thread ----

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background thread."""
        if self.is_running:
            raise RuntimeError("The worker is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="saarthi-worker", daemon=True
        )
        self._thread.start()
        logger.info("Worker started")

    def stop(self, timeout: float | None = None) -> None:
        """Ask the thread to stop and wait for it.

        The worker finishes the image that it holds. This can take as long as
        one image (about 10 seconds or more on a slow device).
        """
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("The worker did not stop within %s seconds", timeout)
                return
        logger.info("Worker stopped")

    def wake(self) -> None:
        """Tell the worker that a new image waits. It is safe to call from any thread."""
        self._wake.set()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    worked: bool = self.process_next()
                except Exception:  # The loop must never die. Pause, then go on.
                    logger.exception("Unexpected error in the worker loop")
                    self._stop.wait(self._config.error_pause_seconds)
                    continue
                if not worked and self._run_idle_task():
                    continue  # Look for a new frame before the next idle task
                if not worked:
                    self._wake.wait(self._config.idle_wait_seconds)
                    self._wake.clear()
        finally:
            db.close_connection()

    def _run_idle_task(self) -> bool:
        """Run the idle task. Return True if it did something."""
        if self._idle_task is None or self._stop.is_set():
            return False
        try:
            return self._idle_task()
        except Exception:  # The loop must never die because of an extra task.
            logger.exception("The idle task of the worker failed")
            return False

    # ---- One image ----

    def process_next(self) -> bool:
        """Process one reference photo, or else one frame.

        A reference photo comes first: a carer waits for it, and it is only
        the work of a few seconds. Claim the oldest `pending` image if no
        photo waits.

        Return True if something was processed (also when it was rejected or
        failed), and False if nothing waited. This function does not raise
        for errors in one image.
        """
        if self._process_next_photo():
            return True
        image: Image | None = db.claim_next_pending()
        if image is None:
            return False
        try:
            self._process(image)
        except (AiError, storage.StorageError, db.DbError) as error:
            logger.error("Image %d failed: %s", image.id, type(error).__name__)
            self._fail(image.id, type(error).__name__)
        except (
            Exception
        ) as error:  # Any other error is also a failure of this image only.
            logger.exception("Image %d failed with an unexpected error", image.id)
            self._fail(image.id, type(error).__name__)
        return True

    def _process_next_photo(self) -> bool:
        """Claim the oldest `pending` reference photo and process it (stages E2 and E3)."""
        photo: ReferencePhoto | None = db.gallery.claim_next_reference_photo()
        if photo is None:
            return False
        try:
            self._locate_and_embed(photo)
        except (AiError, storage.StorageError, db.DbError) as error:
            logger.error(
                "Reference photo %d failed: %s", photo.id, type(error).__name__
            )
            self._fail_photo(photo.id, type(error).__name__)
        except (
            Exception
        ) as error:  # Any other error is also a failure of this photo only.
            logger.exception(
                "Reference photo %d failed with an unexpected error", photo.id
            )
            self._fail_photo(photo.id, type(error).__name__)
        return True

    def _locate_and_embed(self, photo: ReferencePhoto) -> None:
        """Find the object in the photo, then make its views. Both steps or none."""
        enrolled: EnrolledObject | None = db.gallery.get_object(
            photo.enrolled_object_id
        )
        if enrolled is None:
            raise db.InvalidStateError("The object of the photo does not exist")
        picture: PilImage.Image = self._engine.open_image(
            storage.read_image(photo.file_path)
        )
        located = self._engine.locate_object(
            picture, enrolled.category_phrase, list(enrolled.synonyms)
        )
        if located is None:
            db.gallery.mark_photo_failed(photo.id, REASON_OBJECT_NOT_FOUND)
            return
        views = self._engine.embed_views(picture, located.box)
        box = located.box
        db.gallery.store_photo_views(
            photo.id,
            (box.x0, box.y0, box.x1, box.y1),
            located.det_score,
            views,
            self._engine.embedding_model,
        )

    def _fail_photo(self, photo_id: int, reason: str) -> None:
        """Mark a photo as failed. The reason is a class name, never a message with data."""
        try:
            db.gallery.mark_photo_failed(photo_id, reason)
        except db.DbError:
            logger.error("Cannot mark reference photo %d as failed", photo_id)

    def _fail(self, image_id: int, reason: str) -> None:
        """Mark an image as failed. The reason is a class name, never a message with data."""
        try:
            db.mark_image_failed(image_id, reason)
        except db.DbError:
            logger.error("Cannot mark image %d as failed", image_id)

    def _process(self, image: Image) -> None:
        data: bytes = storage.read_image(image.file_path)
        picture: PilImage.Image = self._engine.open_image(data)

        report = self._engine.assess_quality(picture)
        db.set_image_analysis(image.id, report.phash, report.blur_score)
        if self._engine.is_blurry(report):
            db.mark_image_rejected(image.id, REASON_BLURRY)
            return
        since = utc_now() - timedelta(minutes=self._config.duplicate_window_minutes)
        recent: list[str] = db.list_recent_phashes(
            since, self._config.duplicate_max_compared
        )
        if self._engine.is_near_duplicate(report.phash, recent):
            db.mark_image_rejected(image.id, REASON_DUPLICATE)
            return

        self._analysed += 1
        every: int = self._config.generic_pass_every
        # The detector looks for the objects of the owner. Without a ready
        # object, it uses the placeholder list of `ai/config.toml`.
        candidates: list[Candidate] = self._engine.detect_candidates(
            picture,
            phrases=db.gallery.gallery_phrases() or None,
            include_generic=every > 0 and self._analysed % every == 0,
        )
        # The model work comes first. An error here leaves no regions behind.
        vectors: np.ndarray = self._engine.embed_regions(
            picture, [c.box for c in candidates]
        )
        regions = db.add_regions(image.id, [_to_new_region(c) for c in candidates])
        if regions:
            db.store_region_vectors(
                [r.id for r in regions], vectors, self._engine.embedding_model
            )
            # Stage I4 and I6. An error here marks the image `failed`. Its regions and vectors stay.
            self._memory.remember(image, candidates, regions, vectors, picture)
        open_boxes: int = self._store_open_boxes(image, picture, candidates, regions)
        db.mark_image_done(image.id)
        logger.info(
            "Image %d is done with %d region(s) and %d open box(es)",
            image.id,
            len(regions),
            open_boxes,
        )

    def _store_open_boxes(
        self,
        image: Image,
        picture: PilImage.Image,
        candidates: list[Candidate],
        regions: list,
    ) -> int:
        """Find the objects that nobody registered, and store their boxes with a SigLIP vector each.

        Return the number of boxes stored. The step needs the broad detector and the scene model (SigLIP).
        An error is logged, and the image goes on: the open search is an extra.
        """
        if not (self._engine.open_detection_enabled and self._engine.scene_enabled):
            return 0
        try:
            found = self._engine.detect_open(picture)
            if not found:
                return 0
            vectors: np.ndarray = self._engine.embed_open_crops(
                picture, [d.box for d in found]
            )
            items: list[db.open_regions.NewOpenRegion] = [
                db.open_regions.NewOpenRegion(
                    d.box.x0,
                    d.box.y0,
                    d.box.x1,
                    d.box.y1,
                    d.label,
                    d.confidence,
                    self._linked_region(d.box, candidates, regions),
                )
                for d in found
            ]
            db.open_regions.add_open_regions(
                image.id, items, vectors, self._engine.open_vector_model
            )
        except (AiError, db.DbError) as error:
            logger.warning(
                "The broad detection of image %d failed: %s",
                image.id,
                type(error).__name__,
            )
            return 0
        return len(items)

    def _linked_region(
        self, box: Box, candidates: list[Candidate], regions: list
    ) -> int | None:
        """Return the id of the registered-object region that this box overlaps most, if the overlap is enough."""
        best_id: int | None = None
        best: float = self._config.open_link_iou
        for candidate, region in zip(candidates, regions, strict=True):
            overlap: float = _iou(box, candidate.box)
            if overlap >= best:
                best, best_id = overlap, region.id
        return best_id
