"""The worker: takes `pending` images and runs the ai stages on them.

One image at a time. For each image the worker does these steps:

1. Read the stored file and decode it.
2. Quality check. Save the blur score and the hash.
3. Reject a blurry image and a near duplicate of a recent image.
4. Detect candidate boxes.
5. Make one vector for each box. This comes before any region row is written,
   so a model error leaves no half-written regions.
6. Add the regions, store the vectors, and mark the image `done`.

Any error marks the image `failed`. The image is not tried again. The worker
goes on with the next image.
"""

import logging
import threading
from datetime import timedelta

import numpy as np
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import AiError
from ai.types import Candidate
from backend.settings import BackendConfig, WorkerConfig
from db.schema import Image, RegionSource, utc_now

logger: logging.Logger = logging.getLogger(__name__)

REASON_BLURRY: str = "too blurry"
REASON_DUPLICATE: str = "near duplicate of a recent image"


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

    def __init__(self, config: BackendConfig, engine: AiEngine) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: WorkerConfig = config.worker
        self._engine: AiEngine = engine
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
                if not worked:
                    self._wake.wait(self._config.idle_wait_seconds)
                    self._wake.clear()
        finally:
            db.close_connection()

    # ---- One image ----

    def process_next(self) -> bool:
        """Claim the oldest `pending` image and process it.

        Return True if an image was processed (also when it was rejected or
        failed), and False if no image waited. This function does not raise
        for errors in one image.
        """
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
        candidates: list[Candidate] = self._engine.detect_candidates(
            picture, include_generic=every > 0 and self._analysed % every == 0
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
        db.mark_image_done(image.id)
        logger.info("Image %d is done with %d region(s)", image.id, len(regions))
