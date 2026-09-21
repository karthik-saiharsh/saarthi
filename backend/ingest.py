"""The upload step of the ingestion pipeline.

`IngestService.submit` takes the bytes of one uploaded image and gives it to
the storage and db components in the right order. It is the only place that
decides that order. The HTTP endpoint and the UI both call it.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

import db.api as db
import storage.api as storage
from backend.errors import QueueFullError
from backend.settings import BackendConfig
from db.schema import Image, ImageStatus

logger: logging.Logger = logging.getLogger(__name__)


class IngestStatus(StrEnum):
    """What happened to an upload."""

    STORED = "stored"  # A new image waits for the worker
    DUPLICATE = "duplicate"  # The same image was uploaded before


@dataclass(frozen=True)
class IngestResult:
    status: IngestStatus
    image_id: int


class IngestService:
    """Accepts one image at a time. It is safe to call from several threads."""

    def __init__(self, config: BackendConfig) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        self._config: BackendConfig = config

    def submit(self, data: bytes) -> IngestResult:
        """Check, store and register one uploaded image.

        Steps: refuse if the queue is full, strip the metadata, look for a
        duplicate, write the file, add the database row. If the row cannot be
        added, the file is deleted again.

        Raise `QueueFullError` if too many images wait. Other errors come from
        the storage component (`UnsupportedImageError`, `ImageTooLargeError`,
        `StorageFullError`, ...) and from the db component (`DbError`).
        """
        assert isinstance(data, bytes), "data must be bytes"

        waiting: int = db.count_images_by_status(ImageStatus.PENDING)
        if waiting >= self._config.max_pending:
            logger.warning("Upload refused: %d image(s) wait for the worker", waiting)
            raise QueueFullError(f"{waiting} images wait for the worker")

        prepared: storage.PreparedImage = storage.prepare_image(data)
        existing: Image | None = db.find_image_by_sha256(prepared.sha256)
        if existing is not None:
            logger.info("Duplicate upload of image %d", existing.id)
            return IngestResult(IngestStatus.DUPLICATE, existing.id)

        path: str = storage.save_image(prepared)
        try:
            image: Image = db.add_image(
                path, prepared.sha256, prepared.width, prepared.height
            )
        except db.DuplicateImageError:
            # Another upload of the same picture finished first.
            self._remove_file(path)
            winner: Image | None = db.find_image_by_sha256(prepared.sha256)
            if winner is None:
                raise
            logger.info("Duplicate upload of image %d (found at the end)", winner.id)
            return IngestResult(IngestStatus.DUPLICATE, winner.id)
        except db.DbError:
            self._remove_file(path)
            raise

        logger.info(
            "Stored image %d (%dx%d)", image.id, prepared.width, prepared.height
        )
        return IngestResult(IngestStatus.STORED, image.id)

    @staticmethod
    def _remove_file(path: str) -> None:
        """Delete a file that has no row. A failure leaves an orphan for the orphan sweep."""
        report: storage.DeleteReport = storage.delete_images([path])
        if report.failed:
            logger.warning("Cannot delete the file of a failed upload: %s", path)
