"""Connects the UI to the upload step.

The UI calls `submit_image` through the interface in `ui/ports.py`. This
adapter runs `IngestService.submit` and turns the result or the error into
an outcome that the UI shows. The backend is the only place where the UI and
the other components meet, so this file is here and not in `ui/`.

The file name and the content type from the browser are not used or stored.
The storage component finds the format from the content.
"""

import asyncio
import logging

import db.api as db
import storage.api as storage
from backend.errors import BackendError, QueueFullError
from backend.ingest import IngestResult, IngestService, IngestStatus
from ui.ports import OutcomeStatus, UploadOutcome

logger: logging.Logger = logging.getLogger(__name__)

MESSAGE_DUPLICATE: str = "This image is already stored."
MESSAGE_UNSUPPORTED: str = "This file is not a usable image."
MESSAGE_TOO_LARGE: str = "This file is too large."
MESSAGE_BUSY: str = "The server is busy. Try again later."
MESSAGE_DISK_FULL: str = "The disk of the server is full."
MESSAGE_FAILED: str = "The server could not store the image."


class UiUploadService:
    """Implements `ui.ports.UploadService` with the real upload step."""

    def __init__(self, service: IngestService) -> None:
        assert isinstance(service, IngestService), "service must be an IngestService"
        self._service: IngestService = service

    async def submit_image(
        self, filename: str, content_type: str, data: bytes
    ) -> UploadOutcome:
        """Store one image. Return the outcome for the UI. Errors become outcomes."""
        assert isinstance(filename, str), "filename must be a string"
        assert isinstance(content_type, str), "content_type must be a string"
        assert isinstance(data, bytes), "data must be bytes"
        try:
            # The work is blocking (image decoding, disk, database). Keep it off the event loop.
            result: IngestResult = await asyncio.to_thread(self._service.submit, data)
        except QueueFullError:
            return UploadOutcome(OutcomeStatus.FAILED, MESSAGE_BUSY)
        except storage.ImageTooLargeError:
            return UploadOutcome(OutcomeStatus.REJECTED, MESSAGE_TOO_LARGE)
        except storage.UnsupportedImageError:
            return UploadOutcome(OutcomeStatus.REJECTED, MESSAGE_UNSUPPORTED)
        except storage.StorageFullError:
            logger.error("Upload failed: the disk is full")
            return UploadOutcome(OutcomeStatus.FAILED, MESSAGE_DISK_FULL)
        except (storage.StorageError, db.DbError, BackendError):
            logger.exception("Upload failed")
            return UploadOutcome(OutcomeStatus.FAILED, MESSAGE_FAILED)
        except (
            Exception
        ):  # The boundary to the UI: any other error becomes a failed outcome.
            logger.exception("Upload failed with an unexpected error")
            return UploadOutcome(OutcomeStatus.FAILED, MESSAGE_FAILED)

        if result.status == IngestStatus.DUPLICATE:
            return UploadOutcome(OutcomeStatus.DUPLICATE, MESSAGE_DUPLICATE)
        return UploadOutcome(OutcomeStatus.STORED, "")
