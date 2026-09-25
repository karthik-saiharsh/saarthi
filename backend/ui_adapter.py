"""Connects the UI to the upload step and to enrollment.

The UI calls `submit_image` through the interface in `ui/ports.py`. This
adapter runs `IngestService.submit` and turns the result or the error into
an outcome that the UI shows. `UiEnrollmentService` does the same for the
gallery. The backend is the only place where the UI and the other
components meet, so this file is here and not in `ui/`.

The file name and the content type from the browser are not used or stored.
The storage component finds the format from the content.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TypeVar

import db.api as db
import storage.api as storage
from ai.api import AiError
from backend.enrollment import (
    EnrollmentService,
    ObjectSummary,
)
from backend.enrollment import (
    ObjectDetail as BackendObjectDetail,
)
from backend.enrollment import (
    PhotoInfo as BackendPhotoInfo,
)
from backend.errors import (
    BackendError,
    InvalidInputError,
    LimitReachedError,
    NotFoundError,
    QueueFullError,
)
from backend.ingest import IngestResult, IngestService, IngestStatus
from db.schema import ObjectStatus
from ui.ports import (
    EnrollmentError,
    ObjectDetail,
    ObjectInfo,
    OutcomeStatus,
    PhotoInfo,
    PhotoState,
    UploadOutcome,
)

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


MESSAGE_NOT_FOUND: str = "This item does not exist any more. Reload the page."
MESSAGE_NAME_EXISTS: str = "An object with this name exists."
MESSAGE_PHOTO_EXISTS: str = "This object has this photo already."
MESSAGE_WRONG_STATE: str = "This action is not possible in the present state."
MESSAGE_FILE_MISSING: str = "The file of the photo is missing."

_Result = TypeVar("_Result")


def _enrollment_error(error: Exception) -> EnrollmentError:
    """Turn an error of the service into an error with a fixed text for the UI."""
    if isinstance(error, (InvalidInputError, LimitReachedError)):
        return EnrollmentError(str(error))  # The service messages are fixed texts
    if isinstance(error, NotFoundError):
        return EnrollmentError(MESSAGE_NOT_FOUND)
    if isinstance(error, db.DuplicateObjectError):
        return EnrollmentError(MESSAGE_NAME_EXISTS)
    if isinstance(error, db.DuplicateImageError):
        return EnrollmentError(MESSAGE_PHOTO_EXISTS)
    if isinstance(error, db.InvalidStateError):
        return EnrollmentError(MESSAGE_WRONG_STATE)
    if isinstance(error, storage.ImageTooLargeError):
        return EnrollmentError(MESSAGE_TOO_LARGE)
    if isinstance(error, storage.UnsupportedImageError):
        return EnrollmentError(MESSAGE_UNSUPPORTED)
    if isinstance(error, storage.StorageFullError):
        logger.error("Enrollment failed: the disk is full")
        return EnrollmentError(MESSAGE_DISK_FULL)
    if isinstance(error, storage.StorageNotFoundError):
        return EnrollmentError(MESSAGE_FILE_MISSING)
    if isinstance(error, (storage.StorageError, db.DbError, BackendError, AiError)):
        logger.error("Enrollment failed: %s", type(error).__name__, exc_info=error)
        return EnrollmentError(MESSAGE_FAILED)
    logger.error("Enrollment failed unexpectedly", exc_info=error)
    return EnrollmentError(MESSAGE_FAILED)


def _object_info(summary: ObjectSummary) -> ObjectInfo:
    return ObjectInfo(
        id=summary.id,
        name=summary.name,
        category_phrase=summary.category_phrase,
        synonyms=tuple(summary.synonyms),
        tracked=summary.tracked,
        ready=summary.status == ObjectStatus.READY.value,
        note=summary.note,
        photo_counts={
            PhotoState(status): count for status, count in summary.photo_counts.items()
        },
    )


def _photo_info(photo: BackendPhotoInfo) -> PhotoInfo:
    return PhotoInfo(photo.id, PhotoState(photo.status), photo.detail)


class UiEnrollmentService:
    """Implements `ui.ports.EnrollmentService` with the real enrollment service.

    The service blocks, so each call runs in a thread. Each error becomes an
    `EnrollmentError` with a fixed text.
    """

    def __init__(self, service: EnrollmentService) -> None:
        assert isinstance(
            service, EnrollmentService
        ), "service must be an EnrollmentService"
        self._service: EnrollmentService = service

    async def _call(
        self, function: Callable[..., _Result], *arguments: object
    ) -> _Result:
        try:
            return await asyncio.to_thread(function, *arguments)
        except Exception as error:  # The boundary to the UI: map every error.
            raise _enrollment_error(error) from error

    async def create_object(
        self,
        name: str,
        category_phrase: str,
        synonyms: tuple[str, ...],
        tracked: bool,
    ) -> ObjectInfo:
        assert isinstance(name, str), "name must be a string"
        assert isinstance(category_phrase, str), "category_phrase must be a string"
        summary: ObjectSummary = await self._call(
            self._service.create_object, name, category_phrase, list(synonyms), tracked
        )
        return _object_info(summary)

    async def list_objects(self) -> list[ObjectInfo]:
        summaries: list[ObjectSummary] = await self._call(self._service.list_objects)
        return [_object_info(summary) for summary in summaries]

    async def get_object(self, object_id: int) -> ObjectDetail:
        assert isinstance(object_id, int), "object_id must be an int"
        detail: BackendObjectDetail = await self._call(
            self._service.get_object, object_id
        )
        return ObjectDetail(
            _object_info(detail.summary),
            tuple(_photo_info(photo) for photo in detail.photos),
        )

    async def delete_object(self, object_id: int) -> None:
        assert isinstance(object_id, int), "object_id must be an int"
        await self._call(self._service.delete_object, object_id)

    async def add_photo(self, object_id: int, data: bytes) -> PhotoInfo:
        assert isinstance(object_id, int), "object_id must be an int"
        assert isinstance(data, bytes), "data must be bytes"
        photo: BackendPhotoInfo = await self._call(
            self._service.add_photo, object_id, data
        )
        return _photo_info(photo)

    async def confirm_photo(self, object_id: int, photo_id: int) -> None:
        assert isinstance(object_id, int) and isinstance(photo_id, int), "ids are int"
        await self._call(self._service.confirm_photo, object_id, photo_id)

    async def remove_photo(self, object_id: int, photo_id: int) -> None:
        assert isinstance(object_id, int) and isinstance(photo_id, int), "ids are int"
        await self._call(self._service.remove_photo, object_id, photo_id)

    async def retry_photo(self, object_id: int, photo_id: int) -> None:
        assert isinstance(object_id, int) and isinstance(photo_id, int), "ids are int"
        await self._call(self._service.retry_photo, object_id, photo_id)

    async def get_crop(self, object_id: int, photo_id: int, max_side: int) -> bytes:
        assert isinstance(object_id, int) and isinstance(photo_id, int), "ids are int"
        return await self._call(self._service.get_crop, object_id, photo_id, max_side)
