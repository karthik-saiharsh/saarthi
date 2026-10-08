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
from backend.browser import FrameDetailView, FrameSummary, MemoryBrowser
from backend.browser import FramePage as BrowserPage
from backend.describe import DescribeService
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
    InvalidAudioError,
    InvalidInputError,
    InvalidQueryError,
    LimitReachedError,
    NotFoundError,
    ObjectNotReadyError,
    QueueFullError,
    VoiceUnavailableError,
)
from backend.ingest import IngestResult, IngestService, IngestStatus
from backend.open_search import OpenAnswer, OpenSearchService
from backend.query import QueryCandidate, QueryResolver, QueryResult, QueryStatus
from backend.search import Answer, SearchService
from backend.voice import VoiceService
from db.schema import ObjectStatus
from ui.ports import (
    BrowseError,
    EnrollmentError,
    FindChoice,
    FindError,
    FindResult,
    FindStatus,
    FrameDetail,
    FrameInfo,
    FramesPage,
    ObjectDetail,
    ObjectInfo,
    OpenBoxInfo,
    OutcomeStatus,
    PhotoInfo,
    PhotoState,
    RegionInfo,
    Sighting,
    SightingInfo,
    UploadOutcome,
    VoiceError,
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


MESSAGE_VOICE_FAILED: str = "The computer could not listen. Please try again."
MESSAGE_FIND_FAILED: str = "The search did not work. Please try again."
MESSAGE_FIND_NOT_READY: str = "This object cannot be searched yet."
MESSAGE_FIND_NOT_FOUND: str = "This object does not exist any more."
# The longest side of the picture on the find page, in pixels.
FIND_PICTURE_SIDE: int = 640


def _find_error(error: Exception) -> FindError:
    """Turn an error of the search into an error with a fixed text for the UI."""
    if isinstance(error, InvalidQueryError):
        return FindError(str(error))  # A fixed text
    if isinstance(error, ObjectNotReadyError):
        return FindError(MESSAGE_FIND_NOT_READY)
    if isinstance(error, NotFoundError):
        return FindError(MESSAGE_FIND_NOT_FOUND)
    if isinstance(error, (storage.StorageError, db.DbError, BackendError, AiError)):
        logger.error("Search failed: %s", type(error).__name__, exc_info=error)
        return FindError(MESSAGE_FIND_FAILED)
    logger.error("Search failed unexpectedly", exc_info=error)
    return FindError(MESSAGE_FIND_FAILED)


class UiFindService:
    """Implements `ui.ports.FindService` with the real resolver and the real search.

    The work blocks, so each call runs in a thread. Each error becomes a
    `FindError` with a fixed text. The words of the user are never logged.
    """

    def __init__(
        self,
        resolver: QueryResolver,
        search: SearchService,
        enrollment: EnrollmentService,
        describer: DescribeService,
        open_search: OpenSearchService | None = None,
    ) -> None:
        assert isinstance(resolver, QueryResolver), "resolver must be a QueryResolver"
        assert isinstance(search, SearchService), "search must be a SearchService"
        assert isinstance(
            enrollment, EnrollmentService
        ), "enrollment must be an EnrollmentService"
        assert isinstance(
            describer, DescribeService
        ), "describer must be a DescribeService"
        self._resolver: QueryResolver = resolver
        self._search: SearchService = search
        self._enrollment: EnrollmentService = enrollment
        self._describer: DescribeService = describer
        self._open_search: OpenSearchService | None = open_search

    async def _call(
        self, function: Callable[..., _Result], *arguments: object
    ) -> _Result:
        try:
            return await asyncio.to_thread(function, *arguments)
        except Exception as error:  # The boundary to the UI: map every error.
            raise _find_error(error) from error

    async def list_choices(self) -> list[FindChoice]:
        summaries: list[ObjectSummary] = await self._call(self._enrollment.list_objects)
        return [
            FindChoice(s.id, s.name)
            for s in summaries
            if s.status == ObjectStatus.READY.value and s.tracked
        ]

    async def ask(self, text: str) -> FindResult:
        assert isinstance(text, str), "text must be a string"
        return await self._call(self._ask, text)

    async def find(self, object_id: int) -> FindResult:
        assert isinstance(object_id, int), "object_id must be an int"
        return await self._call(self._find, object_id, None)

    def _ask(self, text: str) -> FindResult:
        result: QueryResult = self._resolver.resolve(text)
        if result.status == QueryStatus.MATCHED:
            best: QueryCandidate = result.candidates[0]
            return self._find(best.object_id, best.name)
        if result.status == QueryStatus.AMBIGUOUS:
            return FindResult(
                FindStatus.CHOOSE,
                choices=tuple(
                    FindChoice(c.object_id, c.name) for c in result.candidates
                ),
            )
        return self._open_result(text) or FindResult(FindStatus.NO_MATCH)

    def _open_result(self, text: str) -> FindResult | None:
        """No enrolled object fits: look for what the broad detection saw. The result is always weak."""
        if self._open_search is None:
            return None
        found: OpenAnswer | None
        picture: bytes | None
        found, picture = self._open_search.find_with_picture(text, FIND_PICTURE_SIDE)
        if found is None or found.answer.age_seconds is None:
            return None
        name: str = found.label[:1].upper() + found.label[1:]
        description: str = self._describer.describe(
            found.label, found.answer, owned=False
        ).text
        return FindResult(
            FindStatus.FOUND,
            object_name=name,
            sighting=Sighting(
                age_seconds=found.answer.age_seconds,
                in_hand=False,
                strong=False,
                stale=False,
                picture=picture,
            ),
            description=description,
        )

    def _find(self, object_id: int, name: str | None) -> FindResult:
        if name is None:
            obj = db.gallery.get_object(object_id)
            if obj is None:
                raise NotFoundError("The object does not exist")
            name = obj.name
        answer: Answer
        picture: bytes | None
        answer, picture = self._search.find_with_picture(object_id, FIND_PICTURE_SIDE)
        if not answer.found or answer.age_seconds is None:
            return FindResult(
                FindStatus.NOT_SEEN, object_name=name, complete=answer.complete
            )
        description: str = self._describer.describe(name, answer).text
        return FindResult(
            FindStatus.FOUND,
            object_name=name,
            sighting=Sighting(
                age_seconds=answer.age_seconds,
                in_hand=bool(answer.in_hand),
                strong=bool(answer.strong),
                stale=answer.stale,
                picture=picture,
            ),
            complete=answer.complete,
            description=description,
        )


class UiVoiceService:
    """Implements `ui.ports.VoiceService` with the real speech recognizer.

    The work blocks, so the call runs in a thread. Each error becomes a `VoiceError` with a fixed text.
    The audio and the words are never logged.
    """

    def __init__(self, voice: VoiceService) -> None:
        assert isinstance(voice, VoiceService), "voice must be a VoiceService"
        self._voice: VoiceService = voice
        self.max_bytes: int = voice.max_bytes
        self.max_seconds: int = voice.max_seconds

    async def transcribe(self, wav: bytes) -> str:
        assert isinstance(wav, bytes), "wav must be bytes"
        try:
            return await asyncio.to_thread(self._voice.transcribe, wav)
        except (InvalidAudioError, VoiceUnavailableError) as error:
            raise VoiceError(str(error)) from error  # A fixed text
        except Exception as error:  # The boundary to the UI: map every error.
            logger.error("Listening failed: %s", type(error).__name__, exc_info=error)
            raise VoiceError(MESSAGE_VOICE_FAILED) from error


MESSAGE_BROWSE_FAILED: str = "The memory could not be read. Please try again."
MESSAGE_BROWSE_NOT_FOUND: str = "This frame does not exist any more."


def _frame_info(summary: FrameSummary) -> FrameInfo:
    return FrameInfo(
        summary.id,
        summary.status,
        summary.status_detail,
        summary.age_seconds,
        summary.width,
        summary.height,
        summary.regions,
        summary.sightings,
        summary.open_boxes,
    )


class UiBrowseService:
    """Implements `ui.ports.BrowseService` with the real memory browser.

    The work blocks, so each call runs in a thread. Each error becomes a `BrowseError` with a fixed text.
    """

    def __init__(self, browser: MemoryBrowser) -> None:
        assert isinstance(browser, MemoryBrowser), "browser must be a MemoryBrowser"
        self._browser: MemoryBrowser = browser

    async def _call(
        self, function: Callable[..., _Result], *arguments: object
    ) -> _Result:
        try:
            return await asyncio.to_thread(function, *arguments)
        except NotFoundError as error:
            raise BrowseError(MESSAGE_BROWSE_NOT_FOUND) from error
        except InvalidInputError as error:
            raise BrowseError(str(error)) from error  # A fixed text
        except Exception as error:  # The boundary to the UI: map every error.
            logger.error(
                "Memory browser failed: %s", type(error).__name__, exc_info=error
            )
            raise BrowseError(MESSAGE_BROWSE_FAILED) from error

    async def list_frames(
        self, status: str | None, offset: int, limit: int
    ) -> FramesPage:
        page: BrowserPage = await self._call(
            self._browser.list_frames, status, offset, limit
        )
        return FramesPage(
            [_frame_info(f) for f in page.frames],
            page.total,
            page.offset,
            page.limit,
            dict(page.counts),
        )

    async def get_frame(self, image_id: int) -> FrameDetail:
        assert isinstance(image_id, int), "image_id must be an int"
        view: FrameDetailView = await self._call(self._browser.get_frame, image_id)
        return FrameDetail(
            _frame_info(view.summary),
            view.summary.blur_score,
            tuple(
                RegionInfo(
                    r.id,
                    r.box,
                    tuple(r.phrases),
                    r.confidence,
                    r.source,
                    r.in_hand,
                    r.quality_ok,
                )
                for r in view.regions
            ),
            tuple(
                SightingInfo(
                    s.object_name,
                    s.region_id,
                    s.similarity,
                    s.margin,
                    s.strong,
                    s.in_hand,
                )
                for s in view.sightings
            ),
            tuple(
                OpenBoxInfo(o.id, o.box, o.label, o.confidence, o.linked_region_id)
                for o in view.open_boxes
            ),
            tuple(view.rooms),
            tuple(view.last_seen_of),
        )

    async def get_picture(self, image_id: int, max_side: int) -> bytes:
        assert isinstance(image_id, int), "image_id must be an int"
        return await self._call(self._browser.get_picture, image_id, max_side)
