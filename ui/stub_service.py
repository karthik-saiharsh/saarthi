"""Fake services for work on the UI without the backend.

`StubUploadService`: the file name selects the outcome.
`duplicate` -> DUPLICATE, `reject` -> REJECTED, `fail` -> FAILED.
Any other name -> STORED.

`StubEnrollmentService`: keeps the gallery in memory. A new photo is
`pending` for 1 s, `processing` for 1.5 s, and then `located`. Every fourth
photo of an object fails with "object not found".
"""

import asyncio
import io
import logging
import random
import time
from dataclasses import dataclass, field

from PIL import Image

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

_OUTCOMES_BY_KEYWORD: dict[str, UploadOutcome] = {
    "duplicate": UploadOutcome(
        OutcomeStatus.DUPLICATE, "This image is already stored."
    ),
    "reject": UploadOutcome(OutcomeStatus.REJECTED, "The image is too blurry."),
    "fail": UploadOutcome(
        OutcomeStatus.FAILED, "The server could not store the image."
    ),
}
_STORED: UploadOutcome = UploadOutcome(OutcomeStatus.STORED, "")


class StubUploadService:
    """Implements `UploadService` with a short delay and a fixed outcome."""

    async def submit_image(
        self, filename: str, content_type: str, data: bytes
    ) -> UploadOutcome:
        assert isinstance(filename, str) and filename, "filename must be a string"
        assert isinstance(content_type, str), "content_type must be a string"
        assert isinstance(data, bytes) and data, "data must be non-empty bytes"

        await asyncio.sleep(random.uniform(0.4, 1.2))
        lowered: str = filename.lower()
        for keyword, outcome in _OUTCOMES_BY_KEYWORD.items():
            if keyword in lowered:
                logger.info("Stub outcome %s for keyword %s", outcome.status, keyword)
                return outcome
        return _STORED


PENDING_SECONDS: float = 1.0
PROCESSING_SECONDS: float = 1.5
FAIL_EVERY: int = 4
STUB_MIN_PHOTOS: int = 3


@dataclass
class _StubPhoto:
    id: int
    created: float
    fails: bool
    confirmed: bool = False
    retried_at: float | None = None

    def state(self) -> PhotoState:
        if self.confirmed:
            return PhotoState.CONFIRMED
        age: float = time.monotonic() - (self.retried_at or self.created)
        if age < PENDING_SECONDS:
            return PhotoState.PENDING
        if age < PENDING_SECONDS + PROCESSING_SECONDS:
            return PhotoState.PROCESSING
        return PhotoState.FAILED if self.fails else PhotoState.LOCATED


@dataclass
class _StubObject:
    id: int
    name: str
    category_phrase: str
    synonyms: tuple[str, ...]
    tracked: bool
    photos: list[_StubPhoto] = field(default_factory=list)


class StubEnrollmentService:
    """Implements `EnrollmentService` in memory, with fixed delays."""

    def __init__(self) -> None:
        self._objects: dict[int, _StubObject] = {}
        self._next_object: int = 1
        self._next_photo: int = 1

    async def create_object(
        self,
        name: str,
        category_phrase: str,
        synonyms: tuple[str, ...],
        tracked: bool,
    ) -> ObjectInfo:
        assert name and category_phrase, "name and category_phrase are required"
        await asyncio.sleep(0.2)
        if any(o.name == name for o in self._objects.values()):
            raise EnrollmentError("An object with this name exists.")
        obj: _StubObject = _StubObject(
            self._next_object, name, category_phrase, synonyms, tracked
        )
        self._objects[obj.id] = obj
        self._next_object += 1
        return self._info(obj)

    async def list_objects(self) -> list[ObjectInfo]:
        return [self._info(obj) for obj in self._objects.values()]

    async def get_object(self, object_id: int) -> ObjectDetail:
        obj: _StubObject = self._find(object_id)
        photos: tuple[PhotoInfo, ...] = tuple(
            self._photo_info(photo) for photo in obj.photos
        )
        return ObjectDetail(self._info(obj), photos)

    async def delete_object(self, object_id: int) -> None:
        self._find(object_id)
        del self._objects[object_id]

    async def add_photo(self, object_id: int, data: bytes) -> PhotoInfo:
        assert data, "data must not be empty"
        obj: _StubObject = self._find(object_id)
        await asyncio.sleep(0.3)
        photo: _StubPhoto = _StubPhoto(
            self._next_photo, time.monotonic(), (len(obj.photos) + 1) % FAIL_EVERY == 0
        )
        self._next_photo += 1
        obj.photos.append(photo)
        return self._photo_info(photo)

    async def confirm_photo(self, object_id: int, photo_id: int) -> None:
        photo: _StubPhoto = self._find_photo(object_id, photo_id)
        if photo.state() != PhotoState.LOCATED:
            raise EnrollmentError("This action is not possible in the present state.")
        photo.confirmed = True

    async def remove_photo(self, object_id: int, photo_id: int) -> None:
        obj: _StubObject = self._find(object_id)
        photo: _StubPhoto = self._find_photo(object_id, photo_id)
        obj.photos.remove(photo)

    async def retry_photo(self, object_id: int, photo_id: int) -> None:
        photo: _StubPhoto = self._find_photo(object_id, photo_id)
        if photo.state() != PhotoState.FAILED:
            raise EnrollmentError("This action is not possible in the present state.")
        photo.fails = False
        photo.retried_at = time.monotonic()

    async def get_crop(self, object_id: int, photo_id: int, max_side: int) -> bytes:
        photo: _StubPhoto = self._find_photo(object_id, photo_id)
        if photo.state() not in (PhotoState.LOCATED, PhotoState.CONFIRMED):
            raise EnrollmentError("The object is not located in this photo yet.")
        # A plain picture: its colour depends on the photo, so the cards differ.
        hue: int = (photo.id * 47) % 200
        image: Image.Image = Image.new("RGB", (max_side, max_side * 3 // 4))
        image.paste((40 + hue, 120, 200 - hue // 2), (0, 0, *image.size))
        buffer: io.BytesIO = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()

    def _find(self, object_id: int) -> _StubObject:
        obj: _StubObject | None = self._objects.get(object_id)
        if obj is None:
            raise EnrollmentError("This item does not exist any more.")
        return obj

    def _find_photo(self, object_id: int, photo_id: int) -> _StubPhoto:
        obj: _StubObject = self._find(object_id)
        for photo in obj.photos:
            if photo.id == photo_id:
                return photo
        raise EnrollmentError("This item does not exist any more.")

    @staticmethod
    def _photo_info(photo: _StubPhoto) -> PhotoInfo:
        state: PhotoState = photo.state()
        detail: str | None = "object not found" if state == PhotoState.FAILED else None
        return PhotoInfo(photo.id, state, detail)

    @staticmethod
    def _info(obj: _StubObject) -> ObjectInfo:
        counts: dict[PhotoState, int] = {}
        for photo in obj.photos:
            state: PhotoState = photo.state()
            counts[state] = counts.get(state, 0) + 1
        confirmed: int = counts.get(PhotoState.CONFIRMED, 0)
        ready: bool = confirmed >= STUB_MIN_PHOTOS
        note: str | None = (
            None
            if ready
            else f"Needs {STUB_MIN_PHOTOS - confirmed} more confirmed photo(s)."
        )
        return ObjectInfo(
            obj.id,
            obj.name,
            obj.category_phrase,
            obj.synonyms,
            obj.tracked,
            ready,
            note,
            counts,
        )
