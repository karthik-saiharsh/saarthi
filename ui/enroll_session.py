"""State of the enrollment page in one browser tab.

The session keeps the list of objects, the object that is open and the crops
of its photos. It calls the service and tells the view when something changes.
It has no NiceGUI code.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from ui.ports import (
    EnrollmentError,
    EnrollmentService,
    ObjectDetail,
    ObjectInfo,
    PhotoState,
)
from ui.settings import ACCEPTED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from ui.upload_session import MESSAGE_EMPTY, MESSAGE_TOO_LARGE, MESSAGE_UNSUPPORTED_TYPE

logger: logging.Logger = logging.getLogger(__name__)

# The longest side of a crop in the browser. The server cuts and shrinks the crop.
CROP_MAX_SIDE: int = 320
# Photos in these states wait for the server. The page asks again until they are done.
WAITING_STATES: frozenset[PhotoState] = frozenset(
    {PhotoState.PENDING, PhotoState.PROCESSING}
)
# Photos in these states have a crop that the carer can look at.
CROP_STATES: frozenset[PhotoState] = frozenset(
    {PhotoState.LOCATED, PhotoState.CONFIRMED}
)

MESSAGE_NAME_REQUIRED: str = "Enter a name for the object."
MESSAGE_CATEGORY_REQUIRED: str = "Enter the kind of object, for example: a bottle."
MESSAGE_LIST_FAILED: str = "The list of objects could not be loaded."
MESSAGE_OBJECT_CREATED: str = "Object added. Now add photos of it."


def parse_synonyms(text: str) -> tuple[str, ...]:
    """Split `text` at commas. Drop empty parts. The server checks each phrase."""
    assert isinstance(text, str), "text must be a string"
    return tuple(part.strip() for part in text.split(",") if part.strip())


class EnrollSession:
    """The objects and photos that one tab shows.

    `on_changed` runs after the state changed. `on_notice(text, is_error)` runs
    when the user must read a short text. A listener error is logged and does
    not stop the session.
    """

    def __init__(
        self,
        service: EnrollmentService,
        on_changed: Callable[[], None] | None = None,
        on_notice: Callable[[str, bool], None] | None = None,
    ) -> None:
        assert service is not None, "service is required"
        self._service: EnrollmentService = service
        self._on_changed = on_changed
        self._on_notice = on_notice
        # One call at a time, so a slow reload cannot overwrite a newer state.
        self._lock: asyncio.Lock = asyncio.Lock()
        self._objects: list[ObjectInfo] = []
        self._selected_id: int | None = None
        self._detail: ObjectDetail | None = None
        self._crops: dict[int, bytes] = {}
        # What the view shows now. `_reload` compares with it.
        self._shown: object = None

    @property
    def objects(self) -> list[ObjectInfo]:
        """Return a copy of the object list."""
        return list(self._objects)

    @property
    def selected_id(self) -> int | None:
        return self._selected_id

    @property
    def detail(self) -> ObjectDetail | None:
        """The open object with its photos, or None."""
        return self._detail

    @property
    def needs_polling(self) -> bool:
        """True while the server works on a photo of the open object."""
        if self._detail is None:
            return False
        return any(photo.state in WAITING_STATES for photo in self._detail.photos)

    def crop(self, photo_id: int) -> bytes | None:
        """Return the JPEG crop of a photo, or None if there is none."""
        assert isinstance(photo_id, int), "photo_id must be an int"
        return self._crops.get(photo_id)

    # ---- Actions of the user ----

    async def refresh(self) -> None:
        """Load the objects and the open object again."""
        async with self._lock:
            await self._reload()

    async def poll(self) -> None:
        """Called by a timer. It loads again only while the server works on a photo."""
        if self.needs_polling and not self._lock.locked():
            await self.refresh()

    async def select(self, object_id: int) -> None:
        """Open an object."""
        assert isinstance(object_id, int), "object_id must be an int"
        async with self._lock:
            self._selected_id = object_id
            self._detail = None
            self._crops = {}
            self._shown = None
            self._notify_changed()
            await self._reload()

    def close(self) -> None:
        """Close the open object."""
        self._selected_id = None
        self._detail = None
        self._crops = {}
        self._notify_changed()

    async def create_object(
        self, name: str, category_phrase: str, synonyms_text: str, tracked: bool
    ) -> bool:
        """Add an object and open it. Return True if it was added."""
        assert isinstance(name, str), "name must be a string"
        assert isinstance(category_phrase, str), "category_phrase must be a string"
        assert isinstance(synonyms_text, str), "synonyms_text must be a string"
        assert isinstance(tracked, bool), "tracked must be a bool"
        if not name.strip():
            self._notice(MESSAGE_NAME_REQUIRED, True)
            return False
        if not category_phrase.strip():
            self._notice(MESSAGE_CATEGORY_REQUIRED, True)
            return False
        created: list[ObjectInfo] = []

        async def action() -> None:
            created.append(
                await self._service.create_object(
                    name, category_phrase, parse_synonyms(synonyms_text), tracked
                )
            )

        if not await self._run(action):
            return False
        self._notice(MESSAGE_OBJECT_CREATED, False)
        await self.select(created[0].id)
        return True

    async def delete_selected(self) -> None:
        """Delete the open object with its photos."""
        object_id: int | None = self._selected_id
        if object_id is None:
            return

        async def action() -> None:
            await self._service.delete_object(object_id)
            self._selected_id = None
            self._detail = None
            self._crops = {}

        await self._run(action)

    async def add_photo(self, name: str, content_type: str, data: bytes) -> None:
        """Send a photo to the open object. The browser checks type and size first."""
        assert isinstance(name, str) and name, "name must be a non-empty string"
        assert isinstance(content_type, str), "content_type must be a string"
        assert isinstance(data, bytes), "data must be bytes"
        object_id: int | None = self._selected_id
        if object_id is None:
            logger.warning("A photo arrived while no object is open")
            return
        problem: str | None = self._check_limits(content_type, len(data))
        if problem is not None:
            self._notice(f"{name}: {problem}", True)
            return

        async def action() -> None:
            await self._service.add_photo(object_id, data)

        await self._run(action, error_prefix=f"{name}: ")

    async def confirm(self, photo_id: int) -> None:
        """The carer accepts the crop of a photo."""
        await self._photo_action(photo_id, self._service.confirm_photo)

    async def remove(self, photo_id: int) -> None:
        """The carer removes a photo."""
        await self._photo_action(photo_id, self._service.remove_photo)

    async def retry(self, photo_id: int) -> None:
        """The server tries a failed photo again."""
        await self._photo_action(photo_id, self._service.retry_photo)

    # ---- Helpers ----

    async def _photo_action(
        self, photo_id: int, call: Callable[[int, int], Awaitable[None]]
    ) -> None:
        assert isinstance(photo_id, int), "photo_id must be an int"
        object_id: int | None = self._selected_id
        if object_id is None:
            logger.warning("A photo action arrived while no object is open")
            return

        async def action() -> None:
            await call(object_id, photo_id)

        await self._run(action)

    async def _run(
        self, action: Callable[[], Awaitable[None]], error_prefix: str = ""
    ) -> bool:
        """Run a service call, then load the state again. Return True if it worked."""
        async with self._lock:
            worked: bool = True
            try:
                await action()
            except EnrollmentError as error:
                logger.info("Enrollment action refused: %s", error)
                self._notice(f"{error_prefix}{error}", True)
                worked = False
            # Load again also after an error: the state may have changed.
            await self._reload()
        return worked

    async def _reload(self) -> None:
        """Load the objects, the open object and the missing crops. Call it with the lock."""
        try:
            self._objects = await self._service.list_objects()
        except EnrollmentError:
            logger.exception("Cannot load the list of objects")
            self._notice(MESSAGE_LIST_FAILED, True)
            return
        if self._selected_id is not None:
            try:
                self._detail = await self._service.get_object(self._selected_id)
            except EnrollmentError:
                logger.info("The open object %d is gone", self._selected_id)
                self._selected_id = None
                self._detail = None
        await self._load_crops()
        # The view draws again only when something is new, so a poll does not flicker.
        state = (self._objects, self._detail, frozenset(self._crops))
        if state != self._shown:
            self._shown = state
            self._notify_changed()

    async def _load_crops(self) -> None:
        """Get the crop of each located photo once. Drop the crops of removed photos."""
        if self._detail is None or self._selected_id is None:
            self._crops = {}
            return
        photo_ids: set[int] = {
            photo.id for photo in self._detail.photos if photo.state in CROP_STATES
        }
        self._crops = {i: data for i, data in self._crops.items() if i in photo_ids}
        for photo_id in sorted(photo_ids - self._crops.keys()):
            try:
                self._crops[photo_id] = await self._service.get_crop(
                    self._selected_id, photo_id, CROP_MAX_SIDE
                )
            except EnrollmentError:
                # The card shows a text instead of the crop.
                logger.warning("No crop for photo %d", photo_id)

    @staticmethod
    def _check_limits(content_type: str, size: int) -> str | None:
        if size == 0:
            return MESSAGE_EMPTY
        if content_type not in ACCEPTED_CONTENT_TYPES:
            return MESSAGE_UNSUPPORTED_TYPE
        if size > MAX_FILE_SIZE_BYTES:
            return MESSAGE_TOO_LARGE
        return None

    def _notice(self, text: str, is_error: bool) -> None:
        if self._on_notice is None:
            return
        try:
            self._on_notice(text, is_error)
        except Exception:  # A view error must not stop the session
            logger.exception("Listener error for a notice")

    def _notify_changed(self) -> None:
        if self._on_changed is None:
            return
        try:
            self._on_changed()
        except Exception:  # A view error must not stop the session
            logger.exception("Listener error for a change")
