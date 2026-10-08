"""State of the memory page in one browser tab.

The session keeps the filter, the page of frames with their small pictures, and the frame that is open.
It has no NiceGUI code. The view reads the state and listens for changes.
"""

import asyncio
import logging
from collections.abc import Callable

from ui.ports import BrowseError, BrowseService, FrameDetail, FramesPage

logger: logging.Logger = logging.getLogger(__name__)

PAGE_SIZE: int = 24
THUMBNAIL_SIDE: int = 320
DETAIL_SIDE: int = 1280
MESSAGE_FAILED: str = "The memory could not be read. Please try again."


class BrowseSession:
    """The filter, the page, the pictures and the open frame of one tab.

    `on_changed` runs after the state changed. `on_notice(text, is_error)` runs when the user must read a short text.
    A listener error is logged and does not stop the session.
    """

    def __init__(
        self,
        service: BrowseService,
        on_changed: Callable[[], None] | None = None,
        on_notice: Callable[[str, bool], None] | None = None,
    ) -> None:
        assert service is not None, "service is required"
        self._service: BrowseService = service
        self._on_changed = on_changed
        self._on_notice = on_notice
        self._status: str | None = None
        self._offset: int = 0
        self._page: FramesPage | None = None
        self._thumbnails: dict[int, bytes] = {}
        self._busy: bool = False
        self._detail: FrameDetail | None = None
        self._detail_picture: bytes | None = None

    # ---- State ----

    @property
    def status(self) -> str | None:
        """The filter: a status, or None for all frames."""
        return self._status

    @property
    def page(self) -> FramesPage | None:
        return self._page

    @property
    def thumbnails(self) -> dict[int, bytes]:
        """The small pictures of the frames of the page, by frame id."""
        return dict(self._thumbnails)

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def detail(self) -> FrameDetail | None:
        """The open frame, or None."""
        return self._detail

    @property
    def detail_picture(self) -> bytes | None:
        return self._detail_picture

    @property
    def has_previous(self) -> bool:
        return self._offset > 0

    @property
    def has_next(self) -> bool:
        return self._page is not None and self._offset + PAGE_SIZE < self._page.total

    # ---- Actions ----

    async def load(self) -> None:
        """Load the page of frames and the small pictures. The old page stays if the load fails."""
        if self._busy:
            return
        self._busy = True
        self._notify_changed()
        try:
            page: FramesPage = await self._service.list_frames(
                self._status, self._offset, PAGE_SIZE
            )
            pictures: list[bytes | None] = await asyncio.gather(
                *(self._thumbnail(f.id) for f in page.frames)
            )
            self._page = page
            self._thumbnails = {
                f.id: p
                for f, p in zip(page.frames, pictures, strict=True)
                if p is not None
            }
        except BrowseError as error:
            self._notice(str(error), True)
        except (
            Exception
        ):  # The service is outside this module. The page must stay usable.
            logger.exception("The memory page could not be loaded")
            self._notice(MESSAGE_FAILED, True)
        finally:
            self._busy = False
            self._notify_changed()

    async def set_status(self, status: str | None) -> None:
        """Show the frames with this status (None: all). Start at the first page."""
        assert status is None or isinstance(status, str), "status must be text or None"
        if status == self._status and self._page is not None:
            return
        self._status = status
        self._offset = 0
        await self.load()

    async def next_page(self) -> None:
        if self.has_next:
            self._offset += PAGE_SIZE
            await self.load()

    async def previous_page(self) -> None:
        if self.has_previous:
            self._offset = max(0, self._offset - PAGE_SIZE)
            await self.load()

    async def open(self, image_id: int) -> None:
        """Open a frame: its details and a large picture."""
        assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
        try:
            detail: FrameDetail = await self._service.get_frame(image_id)
            picture: bytes | None = await self._picture(image_id, DETAIL_SIDE)
        except BrowseError as error:
            self._notice(str(error), True)
            return
        except (
            Exception
        ):  # The service is outside this module. The page must stay usable.
            logger.exception("The frame could not be opened")
            self._notice(MESSAGE_FAILED, True)
            return
        self._detail = detail
        self._detail_picture = picture
        self._notify_changed()

    def close(self) -> None:
        """Close the open frame."""
        if self._detail is not None:
            self._detail = None
            self._detail_picture = None
            self._notify_changed()

    # ---- Helpers ----

    async def _picture(self, image_id: int, side: int) -> bytes | None:
        """Return a picture, or None if the file is gone. The details still show."""
        try:
            return await self._service.get_picture(image_id, side)
        except BrowseError:
            return None

    async def _thumbnail(self, image_id: int) -> bytes | None:
        return await self._picture(image_id, THUMBNAIL_SIDE)

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
