"""State of the find page in one browser tab.

The session keeps the list of objects that a search can find, the result of the
last search, and a flag that tells the view that a search runs. It has no
NiceGUI code. The view reads the state and listens for changes.
"""

import logging
from collections.abc import Callable

from ui.ports import FindChoice, FindError, FindResult, FindService
from ui.settings import MAX_QUERY_CHARS

logger: logging.Logger = logging.getLogger(__name__)

MESSAGE_QUERY_REQUIRED: str = "Write what you are looking for, or choose an object."
MESSAGE_QUERY_TOO_LONG: str = "Please use fewer words."
MESSAGE_LIST_FAILED: str = "The list of objects could not be loaded."
MESSAGE_FAILED: str = "The search did not work. Please try again."


class FindSession:
    """The objects, the busy flag and the result of the last search of one tab.

    `on_changed` runs after the state changed. `on_notice(text, is_error)` runs
    when the user must read a short text. A listener error is logged and does
    not stop the session.
    """

    def __init__(
        self,
        service: FindService,
        on_changed: Callable[[], None] | None = None,
        on_notice: Callable[[str, bool], None] | None = None,
    ) -> None:
        assert service is not None, "service is required"
        self._service: FindService = service
        self._on_changed = on_changed
        self._on_notice = on_notice
        self._choices: list[FindChoice] = []
        self._result: FindResult | None = None
        self._busy: bool = False

    @property
    def choices(self) -> list[FindChoice]:
        """Return a copy of the list of objects that a search can find."""
        return list(self._choices)

    @property
    def result(self) -> FindResult | None:
        """The result of the last search, or None."""
        return self._result

    @property
    def busy(self) -> bool:
        """True while a search runs."""
        return self._busy

    async def load(self) -> None:
        """Load the list of objects."""
        try:
            self._choices = await self._service.list_choices()
        except FindError:
            logger.exception("Cannot load the objects")
            self._notice(MESSAGE_LIST_FAILED, True)
            return
        except Exception:  # The service is outside this module.
            logger.exception("Cannot load the objects, unexpected error")
            self._notice(MESSAGE_LIST_FAILED, True)
            return
        self._notify_changed()

    async def ask(self, text: str) -> None:
        """Search with the words of the user."""
        assert isinstance(text, str), "text must be a string"
        words: str = " ".join(text.split())
        if not words:
            self._notice(MESSAGE_QUERY_REQUIRED, True)
            return
        if len(words) > MAX_QUERY_CHARS:
            self._notice(MESSAGE_QUERY_TOO_LONG, True)
            return
        await self._search(lambda: self._service.ask(words))

    async def pick(self, object_id: int) -> None:
        """Search for an object that the user chose."""
        assert isinstance(object_id, int), "object_id must be an int"
        await self._search(lambda: self._service.find(object_id))

    def clear(self) -> None:
        """Remove the result."""
        if self._result is not None:
            self._result = None
            self._notify_changed()

    async def _search(self, call: Callable) -> None:
        """Run one search. A second request while a search runs is ignored."""
        if self._busy:
            logger.info("A search runs already. The new request is ignored")
            return
        self._busy = True
        self._result = None
        self._notify_changed()
        try:
            self._result = await call()
        except FindError as error:
            logger.info("The search was refused: %s", error)
            self._notice(str(error), True)
        except (
            Exception
        ):  # The service is outside this module. The page must stay usable.
            logger.exception("The search failed with an unexpected error")
            self._notice(MESSAGE_FAILED, True)
        finally:
            self._busy = False
            self._notify_changed()

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
