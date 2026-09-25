"""File picker and drop area.

The drop area is a Quasar uploader. Its header slot holds our own text and
buttons. A second uploader opens the camera on phones.
"""

import inspect
import logging
from collections.abc import Awaitable, Callable

from nicegui import ui
from nicegui.events import UploadEventArguments

from ui.settings import ACCEPT_ATTRIBUTE, MAX_FILE_SIZE_BYTES

logger: logging.Logger = logging.getLogger(__name__)

TITLE: str = "Add pictures"
DRAG_TEXT: str = "Drag images here, or choose them from this device."
LIMITS_TEXT: str = (
    f"JPEG, PNG or WebP. Up to {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB each."
)
REJECTED_TEXT: str = "Some files were skipped. " + LIMITS_TEXT

# The handler may be a plain function or a coroutine function.
FileHandler = Callable[[str, str, bytes], Awaitable[None] | None]


def _make_upload_handler(
    on_file: FileHandler,
) -> Callable[[UploadEventArguments], Awaitable[None]]:
    async def handle(event: UploadEventArguments) -> None:
        data: bytes = await event.file.read()
        logger.info("Received a file of %d bytes from the browser", len(data))
        result = on_file(event.file.name, event.file.content_type, data)
        if inspect.isawaitable(result):
            await result

    return handle


def _notify_rejected() -> None:
    ui.notify(REJECTED_TEXT, type="warning")


def _trigger_button(label: str, classes: str) -> None:
    """Build a button that opens the file chooser.

    The Quasar add trigger puts a transparent file input over its parent
    element. A custom header slot has no input without this trigger.
    """
    with ui.button(label, color=None).props("unelevated no-caps").classes(classes):
        ui.element("q-uploader-add-trigger")


def upload_picker(on_file: FileHandler, title: str = TITLE) -> None:
    """Build the picker. `on_file(name, content_type, data)` runs for each file."""
    assert callable(on_file), "on_file must be callable"
    assert title, "title is required"
    handler = _make_upload_handler(on_file)

    zone: ui.upload = (
        ui.upload(
            multiple=True,
            on_upload=handler,
            on_rejected=_notify_rejected,
            max_file_size=MAX_FILE_SIZE_BYTES,
            auto_upload=True,
        )
        .props(f'accept="{ACCEPT_ATTRIBUTE}" flat')
        .classes("up-picker__zone")
    )
    with zone.add_slot("header"), ui.element("div").classes("up-picker__content"):
        ui.label(title).classes("up-picker__title")
        ui.label(DRAG_TEXT).classes("up-picker__hint up-picker__draghint")
        ui.label(LIMITS_TEXT).classes("up-picker__hint")
        with ui.element("div").classes("up-picker__actions"):
            _trigger_button("Choose images", "up-btn up-btn--primary")
            # A second uploader, because `capture` applies to the whole input.
            camera: ui.upload = (
                ui.upload(
                    on_upload=handler,
                    on_rejected=_notify_rejected,
                    max_file_size=MAX_FILE_SIZE_BYTES,
                    auto_upload=True,
                )
                .props(f'accept="{ACCEPT_ATTRIBUTE}" capture=environment flat')
                .classes("up-picker__camera")
            )
            with camera.add_slot("header"):
                _trigger_button("Take photo", "up-btn up-btn--secondary")
