"""The open object: its header, and the cards of its photos."""

import base64
from collections.abc import Awaitable, Callable

from nicegui import ui

from ui.components.status_badge import badge, photo_badge
from ui.ports import ObjectDetail, PhotoInfo, PhotoState

PhotoHandler = Callable[[int], Awaitable[None]]

EMPTY_TEXT: str = "No photos yet. Add 5 to 10 photos of this object."
NOT_FOUND_TEXT: str = "The object was not found in this photo. Remove it or try again."
UNREADABLE_TEXT: str = "The server could not use this photo. Remove it or try again."
WAITING_TEXT: str = "The server looks for the object. This takes some seconds."
NO_CROP_TEXT: str = "The crop is not available."

# The detail text that the worker writes when the detector finds nothing.
REASON_OBJECT_NOT_FOUND: str = "object not found"


def failure_text(detail: str | None) -> str:
    """Return the text for a failed photo. Internal reasons are not shown."""
    return NOT_FOUND_TEXT if detail == REASON_OBJECT_NOT_FOUND else UNREADABLE_TEXT


def image_source(jpeg: bytes) -> str:
    """Return a data URI for a JPEG. The crop stays on this page and is not cached."""
    assert jpeg, "jpeg must not be empty"
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def object_header(
    detail: ObjectDetail,
    on_delete: Callable[[], Awaitable[None]],
    on_close: Callable[[], None],
) -> None:
    """Build the name, the words, the note and the buttons of the open object."""
    assert callable(on_delete) and callable(on_close), "handlers must be callable"
    info = detail.info
    with ui.element("div").classes("up-detail__head"):
        with ui.element("div").classes("up-detail__titles"):
            ui.label(info.name).classes("up-detail__name")
            words: str = ", ".join((info.category_phrase, *info.synonyms))
            ui.label(f"Words for the search: {words}").classes("up-detail__meta")
            if info.note:
                ui.label(info.note).classes("up-object__note")
        badge(
            "Ready" if info.ready else "Needs photos",
            "ready" if info.ready else "draft",
        )

    with ui.dialog() as dialog, ui.card().classes("up-dialog"):
        ui.label(f'Delete "{info.name}" and all its photos?').classes("up-dialog__text")
        with ui.element("div").classes("up-dialog__actions"):
            ui.button("Cancel", color=None, on_click=dialog.close).props(
                "unelevated no-caps"
            ).classes("up-btn up-btn--secondary")

            async def confirm_delete() -> None:
                dialog.close()
                await on_delete()

            ui.button("Delete", color=None, on_click=confirm_delete).props(
                "unelevated no-caps"
            ).classes("up-btn up-btn--danger")

    with ui.element("div").classes("up-detail__actions"):
        ui.button("Close", color=None, on_click=on_close).props(
            "unelevated no-caps"
        ).classes("up-btn up-btn--secondary")
        ui.button("Delete object", color=None, on_click=dialog.open).props(
            "unelevated no-caps"
        ).classes("up-btn up-btn--secondary")


def photo_grid(
    detail: ObjectDetail,
    crop_of: Callable[[int], bytes | None],
    on_confirm: PhotoHandler,
    on_remove: PhotoHandler,
    on_retry: PhotoHandler,
) -> None:
    """Build one card for each photo. `crop_of(photo_id)` gives the JPEG crop or None."""
    assert callable(crop_of), "crop_of must be callable"
    if not detail.photos:
        ui.label(EMPTY_TEXT).classes("up-empty")
        return
    with ui.element("ul").classes("up-photos"):
        for photo in detail.photos:
            photo_card(photo, crop_of(photo.id), on_confirm, on_remove, on_retry)


def photo_card(
    photo: PhotoInfo,
    crop: bytes | None,
    on_confirm: PhotoHandler,
    on_remove: PhotoHandler,
    on_retry: PhotoHandler,
) -> None:
    """Build the card of one photo: the crop, the state and the buttons that fit the state."""
    with ui.element("li").classes("up-photo"):
        with ui.element("div").classes("up-photo__frame"):
            if crop is not None:
                ui.image(image_source(crop)).props(
                    'alt="The part of the photo where the object was found"'
                ).classes("up-photo__image")
            else:
                ui.label(_placeholder_text(photo)).classes("up-photo__placeholder")
        photo_badge(photo.state)
        with ui.element("div").classes("up-photo__actions"):
            if photo.state == PhotoState.LOCATED:
                _button("Accept", lambda: on_confirm(photo.id), primary=True)
            if photo.state == PhotoState.FAILED:
                _button("Try again", lambda: on_retry(photo.id), primary=False)
            # The server holds a photo while it works on it, so it cannot be removed then.
            if photo.state not in (PhotoState.PENDING, PhotoState.PROCESSING):
                _button("Remove", lambda: on_remove(photo.id), primary=False)


def _placeholder_text(photo: PhotoInfo) -> str:
    if photo.state == PhotoState.FAILED:
        return failure_text(photo.detail)
    if photo.state in (PhotoState.PENDING, PhotoState.PROCESSING):
        return WAITING_TEXT
    return NO_CROP_TEXT


def _button(label: str, on_click: Callable[[], Awaitable[None]], primary: bool) -> None:
    kind: str = "primary" if primary else "secondary"
    ui.button(label, color=None, on_click=on_click).props("unelevated no-caps").classes(
        f"up-btn up-btn--{kind}"
    )
