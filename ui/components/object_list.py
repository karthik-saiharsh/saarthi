"""The list of objects."""

from collections.abc import Awaitable, Callable, Mapping

from nicegui import ui

from ui.components.status_badge import badge
from ui.ports import ObjectInfo, PhotoState

TITLE: str = "My objects"
EMPTY_TEXT: str = "No objects yet. Add the first object above."

# Text for each photo state, in the order that the summary shows them.
_COUNT_LABELS: tuple[tuple[PhotoState, str], ...] = (
    (PhotoState.CONFIRMED, "accepted"),
    (PhotoState.LOCATED, "to check"),
    (PhotoState.FAILED, "not found"),
    (PhotoState.PENDING, "waiting"),
    (PhotoState.PROCESSING, "waiting"),
)


def photo_summary(counts: Mapping[PhotoState, int]) -> str:
    """Return a short text such as `3 accepted, 1 to check`."""
    parts: dict[str, int] = {}
    for state, label in _COUNT_LABELS:
        if counts.get(state, 0):
            parts[label] = parts.get(label, 0) + counts[state]
    if not parts:
        return "No photos"
    return ", ".join(f"{count} {label}" for label, count in parts.items())


def object_list(
    objects: list[ObjectInfo],
    selected_id: int | None,
    on_open: Callable[[int], Awaitable[None]],
) -> None:
    """Build the list. `on_open(object_id)` runs when the user opens an object."""
    assert callable(on_open), "on_open must be callable"
    ui.label(TITLE).classes("up-section-title")
    if not objects:
        ui.label(EMPTY_TEXT).classes("up-empty")
        return
    with ui.element("ul").classes("up-objects"):
        for obj in objects:
            _object_row(obj, obj.id == selected_id, on_open)


def _object_row(
    obj: ObjectInfo, selected: bool, on_open: Callable[[int], Awaitable[None]]
) -> None:
    classes: str = "up-object" + (" up-object--selected" if selected else "")
    with ui.element("li").classes(classes):
        with ui.element("div").classes("up-object__main"):
            ui.label(obj.name).classes("up-object__name")
            ui.label(
                f"{obj.category_phrase} - {photo_summary(obj.photo_counts)}"
            ).classes("up-object__meta")
            if obj.note:
                ui.label(obj.note).classes("up-object__note")
        with ui.element("div").classes("up-object__side"):
            if not obj.tracked:
                badge("Look-alike", "lookalike")
            if obj.ready:
                badge("Ready", "ready")
            else:
                badge("Needs photos", "draft")
            button = ui.button(
                "Open", color=None, on_click=lambda: on_open(obj.id)
            ).props("unelevated no-caps")
            button.classes("up-btn up-btn--secondary")
            if selected:
                button.props("disable")
