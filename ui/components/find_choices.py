"""The objects that the user can look for, as big buttons."""

from collections.abc import Awaitable, Callable

from nicegui import ui

from ui.ports import FindChoice

TITLE: str = "Choose what you look for"
EMPTY_TEXT: str = (
    "No objects are ready yet. Ask a carer to add them on the page My objects."
)


def find_choices(
    choices: list[FindChoice],
    busy: bool,
    on_pick: Callable[[int], Awaitable[None]],
) -> None:
    """Build one button for each object. `on_pick(object_id)` runs when the user taps one."""
    assert callable(on_pick), "on_pick must be callable"
    ui.label(TITLE).classes("up-section-title")
    if not choices:
        ui.label(EMPTY_TEXT).classes("up-empty")
        return
    with ui.element("ul").classes("up-find-choices"):
        for choice in choices:
            with ui.element("li"):
                button = ui.button(
                    choice.name,
                    color=None,
                    on_click=lambda c=choice: on_pick(c.object_id),
                ).props("unelevated no-caps")
                button.classes("up-btn up-btn--secondary up-find-choice")
                if busy:
                    button.props("disable")
