"""The box where the user writes what they look for, in their own words."""

from collections.abc import Awaitable, Callable

from nicegui import ui

from ui.settings import MAX_QUERY_CHARS

TITLE: str = "Or write it in your own words"
PLACEHOLDER: str = "The blue thing I drink from"
HINT: str = "You can write it any way you like."
SUBMIT_LABEL: str = "Find it"


def find_ask(on_ask: Callable[[str], Awaitable[None]], busy: bool) -> None:
    """Build the box. `on_ask(text)` runs when the user taps the button or presses Enter."""
    assert callable(on_ask), "on_ask must be callable"
    ui.label(TITLE).classes("up-section-title")
    with ui.element("div").classes("up-find-ask"):
        words: ui.input = (
            ui.input(placeholder=PLACEHOLDER)
            .props(f"outlined maxlength={MAX_QUERY_CHARS} hint='{HINT}'")
            .props('aria-label="What do you look for?"')
            .classes("up-find-ask__field")
        )

        async def submit() -> None:
            await on_ask(words.value or "")

        words.on("keydown.enter", submit)
        button = ui.button(SUBMIT_LABEL, color=None, on_click=submit).props(
            "unelevated no-caps"
        )
        button.classes("up-btn up-btn--primary up-find-ask__submit")
        if busy:
            button.props("disable")
