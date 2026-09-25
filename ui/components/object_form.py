"""The form that adds an object."""

from collections.abc import Awaitable, Callable

from nicegui import ui

NAME_MAX: int = 80
PHRASE_MAX: int = 40

TITLE: str = "Add an object"
SUBMIT_LABEL: str = "Add object"
TRACKED_LABEL: str = "This object only looks like one of mine. Never report it."
TRACKED_HINT: str = (
    "Use this for an object that looks like another one, for example the "
    "bottle of a family member. It helps to tell the two apart."
)

# on_submit(name, category_phrase, synonyms_text, tracked) -> True if the object was added
SubmitHandler = Callable[[str, str, str, bool], Awaitable[bool]]


def object_form(on_submit: SubmitHandler) -> None:
    """Build the form. The fields are cleared after the object was added."""
    assert callable(on_submit), "on_submit must be callable"
    ui.label(TITLE).classes("up-section-title")
    with ui.element("div").classes("up-form"):
        name: ui.input = (
            ui.input("Name", placeholder="Blue bottle")
            .props(f"outlined maxlength={NAME_MAX}")
            .classes("up-form__field")
        )
        category: ui.input = (
            ui.input("Kind of object", placeholder="a bottle")
            .props(
                f"outlined maxlength={PHRASE_MAX} hint='Plain words, for example: a bottle'"
            )
            .classes("up-form__field")
        )
        synonyms: ui.input = (
            ui.input("Other words (optional)", placeholder="a flask, a water bottle")
            .props("outlined hint='Separate the words with commas'")
            .classes("up-form__field")
        )
        tracked: ui.checkbox = ui.checkbox(TRACKED_LABEL).classes("up-form__check")
        ui.label(TRACKED_HINT).classes("up-form__hint")

        async def submit() -> None:
            added: bool = await on_submit(
                name.value or "",
                category.value or "",
                synonyms.value or "",
                not tracked.value,
            )
            if added:
                for field in (name, category, synonyms):
                    field.set_value("")
                tracked.set_value(False)

        ui.button(SUBMIT_LABEL, color=None, on_click=submit).props(
            "unelevated no-caps"
        ).classes("up-btn up-btn--primary up-form__submit")
