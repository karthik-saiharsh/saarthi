"""Is this your object? The boxes of recent frames that may show the open object (plan part D2).

The carer answers each box. "Yes" makes the picture a photo of the object. "No" tells the computer not to choose
that kind of box for the object again.
"""

from collections.abc import Awaitable, Callable

from nicegui import ui

from ui.components.find_result import age_text
from ui.components.object_detail import image_source
from ui.ports import ReviewItem

AnswerHandler = Callable[[int, bool], Awaitable[None]]

TITLE: str = "Is this your {name}?"
HINT_TEXT: str = (
    "The computer saw these things in the last days. They may be your {name}. "
    "Answer each picture. A picture from your home helps the computer most."
)
YES_TEXT: str = "Yes, this is it"
NO_TEXT: str = "No"
PICTURE_ALT: str = "A thing that the computer saw, which may be the object"


def object_review(
    name: str,
    items: list[ReviewItem],
    picture_of: Callable[[int], bytes | None],
    on_answer: AnswerHandler,
) -> None:
    """Build one card for each box to check. Nothing shows when there is no box."""
    assert isinstance(name, str), "name must be a string"
    assert callable(picture_of) and callable(on_answer), "handlers must be callable"
    if not items:
        return
    ui.label(TITLE.format(name=name)).classes("up-section-title")
    ui.label(HINT_TEXT.format(name=name)).classes("up-form__hint")
    with ui.element("ul").classes("up-photos"):
        for item in items:
            picture: bytes | None = picture_of(item.region_id)
            if picture is None:
                continue
            with ui.element("li").classes("up-photo"):
                with ui.element("div").classes("up-photo__frame"):
                    ui.image(image_source(picture)).props(
                        f'alt="{PICTURE_ALT}"'
                    ).classes("up-photo__image")
                ui.label(f"Seen {age_text(item.age_seconds)}.").classes(
                    "up-detail__meta"
                )
                with ui.element("div").classes("up-photo__actions"):
                    _answer_button(YES_TEXT, item.region_id, True, on_answer)
                    _answer_button(NO_TEXT, item.region_id, False, on_answer)


def _answer_button(
    label: str, region_id: int, yes: bool, on_answer: AnswerHandler
) -> None:
    kind: str = "primary" if yes else "secondary"

    async def click() -> None:
        await on_answer(region_id, yes)

    ui.button(label, color=None, on_click=click).props("unelevated no-caps").classes(
        f"up-btn up-btn--{kind}"
    )
