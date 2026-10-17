"""The result of a search: where and when the object was seen, or what to do next."""

import base64
from collections.abc import Awaitable, Callable

from nicegui import ui

from ui.ports import FindResult, FindStatus, Sighting
from ui.speech import SAY_AGAIN_TEXT

BUSY_TEXT: str = "Looking..."
CHOOSE_TEXT: str = "Which one do you mean?"
SOMETHING_ELSE_TEXT: str = "Something else"
NO_MATCH_TEXT: str = (
    "The computer does not know which object you mean. Choose an object from the list."
)
NOT_SEEN_TEXT: str = "The computer has not seen it lately."
INCOMPLETE_TEXT: str = (
    "The computer could not look at all the pictures. Try again later."
)
MAYBE_TEXT: str = "Maybe. The computer is not sure. Look at the picture."
IN_HAND_TEXT: str = "You were holding it."
STALE_TEXT: str = "This was a long time ago. The computer found nothing newer."
NO_PICTURE_TEXT: str = "There is no picture."
PICTURE_ALT: str = "The place where the object was seen"
MAYBE_PLACE_TITLE: str = "Maybe it is here now. The computer is not sure."
MAYBE_PICTURE_ALT: str = "A newer place where the object may be"

_MINUTE: int = 60
_HOUR: int = 60 * _MINUTE
_DAY: int = 24 * _HOUR


def age_text(seconds: int) -> str:
    """Return a short, plain time such as `12 minutes ago`."""
    assert isinstance(seconds, int) and seconds >= 0, "seconds must be 0 or more"
    if seconds < _MINUTE:
        return "just now"
    for unit, name in ((_DAY, "day"), (_HOUR, "hour"), (_MINUTE, "minute")):
        if seconds >= unit:
            count: int = seconds // unit
            return f"{count} {name}{'' if count == 1 else 's'} ago"
    raise AssertionError("unreachable")  # pragma: no cover


def sighting_lines(sighting: Sighting) -> list[str]:
    """Return the lines under the picture, most important first."""
    lines: list[str] = [f"Seen {age_text(sighting.age_seconds)}."]
    if sighting.in_hand:
        lines.append(IN_HAND_TEXT)
    if not sighting.strong:
        lines.append(MAYBE_TEXT)
    if sighting.stale:
        lines.append(STALE_TEXT)
    return lines


def picture_source(jpeg: bytes) -> str:
    """Return a data URI for a JPEG. The picture stays on this page and is not cached."""
    assert jpeg, "jpeg must not be empty"
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def find_result(
    result: FindResult | None,
    busy: bool,
    on_pick: Callable[[int], Awaitable[None]],
    on_speak: Callable[[str], None] | None = None,
    on_something_else: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Build the result. A screen reader reads a new result (the area is live).

    `on_something_else` adds the button "Something else" under the choices. It runs when the user taps it.
    """
    assert callable(on_pick), "on_pick must be callable"
    with (
        ui.element("div")
        .classes("up-find-result")
        .props("role=status aria-live=polite")
    ):
        if busy:
            with ui.element("div").classes("up-find-busy"):
                ui.spinner(size="lg")
                ui.label(BUSY_TEXT).classes("up-find-busy__text")
            return
        if result is None:
            return
        if result.status == FindStatus.FOUND and result.sighting is not None:
            _found(result, result.sighting, on_speak)
        elif result.status == FindStatus.NOT_SEEN:
            _not_seen(result)
        elif result.status == FindStatus.CHOOSE:
            _choose(result, on_pick, on_something_else)
        else:
            ui.label(NO_MATCH_TEXT).classes("up-find-text")


def _found(
    result: FindResult,
    sighting: Sighting,
    on_speak: Callable[[str], None] | None,
) -> None:
    ui.label(result.object_name or "").classes("up-find-name")
    # The words come before the picture. A tall picture (a bottle) can fill a whole phone screen.
    lines = sighting_lines(sighting)
    ui.label(lines[0]).classes("up-find-when")
    for line in lines[1:]:
        ui.label(line).classes("up-find-text")
    if result.description:
        # The sentence is the text that the voice will say. It names the place.
        ui.label(result.description).classes("up-find-text up-find-description")
        if on_speak is not None:
            text: str = result.description
            ui.button(
                SAY_AGAIN_TEXT, color=None, on_click=lambda: on_speak(text)
            ).props("unelevated no-caps").classes("up-btn up-btn--primary")
    if sighting.picture is not None:
        ui.image(picture_source(sighting.picture)).props(
            f'fit=contain alt="{PICTURE_ALT}"'
        ).classes("up-find-picture")
    else:
        ui.label(NO_PICTURE_TEXT).classes("up-find-text")
    if result.maybe is not None:
        _maybe_place(result.maybe)


def _maybe_place(maybe: Sighting) -> None:
    """The newer weak sighting: a second place, under the first. The text says that the computer is not sure."""
    ui.label(MAYBE_PLACE_TITLE).classes("up-find-when")
    ui.label(f"Seen {age_text(maybe.age_seconds)}.").classes("up-find-text")
    if maybe.in_hand:
        ui.label(IN_HAND_TEXT).classes("up-find-text")
    if maybe.picture is not None:
        ui.image(picture_source(maybe.picture)).props(
            f'fit=contain alt="{MAYBE_PICTURE_ALT}"'
        ).classes("up-find-picture")


def _not_seen(result: FindResult) -> None:
    ui.label(result.object_name or "").classes("up-find-name")
    ui.label(NOT_SEEN_TEXT).classes("up-find-when")
    if not result.complete:
        ui.label(INCOMPLETE_TEXT).classes("up-find-text")


def _choose(
    result: FindResult,
    on_pick: Callable[[int], Awaitable[None]],
    on_something_else: Callable[[], Awaitable[None]] | None,
) -> None:
    ui.label(CHOOSE_TEXT).classes("up-find-when")
    with ui.element("ul").classes("up-find-choices"):
        for choice in result.choices:
            with ui.element("li"):
                ui.button(
                    choice.name,
                    color=None,
                    on_click=lambda c=choice: on_pick(c.object_id),
                ).props("unelevated no-caps").classes(
                    "up-btn up-btn--primary up-find-choice"
                )
        if on_something_else is not None:
            # The object may be one that nobody registered. This button looks for it with the same words.
            with ui.element("li"):
                ui.button(
                    SOMETHING_ELSE_TEXT,
                    color=None,
                    on_click=lambda: on_something_else(),
                ).props("unelevated no-caps").classes(
                    "up-btn up-btn--secondary up-find-choice"
                )
