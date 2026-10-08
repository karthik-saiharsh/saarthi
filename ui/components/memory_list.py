"""The list of frames: the filter, the cards and the page buttons."""

import base64
from collections.abc import Awaitable, Callable

from nicegui import ui

from ui.components.find_result import age_text
from ui.components.status_badge import badge
from ui.ports import FrameInfo, FramesPage

# (status, label) in display order. None is all frames.
FILTERS: tuple[tuple[str | None, str], ...] = (
    (None, "All"),
    ("done", "Processed"),
    ("rejected", "Rejected"),
    ("failed", "Failed"),
    ("pending", "Waiting"),
    ("processing", "Working"),
)
STATUS_LABELS: dict[str, str] = {
    "done": "Processed",
    "rejected": "Rejected",
    "failed": "Failed",
    "pending": "Waiting",
    "processing": "Working",
}
EMPTY_TEXT: str = "No frame with this filter."
NO_PICTURE_TEXT: str = "No picture"
LOADING_TEXT: str = "Loading..."


def picture_source(jpeg: bytes) -> str:
    """Return a data URI for a JPEG. The frame stays on this page and is not cached."""
    assert jpeg, "jpeg must not be empty"
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def frame_summary(frame: FrameInfo) -> str:
    """Return a short line about what the pipeline stored: `1 match, 2 regions, 7 other boxes`."""

    def plural(count: int, word: str) -> str:
        return f"{count} {word}{'' if count == 1 else 's'}"

    return ", ".join(
        (
            plural(frame.sightings, "match").replace("matchs", "matches"),
            plural(frame.regions, "region"),
            plural(frame.open_boxes, "other box").replace("boxs", "boxes"),
        )
    )


def memory_filters(
    counts: dict[str, int],
    current: str | None,
    on_pick: Callable[[str | None], Awaitable[None]],
) -> None:
    """Build the filter buttons with the number of frames for each status."""
    assert callable(on_pick), "on_pick must be callable"
    total: int = sum(counts.values())
    with (
        ui.element("div")
        .classes("up-filters")
        .props('role="group" aria-label="Filter"')
    ):
        for status, label in FILTERS:
            number: int = total if status is None else counts.get(status, 0)
            button = ui.button(
                f"{label} {number}",
                color=None,
                on_click=lambda s=status: on_pick(s),
            ).props("unelevated no-caps")
            variant: str = "primary" if status == current else "secondary"
            button.classes(f"up-btn up-btn--{variant}")
            if status == current:
                button.props("aria-pressed=true")


def memory_grid(
    page: FramesPage,
    thumbnails: dict[int, bytes],
    on_open: Callable[[int], Awaitable[None]],
) -> None:
    """Build one card for each frame. A tap opens the details."""
    assert callable(on_open), "on_open must be callable"
    if not page.frames:
        ui.label(EMPTY_TEXT).classes("up-empty")
        return
    with ui.element("ul").classes("up-frames"):
        for frame in page.frames:
            with (
                ui.element("li"),
                ui.element("button")
                .classes("up-frame")
                .props(f'type=button aria-label="Open frame {frame.id}"')
                .on("click", lambda f=frame: on_open(f.id)),
            ):
                picture: bytes | None = thumbnails.get(frame.id)
                if picture is not None:
                    ui.element("img").props(
                        f'src="{picture_source(picture)}" alt=""'
                    ).classes("up-frame__picture")
                else:
                    ui.label(NO_PICTURE_TEXT).classes(
                        "up-frame__picture up-frame__none"
                    )
                with ui.element("div").classes("up-frame__text"):
                    badge(
                        STATUS_LABELS.get(frame.status, frame.status),
                        f"frame-{frame.status}",
                    )
                    ui.label(age_text(frame.age_seconds)).classes("up-frame__when")
                    if frame.status == "done":
                        ui.label(frame_summary(frame)).classes("up-frame__what")
                    elif frame.detail:
                        ui.label(frame.detail).classes("up-frame__what")


def memory_pager(
    page: FramesPage,
    has_previous: bool,
    has_next: bool,
    on_previous: Callable[[], Awaitable[None]],
    on_next: Callable[[], Awaitable[None]],
) -> None:
    """Build the page buttons and the position, for example `1 to 24 of 60`."""
    assert callable(on_previous) and callable(on_next), "handlers must be callable"
    if page.total <= page.limit:
        return
    first: int = page.offset + 1
    last: int = min(page.offset + len(page.frames), page.total)
    with ui.element("div").classes("up-pager"):
        previous = ui.button("Newer", color=None, on_click=on_previous).props(
            "unelevated no-caps"
        )
        previous.classes("up-btn up-btn--secondary")
        if not has_previous:
            previous.props("disable")
        ui.label(f"{first} to {last} of {page.total}").classes("up-pager__position")
        following = ui.button("Older", color=None, on_click=on_next).props(
            "unelevated no-caps"
        )
        following.classes("up-btn up-btn--secondary")
        if not has_next:
            following.props("disable")
