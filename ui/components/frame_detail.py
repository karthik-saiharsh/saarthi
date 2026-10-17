"""The details of one frame: the picture with the boxes, and what the pipeline stored for it."""

from collections.abc import Callable

from nicegui import ui

from ui.components.find_result import age_text
from ui.components.memory_list import STATUS_LABELS, picture_source
from ui.components.status_badge import badge
from ui.ports import FrameDetail, OpenBoxInfo, RegionInfo

NO_PICTURE_TEXT: str = "The picture of this frame is not available."
NO_MATCH_TEXT: str = "No registered object was matched in this frame."
NO_REGIONS_TEXT: str = "The detector found no box for a registered object."
NO_OPEN_TEXT: str = "The broad detector found no other object."
NO_ROOM_TEXT: str = (
    "No room score yet. The score is made when an answer that uses this frame is described."
)
OPEN_HINT: str = (
    "The names come from a detector that has no list of classes. A name can be wrong."
)


def box_style(box: tuple[int, int, int, int], width: int, height: int) -> str:
    """Return the CSS that places a box over the picture, in percent of the frame."""
    x0, y0, x1, y1 = box
    assert width > 0 and height > 0, "the frame size must be positive"
    assert (
        0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
    ), "the box must be in the frame"
    return (
        f"left:{x0 / width * 100:.2f}%;top:{y0 / height * 100:.2f}%;"
        f"width:{(x1 - x0) / width * 100:.2f}%;height:{(y1 - y0) / height * 100:.2f}%"
    )


def region_text(region: RegionInfo) -> str:
    """One line about a registered-object box."""
    words: str = ", ".join(region.phrases) or "no words"
    parts: list[str] = [
        f"Box {region.id}: {words}",
        f"detector {region.confidence:.2f}",
        region.source,
    ]
    if region.in_hand:
        parts.append("in a hand")
    if not region.quality_ok:
        parts.append("the crop is not used (small, blurred or cut by the edge)")
    return " - ".join(parts)


def open_text(box: OpenBoxInfo) -> str:
    """One line about a box of the broad detector."""
    text: str = f"{box.label} - {box.confidence:.2f}"
    if box.linked_region_id is not None:
        text += f" - the same place as box {box.linked_region_id}"
    return text


def _overlay(detail: FrameDetail, picture: bytes, layers: dict[str, bool]) -> None:
    frame = detail.frame
    matched: dict[int, str] = {s.region_id: s.object_name for s in detail.sightings}
    with ui.element("div").classes("up-shot"):
        # A plain image, so the boxes (in percent) fit the picture exactly.
        ui.element("img").props(
            f'src="{picture_source(picture)}" alt="Frame {frame.id}"'
        ).classes("up-shot__picture")
        if layers.get("regions", True):
            for region in detail.regions:
                name: str | None = matched.get(region.id)
                kind: str = "match" if name is not None else "region"
                caption: str = name or (region.phrases[0] if region.phrases else "")
                _box(region.box, frame.width, frame.height, kind, caption)
        if layers.get("open", True):
            for box in detail.open_boxes:
                # A box of the broad detector can also hold a sighting (the matching mode keys).
                found: str | None = matched.get(box.id)
                kind = "match" if found is not None else "open"
                _box(box.box, frame.width, frame.height, kind, found or box.label)


def _box(
    box: tuple[int, int, int, int], width: int, height: int, kind: str, caption: str
) -> None:
    with (
        ui.element("div")
        .classes(f"up-box up-box--{kind}")
        .props(f'title="{caption}"')
        .style(box_style(box, width, height))
    ):
        if caption:
            ui.label(caption).classes("up-box__label")


def frame_detail(
    detail: FrameDetail,
    picture: bytes | None,
    layers: dict[str, bool],
    on_layer: Callable[[], None],
    on_close: Callable[[], None],
) -> None:
    """Build the details. `layers` says which boxes show. `on_layer` runs after a change of `layers`."""
    assert callable(on_layer) and callable(on_close), "handlers must be callable"
    frame = detail.frame
    with ui.element("div").classes("up-detail__head"):
        with ui.element("div").classes("up-detail__titles"):
            ui.label(f"Frame {frame.id}").classes("up-detail__name")
            ui.label(f"Received {age_text(frame.age_seconds)}").classes(
                "up-detail__meta"
            )
        badge(STATUS_LABELS.get(frame.status, frame.status), f"frame-{frame.status}")
        ui.button("Close", color=None, on_click=on_close).props(
            "unelevated no-caps"
        ).classes("up-btn up-btn--secondary")

    if picture is not None:
        _overlay(detail, picture, layers)
        with ui.element("div").classes("up-layers"):

            def toggle(key: str, value: bool) -> None:
                layers[key] = value
                on_layer()

            ui.checkbox(
                "Registered objects and matches",
                value=layers.get("regions", True),
                on_change=lambda e: toggle("regions", bool(e.value)),
            )
            ui.checkbox(
                "Other objects",
                value=layers.get("open", True),
                on_change=lambda e: toggle("open", bool(e.value)),
            )
    else:
        ui.label(NO_PICTURE_TEXT).classes("up-empty")

    # Facts
    facts: list[str] = [f"Size {frame.width} x {frame.height} pixels"]
    if detail.blur_score is not None:
        facts.append(f"Sharpness {detail.blur_score:.0f} (higher is sharper)")
    if frame.detail:
        facts.append(f"Note: {frame.detail}")
    with ui.element("ul").classes("up-facts"):
        for fact in facts:
            ui.label(fact)

    ui.label("Registered objects").classes("up-section-title")
    if detail.sightings:
        with ui.element("ul").classes("up-facts"):
            for s in detail.sightings:
                text: str = (
                    f"{s.object_name} - match {s.similarity:.2f} - "
                    f"{'strong' if s.strong else 'weak'} - box {s.region_id}"
                )
                if s.in_hand:
                    text += " - in a hand"
                ui.label(text)
    else:
        ui.label(NO_MATCH_TEXT).classes("up-detail__meta")
    if detail.last_seen_of:
        ui.label(
            "This frame is the last place where Saarthi saw: "
            + ", ".join(detail.last_seen_of)
        ).classes("up-detail__meta")
    if detail.regions:
        with ui.element("ul").classes("up-facts up-facts--small"):
            for region in detail.regions:
                ui.label(region_text(region))
    else:
        ui.label(NO_REGIONS_TEXT).classes("up-detail__meta")

    ui.label("Other objects").classes("up-section-title")
    ui.label(OPEN_HINT).classes("up-detail__meta")
    if detail.open_boxes:
        with ui.element("ul").classes("up-facts up-facts--small"):
            for box in detail.open_boxes:
                ui.label(open_text(box))
    else:
        ui.label(NO_OPEN_TEXT).classes("up-detail__meta")

    if detail.descriptions:
        ui.label("Description").classes("up-section-title")
        for name, text in detail.descriptions:
            ui.label(f"{name}: {text}").classes("up-detail__meta")

    ui.label("Room").classes("up-section-title")
    if detail.rooms:
        ui.label(
            ", ".join(f"{room} {score:.2f}" for room, score in detail.rooms)
        ).classes("up-detail__meta")
    else:
        ui.label(NO_ROOM_TEXT).classes("up-detail__meta")
