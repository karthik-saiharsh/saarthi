"""Badges: a text label and a colour for a status."""

from nicegui import ui

from ui.models import ItemStatus
from ui.ports import PhotoState

STATUS_LABELS: dict[ItemStatus, str] = {
    ItemStatus.WAITING: "Waiting",
    ItemStatus.SENDING: "Sending",
    ItemStatus.STORED: "Stored",
    ItemStatus.DUPLICATE: "Already stored",
    ItemStatus.REJECTED: "Rejected",
    ItemStatus.FAILED: "Failed",
}

PHOTO_LABELS: dict[PhotoState, str] = {
    PhotoState.PENDING: "Waiting",
    PhotoState.PROCESSING: "Looking",
    PhotoState.LOCATED: "Check the crop",
    PhotoState.CONFIRMED: "Accepted",
    PhotoState.FAILED: "Not found",
}


def badge(label: str, variant: str) -> ui.label:
    """Build a badge. The CSS class `up-badge--<variant>` sets the colour."""
    assert label and variant, "label and variant are required"
    return ui.label(label).classes(f"up-badge up-badge--{variant}")


def status_badge(status: ItemStatus) -> ui.label:
    """Build a badge. The text is always shown, so colour is not the only signal."""
    assert isinstance(status, ItemStatus), "status must be an ItemStatus"
    return badge(STATUS_LABELS[status], status.value)


def photo_badge(state: PhotoState) -> ui.label:
    """Build a badge for the state of a reference photo."""
    assert isinstance(state, PhotoState), "state must be a PhotoState"
    return badge(PHOTO_LABELS[state], f"photo-{state.value}")
