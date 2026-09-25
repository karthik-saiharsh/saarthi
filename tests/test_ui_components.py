import base64

import pytest

from ui.components.enroll_guide import STEPS
from ui.components.object_detail import (
    NOT_FOUND_TEXT,
    UNREADABLE_TEXT,
    failure_text,
    image_source,
)
from ui.components.object_list import photo_summary
from ui.components.status_badge import PHOTO_LABELS
from ui.ports import PhotoState
from ui.routes import ENROLL_ROUTE, UPLOAD_ROUTE


def test_photo_summary_counts_the_states() -> None:
    counts: dict[PhotoState, int] = {
        PhotoState.CONFIRMED: 3,
        PhotoState.LOCATED: 1,
        PhotoState.PENDING: 1,
        PhotoState.PROCESSING: 2,
        PhotoState.FAILED: 0,
    }
    assert photo_summary(counts) == "3 accepted, 1 to check, 3 waiting"


def test_photo_summary_without_photos() -> None:
    assert photo_summary({}) == "No photos"
    assert photo_summary({PhotoState.FAILED: 0}) == "No photos"


def test_failure_text_hides_internal_reasons() -> None:
    assert failure_text("object not found") == NOT_FOUND_TEXT
    assert failure_text("InferenceError") == UNREADABLE_TEXT
    assert failure_text(None) == UNREADABLE_TEXT


def test_image_source_is_a_jpeg_data_uri() -> None:
    source: str = image_source(b"\xff\xd8data")
    prefix: str = "data:image/jpeg;base64,"
    assert source.startswith(prefix)
    assert base64.b64decode(source[len(prefix) :]) == b"\xff\xd8data"
    with pytest.raises(AssertionError):
        image_source(b"")


def test_every_photo_state_has_a_label() -> None:
    assert set(PHOTO_LABELS) == set(PhotoState)


def test_routes_are_different_and_absolute() -> None:
    assert UPLOAD_ROUTE != ENROLL_ROUTE
    assert UPLOAD_ROUTE.startswith("/") and ENROLL_ROUTE.startswith("/")


def test_help_steps_are_short() -> None:
    # The help text follows the writing rule: short sentences.
    for step in STEPS:
        for sentence in step.split(". "):
            assert len(sentence.split()) <= 25, sentence
