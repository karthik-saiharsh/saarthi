import base64

import pytest

from ui.components import find_ask, find_choices, find_result
from ui.components.app_nav import PAGES
from ui.components.enroll_guide import STEPS
from ui.components.find_result import age_text, picture_source, sighting_lines
from ui.components.object_detail import (
    NOT_FOUND_TEXT,
    UNREADABLE_TEXT,
    failure_text,
    image_source,
)
from ui.components.object_list import photo_summary
from ui.components.status_badge import PHOTO_LABELS
from ui.ports import PhotoState, Sighting
from ui.routes import ENROLL_ROUTE, FIND_ROUTE, UPLOAD_ROUTE


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
    routes: list[str] = [UPLOAD_ROUTE, ENROLL_ROUTE, FIND_ROUTE]
    assert len(set(routes)) == 3 and all(r.startswith("/") for r in routes)


def test_every_page_is_in_the_navigation_and_find_comes_first() -> None:
    assert [route for route, _ in PAGES] == [FIND_ROUTE, UPLOAD_ROUTE, ENROLL_ROUTE]


def test_help_steps_are_short() -> None:
    # The help text follows the writing rule: short sentences.
    for step in STEPS:
        for sentence in step.split(". "):
            assert len(sentence.split()) <= 25, sentence


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "just now"),
        (59, "just now"),
        (60, "1 minute ago"),
        (119, "1 minute ago"),
        (12 * 60, "12 minutes ago"),
        (3599, "59 minutes ago"),
        (3600, "1 hour ago"),
        (3 * 3600 + 5, "3 hours ago"),
        (86399, "23 hours ago"),
        (86400, "1 day ago"),
        (2 * 86400, "2 days ago"),
        (30 * 86400, "30 days ago"),
    ],
)
def test_age_text_is_short_and_plain(seconds: int, text: str) -> None:
    assert age_text(seconds) == text


def test_age_text_refuses_a_negative_age() -> None:
    with pytest.raises(AssertionError):
        age_text(-1)


def test_the_lines_of_a_sure_sighting_are_only_the_time() -> None:
    sighting: Sighting = Sighting(720, False, True, False, None)
    assert sighting_lines(sighting) == ["Seen 12 minutes ago."]


def test_the_lines_say_maybe_in_hand_and_old_when_that_is_true() -> None:
    sighting: Sighting = Sighting(7200, True, False, True, None)
    lines: list[str] = sighting_lines(sighting)
    assert lines[0] == "Seen 2 hours ago." and len(lines) == 4
    assert find_result.IN_HAND_TEXT in lines
    assert find_result.MAYBE_TEXT in lines
    assert find_result.STALE_TEXT in lines


def test_picture_source_is_a_jpeg_data_uri() -> None:
    source: str = picture_source(b"\xff\xd8data")
    assert source.startswith("data:image/jpeg;base64,")
    with pytest.raises(AssertionError):
        picture_source(b"")


def test_the_texts_of_the_find_page_are_short() -> None:
    # The same writing rule as the help: short sentences. The page is for people who may be confused.
    texts: list[str] = [
        find_result.NO_MATCH_TEXT,
        find_result.NOT_SEEN_TEXT,
        find_result.INCOMPLETE_TEXT,
        find_result.MAYBE_TEXT,
        find_result.IN_HAND_TEXT,
        find_result.STALE_TEXT,
        find_choices.EMPTY_TEXT,
        find_ask.HINT,
    ]
    for text in texts:
        for sentence in text.split(". "):
            assert len(sentence.split()) <= 20, sentence
