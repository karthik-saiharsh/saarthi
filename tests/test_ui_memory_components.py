"""Tests of the text and the geometry of the memory page. No browser is needed."""

import pytest

from ui.components.frame_detail import box_style, open_text, region_text
from ui.components.memory_list import FILTERS, STATUS_LABELS, frame_summary
from ui.ports import FrameInfo, OpenBoxInfo, RegionInfo


def frame(regions: int = 1, sightings: int = 0, open_boxes: int = 2) -> FrameInfo:
    return FrameInfo(1, "done", None, 60, 100, 50, regions, sightings, open_boxes)


def test_the_summary_line_counts_each_kind() -> None:
    assert frame_summary(frame(1, 1, 7)) == "1 match, 1 region, 7 other boxes"
    assert frame_summary(frame(0, 0, 1)) == "0 matches, 0 regions, 1 other box"
    assert frame_summary(frame(3, 2, 0)) == "2 matches, 3 regions, 0 other boxes"


def test_a_box_is_placed_in_percent_of_the_frame() -> None:
    assert (
        box_style((10, 5, 60, 25), 100, 50)
        == "left:10.00%;top:10.00%;width:50.00%;height:40.00%"
    )
    assert (
        box_style((0, 0, 100, 50), 100, 50)
        == "left:0.00%;top:0.00%;width:100.00%;height:100.00%"
    )


def test_a_box_outside_the_frame_is_refused() -> None:
    with pytest.raises(AssertionError):
        box_style((10, 5, 101, 25), 100, 50)
    with pytest.raises(AssertionError):
        box_style((10, 5, 10, 25), 100, 50)


def test_the_text_of_a_region_says_why_a_crop_is_not_used() -> None:
    region = RegionInfo(7, (0, 0, 5, 5), ("a bottle",), 0.82, "full", True, False)
    text = region_text(region)
    assert (
        "Box 7: a bottle" in text
        and "0.82" in text
        and "in a hand" in text
        and "not used" in text
    )
    plain = region_text(RegionInfo(8, (0, 0, 5, 5), (), 0.5, "generic", False, True))
    assert "no words" in plain and "not used" not in plain and "in a hand" not in plain


def test_the_text_of_an_open_box_shows_the_link() -> None:
    assert (
        open_text(OpenBoxInfo(1, (0, 0, 5, 5), "laptop", 0.9, None)) == "laptop - 0.90"
    )
    assert "the same place as box 7" in open_text(
        OpenBoxInfo(1, (0, 0, 5, 5), "bottle", 0.8, 7)
    )


def test_every_filter_has_a_label_and_the_statuses_are_known() -> None:
    assert [s for s, _ in FILTERS if s is not None] == list(STATUS_LABELS)
    assert FILTERS[0] == (None, "All")
