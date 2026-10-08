"""Tests of the memory browser and its UI adapter. They use the actual database and storage."""

import asyncio
from collections.abc import Iterator

import pytest

import db.api as db
import storage.api as storage
from backend.browser import MemoryBrowser
from backend.errors import InvalidInputError, NotFoundError
from backend.ui_adapter import UiBrowseService
from db.schema import EnrolledObject, Image
from tests.browse_support import full_frame
from tests.db_support import make_image, skip_if_actual_data_has_problems
from tests.search_support import PREFIX, store_frame
from ui.ports import BrowseError


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    db.init_db()
    skip_if_actual_data_has_problems()
    before: set[str] = set(db.list_image_paths())
    yield
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        paths = db.delete_images([i.id for i in made])
        # A frame made with `make_image` has a fake path and no file.
        storage.delete_images([p for p in paths if not p.startswith("__test__/")])


def test_the_page_has_the_frames_the_total_and_the_counts() -> None:
    frame, _ = full_frame(20)
    page = MemoryBrowser().list_frames()
    assert page.frames[0].id == frame.id and page.total == sum(page.counts.values())
    summary = page.frames[0]
    assert summary.status == "done" and 19 * 60 <= summary.age_seconds <= 21 * 60
    assert (summary.regions, summary.sightings, summary.open_boxes) == (1, 1, 1)
    assert (summary.width, summary.height) == (80, 60)


def test_the_filter_changes_the_total() -> None:
    store_frame(10)
    rejected = make_image()
    db.mark_image_rejected(rejected.id, "too blurry")
    browser = MemoryBrowser()
    page = browser.list_frames("rejected")
    assert page.total == page.counts["rejected"] and rejected.id in [
        f.id for f in page.frames
    ]


@pytest.mark.parametrize(
    ("status", "offset", "limit"),
    [("nope", 0, 10), (None, -1, 10), (None, 0, 0), (None, 0, 101)],
)
def test_a_wrong_request_is_refused(
    status: str | None, offset: int, limit: int
) -> None:
    with pytest.raises(InvalidInputError):
        MemoryBrowser().list_frames(status, offset, limit)


def test_the_detail_view_has_the_pipeline_data() -> None:
    frame, obj = full_frame(15, "my bottle")
    view = MemoryBrowser().get_frame(frame.id)
    assert view.summary.id == frame.id and view.last_seen_of == [obj.name]
    assert view.regions[0].phrases == ["a bottle"] and view.regions[0].quality_ok
    assert [(s.object_name, s.strong) for s in view.sightings] == [(obj.name, True)]
    assert (
        view.open_boxes[0].label == "laptop"
        and view.open_boxes[0].linked_region_id == view.regions[0].id
    )
    assert view.rooms == [("a bedroom", 0.7), ("a corridor", 0.2), ("a kitchen", 0.1)]


def test_an_unknown_frame_is_not_found() -> None:
    with pytest.raises(NotFoundError):
        MemoryBrowser().get_frame(2_000_000_000)
    with pytest.raises(NotFoundError):
        MemoryBrowser().get_picture(2_000_000_000)


def test_the_picture_is_a_jpeg_of_the_whole_frame() -> None:
    frame = store_frame(5)
    picture = MemoryBrowser().get_picture(frame.id, 64)
    assert picture[:2] == b"\xff\xd8"


def test_a_missing_file_is_not_found() -> None:
    frame = store_frame(5)
    storage.delete_images([frame.file_path])
    with pytest.raises(NotFoundError):
        MemoryBrowser().get_picture(frame.id)


# ---- The UI adapter ----


def test_the_adapter_gives_the_ui_types() -> None:
    frame, obj = full_frame(9, "my bottle")
    service = UiBrowseService(MemoryBrowser())
    page = asyncio.run(service.list_frames("done", 0, 5))
    assert page.frames[0].id == frame.id and page.counts["done"] >= 1
    detail = asyncio.run(service.get_frame(frame.id))
    assert detail.frame.sightings == 1 and detail.last_seen_of == (obj.name,)
    assert (
        detail.regions[0].phrases == ("a bottle",)
        and detail.open_boxes[0].label == "laptop"
    )
    assert asyncio.run(service.get_picture(frame.id, 64))[:2] == b"\xff\xd8"


def test_the_adapter_maps_errors_to_fixed_texts() -> None:
    service = UiBrowseService(MemoryBrowser())
    with pytest.raises(BrowseError, match="does not exist"):
        asyncio.run(service.get_frame(2_000_000_000))
    with pytest.raises(BrowseError, match="status"):
        asyncio.run(service.list_frames("nope", 0, 5))
