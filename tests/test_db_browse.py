"""Tests of the read-only views for the memory browser. They use the actual database and storage."""

from collections.abc import Iterator

import pytest

import db.api as db
import storage.api as storage
from db.schema import EnrolledObject, Image, ImageStatus
from tests.browse_support import full_frame
from tests.db_support import make_image, set_status, skip_if_actual_data_has_problems
from tests.search_support import PREFIX, store_frame


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


def ids(rows) -> list[int]:
    return [r.image.id for r in rows]


def test_the_frames_come_newest_first_with_the_counts() -> None:
    old, _ = full_frame(30)
    new = store_frame(5)  # A frame with one region and nothing else
    rows = db.browse.list_frames(None, 10, 0)
    assert ids(rows)[:2] == [new.id, old.id]
    by_id = {r.image.id: r for r in rows}
    assert (
        by_id[old.id].regions,
        by_id[old.id].sightings,
        by_id[old.id].open_boxes,
    ) == (1, 1, 1)
    assert (
        by_id[new.id].regions,
        by_id[new.id].sightings,
        by_id[new.id].open_boxes,
    ) == (1, 0, 0)


def test_the_filter_and_the_page_limit_work() -> None:
    done = store_frame(10)
    rejected = make_image()
    db.mark_image_rejected(rejected.id, "too blurry")
    assert done.id in ids(db.browse.list_frames("done", 100, 0))
    assert done.id not in ids(db.browse.list_frames("rejected", 100, 0))
    assert rejected.id in ids(db.browse.list_frames("rejected", 100, 0))
    first = db.browse.list_frames(None, 1, 0)
    second = db.browse.list_frames(None, 1, 1)
    assert (
        len(first) == 1 and len(second) == 1 and first[0].image.id != second[0].image.id
    )


def test_the_status_counts_have_every_status() -> None:
    store_frame(10)
    counts = db.browse.count_by_status()
    assert set(counts) == {s.value for s in ImageStatus} and counts["done"] >= 1


def test_the_detail_has_everything_that_the_pipeline_stored() -> None:
    frame, obj = full_frame(12, "my bottle")
    detail = db.browse.get_frame_detail(frame.id)
    assert detail is not None and detail.image.id == frame.id
    assert len(detail.regions) == 1 and detail.regions[0].phrases == ["a bottle"]
    assert [(s.object_name, s.sighting.strong) for s in detail.sightings] == [
        (obj.name, True)
    ]
    assert [(o.label, o.source) for o in detail.open_regions] == [("laptop", "broad")]
    assert (
        len(detail.regions) == 1
    )  # The box of the broad detector is not in the candidates
    assert detail.rooms is not None and detail.rooms["a bedroom"] == 0.7
    assert detail.last_seen_of == [obj.name]


def test_a_frame_without_extras_has_empty_lists_and_no_rooms() -> None:
    frame = store_frame(8)
    detail = db.browse.get_frame_detail(frame.id)
    assert detail is not None
    assert detail.sightings == [] and detail.open_regions == []
    assert detail.rooms is None and detail.last_seen_of == []


def test_an_unknown_frame_gives_none() -> None:
    assert db.browse.get_frame_detail(2_000_000_000) is None


def test_a_rejected_frame_has_its_reason() -> None:
    frame = make_image()
    db.mark_image_rejected(frame.id, "too blurry")
    detail = db.browse.get_frame_detail(frame.id)
    assert detail is not None and detail.image.status_detail == "too blurry"
    set_status(frame, ImageStatus.REJECTED)


def test_the_limit_is_checked() -> None:
    with pytest.raises(AssertionError):
        db.browse.list_frames(None, 0, 0)
    with pytest.raises(AssertionError):
        db.browse.list_frames(None, 101, 0)
    with pytest.raises(AssertionError):
        db.browse.list_frames("unknown", 10, 0)
