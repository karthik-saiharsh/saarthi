"""Tests of the sighting log and the last-seen record. They use the actual database.

Every object of a test has a name that starts with `__test__ `, and every image has a
path that starts with `__test__/`. The objects go first (their sightings and their
last-seen records go by cascade), then the images.
"""

import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest

import db.api as db
from db.api import gallery, sightings
from db.schema import EnrolledObject, Image, ImageStatus, LastSeen, Sighting
from tests.db_support import (
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
)

PREFIX: str = "__test__ "
NOW: datetime = datetime(2000, 1, 1, 12, 0, 0)
BOX: tuple[int, int, int, int] = (10, 20, 60, 80)


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    yield
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    ids: list[int] = [
        i.id for i in Image.select().where(Image.file_path.startswith(TEST_PREFIX))
    ]
    db.delete_images(ids)


def make_object() -> EnrolledObject:
    return gallery.create_object(f"{PREFIX}{uuid.uuid4().hex[:8]}", "a bottle")


def make_frame(seen_at: datetime = NOW) -> tuple[Image, int]:
    """A `done` image with one region. Return the image and the region id."""
    image: Image = make_image()
    region_id: int = make_regions(image, 1)[0].id
    set_status(image, ImageStatus.DONE)
    Image.update(captured_at=seen_at).where(Image.id == image.id).execute()
    return image, region_id


def add(object_id: int, seen_at: datetime = NOW, strong: bool = False) -> int:
    image, region_id = make_frame(seen_at)
    sightings.add_sighting(
        object_id, image.id, region_id, seen_at, 0.9, 0.5, strong, False
    )
    return image.id


def test_count_recent_counts_frames_inside_the_window_only() -> None:
    obj = make_object()
    for seconds in (-120, -60, -30, 0, 30):
        add(obj.id, NOW + timedelta(seconds=seconds))
    window = timedelta(seconds=60)
    assert sightings.count_recent(obj.id, NOW - window, NOW) == 3  # -60, -30, 0
    assert sightings.count_recent(obj.id, NOW - window, NOW + window) == 4


def test_count_recent_counts_one_frame_one_time() -> None:
    obj = make_object()
    image, region_id = make_frame()
    for _ in range(2):  # Two rows for the same frame
        sightings.add_sighting(obj.id, image.id, region_id, NOW, 0.9, 0.5, False, False)
    assert sightings.count_recent(obj.id, NOW, NOW) == 1


def test_count_recent_ignores_other_objects() -> None:
    first, second = make_object(), make_object()
    add(first.id)
    assert sightings.count_recent(second.id, NOW, NOW) == 0


def test_the_last_seen_record_is_set_and_replaced_by_a_newer_sighting() -> None:
    obj = make_object()
    first, _ = make_frame(NOW)
    second, _ = make_frame(NOW + timedelta(seconds=10))
    assert sightings.get_last_seen(obj.id) is None
    assert sightings.set_last_seen(obj.id, first.id, BOX, NOW, 0.8, 0.4, False, True)
    record: LastSeen = sightings.get_last_seen(obj.id)  # type: ignore[assignment]
    assert (record.image_id, record.in_hand, record.strong) == (first.id, True, False)
    assert (record.x0, record.y0, record.x1, record.y1) == BOX
    later = NOW + timedelta(seconds=10)
    assert sightings.set_last_seen(obj.id, second.id, BOX, later, 0.9, 0.5, True, False)
    record = sightings.get_last_seen(obj.id)  # type: ignore[assignment]
    assert (record.image_id, record.strong, record.seen_at) == (second.id, True, later)
    assert len(sightings.list_last_seen()) >= 1
    assert LastSeen.select().where(LastSeen.enrolled_object == obj.id).count() == 1


def test_an_older_sighting_does_not_replace_a_newer_record() -> None:
    obj = make_object()
    new, _ = make_frame(NOW)
    old, _ = make_frame(NOW - timedelta(minutes=5))
    assert sightings.set_last_seen(obj.id, new.id, BOX, NOW, 0.8, 0.4, False, False)
    older = NOW - timedelta(minutes=5)
    assert not sightings.set_last_seen(
        obj.id, old.id, BOX, older, 0.9, 0.5, True, False
    )
    assert sightings.get_last_seen(obj.id).image_id == new.id  # type: ignore[union-attr]


def test_the_frame_of_a_last_seen_record_cannot_be_deleted_or_expire() -> None:
    obj = make_object()
    image, _ = make_frame(NOW)
    sightings.set_last_seen(obj.id, image.id, BOX, NOW, 0.8, 0.4, False, False)
    cutoff = NOW + timedelta(days=1)
    assert image.id not in [i.id for i in db.list_images_older_than(cutoff, 10000)]
    with pytest.raises(db.InvalidStateError, match="last-seen"):
        db.delete_images([image.id])
    assert db.get_image(image.id) is not None


def test_the_frame_can_go_after_the_record_is_replaced() -> None:
    obj = make_object()
    old, _ = make_frame(NOW)
    new, _ = make_frame(NOW + timedelta(seconds=5))
    sightings.set_last_seen(obj.id, old.id, BOX, NOW, 0.8, 0.4, False, False)
    later = NOW + timedelta(seconds=5)
    sightings.set_last_seen(obj.id, new.id, BOX, later, 0.8, 0.4, False, False)
    assert old.id in [
        i.id for i in db.list_images_older_than(NOW + timedelta(days=1), 10000)
    ]
    db.delete_images([old.id])
    assert db.get_image(old.id) is None


def test_deleting_the_object_removes_its_sightings_and_its_record() -> None:
    obj = make_object()
    image_id: int = add(obj.id)
    sightings.set_last_seen(obj.id, image_id, BOX, NOW, 0.8, 0.4, False, False)
    gallery.delete_object(obj.id)
    assert not Sighting.select().where(Sighting.enrolled_object == obj.id).exists()
    assert sightings.get_last_seen(obj.id) is None
    db.delete_images([image_id])  # The frame is free now


def test_deleting_a_frame_removes_its_sightings() -> None:
    obj = make_object()
    image_id: int = add(obj.id)
    db.delete_images([image_id])
    assert sightings.count_recent(obj.id, NOW, NOW) == 0


def test_a_sighting_of_an_unknown_object_is_refused() -> None:
    image, region_id = make_frame()
    with pytest.raises(db.InvalidRecordError):
        sightings.add_sighting(2**31, image.id, region_id, NOW, 0.9, 0.5, False, False)
    with pytest.raises(db.InvalidRecordError):
        sightings.set_last_seen(2**31, image.id, BOX, NOW, 0.9, 0.5, False, False)


def test_a_wrong_box_is_refused_by_the_database() -> None:
    obj = make_object()
    image, _ = make_frame()
    with pytest.raises(db.InvalidRecordError):
        sightings.set_last_seen(
            obj.id, image.id, (50, 20, 10, 80), NOW, 0.9, 0.5, False, False
        )
