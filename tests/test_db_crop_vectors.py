"""Tests of the one table for all boxes: the boxes of the broad detector and the crop vectors of each region.

They use the actual database and the actual indexes. Every image of a test has a path that starts with `__test__/`.
The cleanup removes the images, their regions and all their vectors.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta

import numpy as np
import pytest
from peewee import SqliteDatabase

import db.api as db
from db.api import crop_vectors
from db.errors import DbError, VectorIndexError
from db.schema import Image, ImageStatus, Region, RegionSource
from tests.db_support import (
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
)

CROP_MODEL: str = "test-crop-model"
KEY_MODEL: str = "test-key-model"
NOW: datetime = datetime(2000, 1, 1, 12, 0, 0)


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    yield
    mine = Image.file_path.startswith(TEST_PREFIX)
    # A test can leave a frame in `processing`. Such a frame cannot be deleted, so it is set to `done` first.
    Image.update(status=ImageStatus.DONE.value).where(mine).execute()
    db.delete_images([i.id for i in Image.select().where(mine)])


def unit(count: int, size: int, seed: int = 0) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(count, size)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def crops(count: int, seed: int = 0) -> np.ndarray:
    return unit(count, db.current_config().crop_vector_dimension, seed)


def keys(count: int, seed: int = 0) -> np.ndarray:
    return unit(count, db.current_config().key_vector_dimension, seed)


def broad(label: str = "laptop", confidence: float = 0.9) -> db.NewRegion:
    return db.NewRegion(
        10, 20, 110, 220, confidence, RegionSource.BROAD, [], label=label
    )


def frame_with(
    regions: list[db.NewRegion],
    seconds: int = 0,
    status: ImageStatus = ImageStatus.DONE,
    crop_model: str | None = CROP_MODEL,
    key_model: str | None = KEY_MODEL,
) -> tuple[Image, list[Region]]:
    """A frame with these regions, each with a region vector and the crop vectors of the given models."""
    image: Image = make_image()
    rows: list[Region] = db.add_regions(image.id, regions)
    ids: list[int] = [r.id for r in rows]
    db.store_region_vectors(ids, unit(len(ids), 384), "test-model")
    crop_vectors.store(
        ids,
        crops(len(ids)) if crop_model else None,
        crop_model,
        keys(len(ids)) if key_model else None,
        key_model,
    )
    set_status(image, status)
    Image.update(captured_at=NOW + timedelta(seconds=seconds)).where(
        Image.id == image.id
    ).execute()
    return db.get_image(image.id), db.get_regions(image.id)  # type: ignore[return-value]


# ---- The rows ----


def test_a_box_of_the_broad_detector_needs_a_name_and_a_candidate_has_none() -> None:
    with pytest.raises(AssertionError):
        db.NewRegion(0, 0, 10, 10, 0.5, RegionSource.BROAD, [])
    with pytest.raises(AssertionError):
        db.NewRegion(0, 0, 10, 10, 0.5, RegionSource.BROAD, [], label="x" * 81)
    with pytest.raises(AssertionError):
        db.NewRegion(0, 0, 10, 10, 0.5, RegionSource.FULL, ["a cup"], label="cup")


def test_broad_boxes_are_stored_with_their_name_most_confident_first() -> None:
    image, _ = frame_with([broad("laptop", 0.6), broad("bottle", 0.95)])
    assert [(r.label, r.source) for r in db.get_broad_regions(image.id)] == [
        ("bottle", "broad"),
        ("laptop", "broad"),
    ]


# ---- The vectors ----


def test_crop_vectors_are_stored_marked_and_read_back() -> None:
    image = make_image()
    rows = make_regions(image, 2)
    ids = [r.id for r in rows]
    first, second = crops(2, seed=1), keys(2, seed=2)
    crop_vectors.store(ids, first, CROP_MODEL, second, KEY_MODEL)
    assert np.allclose(crop_vectors.get_crop_vectors(ids), first, atol=1e-6)
    assert np.allclose(crop_vectors.get_key_vectors(ids), second, atol=1e-6)
    assert {(r.crop_model, r.key_model) for r in db.get_regions(image.id)} == {
        (CROP_MODEL, KEY_MODEL)
    }


def test_only_keys_can_be_stored_when_the_scene_model_is_off() -> None:
    image = make_image()
    ids = [r.id for r in make_regions(image, 1)]
    crop_vectors.store(ids, None, None, keys(1), KEY_MODEL)
    assert db.get_regions(image.id)[0].crop_model is None
    with pytest.raises(VectorIndexError):
        crop_vectors.get_crop_vectors(ids)


def test_a_bad_vector_adds_nothing_to_either_index() -> None:
    image = make_image()
    ids = [r.id for r in make_regions(image, 1)]
    before = crop_vectors.counts()
    with pytest.raises(VectorIndexError):
        crop_vectors.store(ids, crops(1), CROP_MODEL, keys(1) * 2.0, KEY_MODEL)
    assert crop_vectors.counts() == before
    assert db.get_regions(image.id)[0].crop_model is None


def test_an_unknown_region_is_refused_and_leaves_no_vector() -> None:
    before = crop_vectors.counts()
    with pytest.raises(DbError):
        crop_vectors.store([2_000_000_000], crops(1), CROP_MODEL, keys(1), KEY_MODEL)
    assert crop_vectors.counts() == before


def test_delete_images_removes_the_crop_vectors() -> None:
    image, rows = frame_with([broad()])
    before = crop_vectors.counts()
    db.delete_images([image.id])
    assert crop_vectors.counts() == (before[0] - 1, before[1] - 1)
    with pytest.raises(VectorIndexError):
        crop_vectors.get_crop_vectors([rows[0].id])


def test_recovery_removes_orphans_and_clears_marks_without_a_vector() -> None:
    image, rows = frame_with([broad(), broad("cup")])
    crop_vectors._indexes()[0].remove([rows[0].id])  # A stop lost one vector
    removed, cleared = crop_vectors.recover()
    assert cleared == 1
    assert db.get_regions(image.id)[0].crop_model is None
    assert db.get_regions(image.id)[1].crop_model == CROP_MODEL
    assert removed == 0


# ---- The lists of the searches ----


def test_the_open_search_lists_broad_boxes_of_done_frames_and_one_model() -> None:
    old, _ = frame_with([broad()], -120)
    new, _ = frame_with(
        [broad(), db.NewRegion(1, 1, 9, 9, 0.5, RegionSource.FULL, [])], -10
    )
    pending, _ = frame_with([broad()], -5, ImageStatus.PENDING)
    other, _ = frame_with([broad()], -1, crop_model="another-model")
    rows = crop_vectors.list_broad_regions(
        NOW - timedelta(seconds=300), NOW, CROP_MODEL
    )
    mine = [r for r in rows if r.image_id in {old.id, new.id, pending.id, other.id}]
    assert [r.image_id for r in mine] == [new.id, old.id]
    assert all(r.source == "broad" for r in mine)


def test_the_keys_search_lists_all_boxes_with_both_vectors() -> None:
    _, rows = frame_with(
        [
            broad(),
            db.NewRegion(1, 1, 9, 9, 0.5, RegionSource.FULL, [], quality_ok=False),
        ]
    )
    without_keys, _ = frame_with([broad()], key_model=None)
    listed = crop_vectors.list_keyed_regions(
        NOW - timedelta(seconds=60), NOW + timedelta(seconds=60), CROP_MODEL, KEY_MODEL
    )
    ids = {r.id for r in listed}
    assert {r.id for r in rows} <= ids  # The box with a bad crop is listed too
    assert all(r.image_id != without_keys.id for r in listed)


def test_the_tau_and_evidence_search_does_not_list_broad_boxes() -> None:
    image, _ = frame_with(
        [broad(), db.NewRegion(1, 1, 9, 9, 0.5, RegionSource.FULL, [])]
    )
    listed = db.list_searchable_regions(
        NOW - timedelta(seconds=60), NOW + timedelta(seconds=60), "test-model"
    )
    assert [r.source for r in listed if r.image_id == image.id] == ["full"]


# ---- The schema ----


def test_a_database_with_the_old_region_table_is_refused() -> None:
    old = SqliteDatabase(":memory:")
    old.connect()
    old.execute_sql("CREATE TABLE region (id INTEGER PRIMARY KEY, x0 INTEGER)")
    with pytest.raises(DbError, match="./clear"):
        db._check_schema(old)
    old.close()
