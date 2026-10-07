"""Tests of the boxes of the broad detector and their vectors. They use the actual database and the actual open index.

Every image of a test has a path that starts with `__test__/`. The cleanup removes the images, their boxes and their vectors.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import db.api as db
from db.api import open_regions
from db.errors import ConfigError, InvalidRecordError, VectorIndexError
from db.open_regions import NewOpenRegion
from db.schema import Image, ImageStatus, OpenRegion, Region
from tests.db_support import (
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
)

MODEL: str = "test-open-model"
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
    ids: list[int] = [
        i.id for i in Image.select().where(Image.file_path.startswith(TEST_PREFIX))
    ]
    db.delete_images(ids)


def dimension() -> int:
    return db.current_config().open_vector_dimension


def unit_vectors(count: int, seed: int = 0) -> np.ndarray:
    raw = (
        np.random.default_rng(seed).normal(size=(count, dimension())).astype(np.float32)
    )
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def make_frame(seconds: int = 0, status: ImageStatus = ImageStatus.DONE) -> Image:
    image: Image = make_image()
    set_status(image, status)
    Image.update(captured_at=NOW + timedelta(seconds=seconds)).where(
        Image.id == image.id
    ).execute()
    return db.get_image(image.id)  # type: ignore[return-value]


def box(
    label: str = "laptop", confidence: float = 0.9, region_id: int | None = None
) -> NewOpenRegion:
    return NewOpenRegion(10, 20, 110, 220, label, confidence, region_id)


# ---- Adding and reading ----


def test_boxes_and_vectors_are_added_and_read_back() -> None:
    frame = make_frame()
    vectors = unit_vectors(2)
    rows = open_regions.add_open_regions(
        frame.id, [box("laptop", 0.9), box("bottle", 0.95)], vectors, MODEL
    )
    assert [r.label for r in rows] == ["laptop", "bottle"]
    stored = open_regions.get_vectors([r.id for r in rows])
    assert np.allclose(stored, vectors, atol=1e-6)
    # The most confident box comes first.
    assert [r.label for r in open_regions.get_open_regions(frame.id)] == [
        "bottle",
        "laptop",
    ]
    assert rows[0].vector_model == MODEL and (rows[0].x0, rows[0].y1) == (10, 220)


def test_a_wrong_vector_is_refused_and_leaves_no_row() -> None:
    frame = make_frame()
    before = open_regions.count()
    bad = unit_vectors(1) * 2.0  # Not unit length
    with pytest.raises(VectorIndexError):
        open_regions.add_open_regions(frame.id, [box()], bad, MODEL)
    assert open_regions.get_open_regions(frame.id) == []
    assert open_regions.count() == before


def test_an_unknown_frame_is_refused_and_adds_no_vector() -> None:
    before = open_regions.count()
    with pytest.raises(InvalidRecordError):
        open_regions.add_open_regions(2_000_000_000, [box()], unit_vectors(1), MODEL)
    assert open_regions.count() == before


def test_new_open_region_checks_its_values() -> None:
    with pytest.raises(AssertionError):
        NewOpenRegion(10, 10, 10, 20, "laptop", 0.5)  # An empty box
    with pytest.raises(AssertionError):
        NewOpenRegion(0, 0, 10, 10, "  ", 0.5)  # No label
    with pytest.raises(AssertionError):
        NewOpenRegion(0, 0, 10, 10, "laptop", 1.5)  # A confidence above 1
    with pytest.raises(AssertionError):
        NewOpenRegion(0, 0, 10, 10, "x" * 81, 0.5)  # A long label


# ---- The window of a search ----


def test_the_search_lists_done_frames_of_the_model_in_the_window_newest_first() -> None:
    old = make_frame(-120)
    new = make_frame(-10)
    pending = make_frame(-5, ImageStatus.PENDING)
    for frame in (old, new, pending):
        open_regions.add_open_regions(frame.id, [box()], unit_vectors(1), MODEL)
    other_model = make_frame(-1)
    open_regions.add_open_regions(
        other_model.id, [box()], unit_vectors(1), "another-model"
    )

    rows = open_regions.list_open_regions(NOW - timedelta(seconds=300), NOW, MODEL)
    mine = [
        r for r in rows if r.image_id in {old.id, new.id, pending.id, other_model.id}
    ]
    assert [r.image_id for r in mine] == [
        new.id,
        old.id,
    ]  # Not pending, not the other model
    narrow = open_regions.list_open_regions(NOW - timedelta(seconds=60), NOW, MODEL)
    assert [r.image_id for r in narrow if r.image_id in {old.id, new.id}] == [new.id]


# ---- Removing ----


def test_deleting_a_frame_removes_its_boxes_and_vectors() -> None:
    frame = make_frame()
    rows = open_regions.add_open_regions(
        frame.id, [box(), box("cup")], unit_vectors(2), MODEL
    )
    ids = [r.id for r in rows]
    before = open_regions.count()
    db.delete_images([frame.id])
    assert OpenRegion.select().where(OpenRegion.id.in_(ids)).count() == 0
    assert open_regions.count() == before - 2


def test_drop_for_images_removes_boxes_and_vectors_only() -> None:
    frame = make_frame()
    open_regions.add_open_regions(frame.id, [box()], unit_vectors(1), MODEL)
    before = open_regions.count()
    assert open_regions.drop_for_images([frame.id]) == 1
    assert open_regions.get_open_regions(frame.id) == []
    assert open_regions.count() == before - 1
    assert db.get_image(frame.id) is not None  # The frame stays
    assert (
        open_regions.drop_for_images([frame.id]) == 0
    )  # A second call has nothing to do


def test_a_removed_registered_region_keeps_the_open_box() -> None:
    frame = make_image()  # Regions can be added to a frame that is `processing`
    region = make_regions(frame, 1)[0]
    set_status(
        frame, ImageStatus.DONE
    )  # The cleanup cannot delete a frame in `processing`
    open_regions.add_open_regions(
        frame.id, [box(region_id=region.id)], unit_vectors(1), MODEL
    )
    Region.delete().where(Region.id == region.id).execute()
    row = open_regions.get_open_regions(frame.id)[0]
    assert row.region_id is None


# ---- Recovery ----


def test_recover_removes_orphan_vectors_and_deletes_boxes_without_vector() -> None:
    frame = make_frame()
    row = OpenRegion.create(
        image=frame.id,
        x0=1,
        y0=1,
        x1=5,
        y1=5,
        label="lost",
        confidence=0.5,
        vector_model=MODEL,
    )  # A row with no vector
    orphan_id = row.id + 1_000_000
    open_regions._require_index().add(
        [orphan_id], unit_vectors(1, seed=5)
    )  # A vector with no row
    removed, deleted = open_regions.recover()
    assert removed >= 1 and deleted >= 1
    assert OpenRegion.select().where(OpenRegion.id == row.id).count() == 0
    assert orphan_id not in open_regions._require_index().ids()


def test_the_start_up_recovery_clears_the_boxes_of_an_interrupted_frame() -> None:
    frame = (
        make_image()
    )  # The status is `processing`, as after a stop in the middle of a frame
    open_regions.add_open_regions(frame.id, [box()], unit_vectors(1), MODEL)
    before = open_regions.count()
    report = db.startup_recovery()
    assert frame.id in report.interrupted_image_ids
    assert open_regions.get_open_regions(frame.id) == []
    assert open_regions.count() == before - 1
    assert db.get_image(frame.id).status == ImageStatus.PENDING.value  # type: ignore[union-attr]


# ---- The settings ----


def test_the_config_needs_the_open_keys(tmp_path: Path) -> None:
    text = (db.DEFAULT_CONFIG_PATH).read_text()
    for old, new in (
        ("open_dimension = 768", "open_dimension = 0"),
        (
            'open_faiss_index = "data/open_regions.faiss"',
            'open_faiss_index = "/tmp/x.faiss"',
        ),
        (
            'open_faiss_index = "data/open_regions.faiss"',
            'open_faiss_index = "../x.faiss"',
        ),
        ("open_dimension = 768", ""),
    ):
        assert old in text
        path = tmp_path / "config.toml"
        path.write_text(text.replace(old, new, 1))
        with pytest.raises(ConfigError):
            db.load_config(path)
