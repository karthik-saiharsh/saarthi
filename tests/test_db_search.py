"""Tests of the helpers that a search uses. They use the actual database and index.

Every image of a test has a path that starts with `__test__/`. The cleanup removes
the vectors and the images of the test, also when the test fails.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta

import numpy as np
import pytest

import db.api as db
from db.schema import Image, ImageStatus, Region
from tests.db_support import (
    DIMENSION,
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
    unit_vectors,
)

MODEL: str = "test-model"
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


def make_frame(
    seen_at: datetime,
    count: int = 1,
    status: ImageStatus = ImageStatus.DONE,
    model: str | None = MODEL,
) -> tuple[Image, list[Region]]:
    """A frame with `count` regions. The regions have vectors if `model` is given."""
    image: Image = make_image()
    regions: list[Region] = make_regions(image, count)
    if model is not None:
        db.store_region_vectors(
            [r.id for r in regions], unit_vectors(count, seed=image.id), model
        )
    set_status(image, status)
    Image.update(captured_at=seen_at).where(Image.id == image.id).execute()
    return image, regions


def listed(since: datetime, until: datetime, model: str = MODEL) -> list[Region]:
    return [
        r
        for r in db.list_searchable_regions(since, until, model, 10000)
        if r.image.file_path.startswith(TEST_PREFIX)
    ]


def test_regions_are_listed_newest_frame_first_with_their_frame() -> None:
    old, _ = make_frame(NOW - timedelta(minutes=30), 2)
    new, _ = make_frame(NOW - timedelta(minutes=5), 1)
    found: list[Region] = listed(NOW - timedelta(hours=1), NOW)
    assert [r.image.id for r in found] == [new.id, old.id, old.id]
    assert found[1].id < found[2].id  # The regions of one frame are ordered by id


def test_the_window_includes_the_end_and_excludes_the_start() -> None:
    inside, _ = make_frame(NOW)
    edge, _ = make_frame(NOW - timedelta(hours=1))
    after, _ = make_frame(NOW + timedelta(seconds=1))
    ids: set[int] = {r.image.id for r in listed(NOW - timedelta(hours=1), NOW)}
    assert ids == {inside.id}
    assert edge.id not in ids and after.id not in ids


def test_only_done_frames_with_good_vectors_of_the_model_are_listed() -> None:
    good, regions = make_frame(NOW)
    make_frame(NOW, status=ImageStatus.REJECTED)
    make_frame(NOW, model=None)  # No vector
    other, _ = make_frame(NOW, model="other-model")
    Region.update(quality_ok=False).where(Region.id == regions[0].id).execute()
    assert listed(NOW - timedelta(minutes=1), NOW) == []  # The only good one is marked
    Region.update(quality_ok=True).where(Region.id == regions[0].id).execute()
    assert [r.image.id for r in listed(NOW - timedelta(minutes=1), NOW)] == [good.id]
    assert [
        r.image.id for r in listed(NOW - timedelta(minutes=1), NOW, "other-model")
    ] == [other.id]


def test_the_limit_cuts_the_oldest_regions() -> None:
    make_frame(NOW - timedelta(minutes=10))
    new, _ = make_frame(NOW - timedelta(minutes=1))
    result: list[Region] = db.list_searchable_regions(
        NOW - timedelta(hours=1), NOW, MODEL, 1
    )
    assert [r.image.id for r in result] == [new.id]


def test_region_vectors_come_back_in_the_order_of_the_ids() -> None:
    _, regions = make_frame(NOW, 3)
    ids: list[int] = [regions[2].id, regions[0].id]
    stored: np.ndarray = db.get_region_vectors([r.id for r in regions])
    assert stored.shape == (3, DIMENSION)
    assert np.allclose(db.get_region_vectors(ids), stored[[2, 0]])


def test_a_region_without_a_vector_is_refused() -> None:
    _, regions = make_frame(NOW, model=None)
    with pytest.raises(db.VectorIndexError):
        db.get_region_vectors([regions[0].id])


def test_a_wrong_window_is_refused() -> None:
    with pytest.raises(AssertionError):
        db.list_searchable_regions(NOW, NOW - timedelta(seconds=1), MODEL)
