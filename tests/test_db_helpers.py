"""Tests of the query helpers. They run on the actual database from config.toml.

Every test makes its own rows with the path prefix `__test__/` and removes them.
"""

import uuid
from collections.abc import Iterator
from datetime import timedelta

import pytest

from db.api import (
    DuplicateImageError,
    InvalidStateError,
    NewRegion,
    RecoveryReport,
    add_image,
    add_regions,
    claim_next_pending,
    count_images_by_status,
    find_image_by_sha256,
    get_image,
    get_regions,
    init_db,
    list_embedded_region_ids,
    list_recent_phashes,
    mark_image_done,
    mark_image_failed,
    mark_image_rejected,
    mark_regions_embedded,
    recover_interrupted,
    set_image_analysis,
)
from db.schema import Image, ImageStatus, Region, RegionSource, utc_now

TEST_PREFIX: str = "__test__/"


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    """Open the actual database. Remove leftover test rows before and after."""
    database, _ = init_db()
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()
    yield
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()
    database.close()


@pytest.fixture(autouse=True)
def clean_test_rows() -> Iterator[None]:
    yield
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()


def new_image(**overrides: object) -> Image:
    fields: dict[str, object] = {
        "file_path": f"{TEST_PREFIX}{uuid.uuid4().hex}.jpg",
        "sha256": uuid.uuid4().hex + uuid.uuid4().hex,
        "width": 640,
        "height": 480,
    }
    fields.update(overrides)
    return add_image(**fields)


def set_status(image: Image, status: ImageStatus) -> None:
    Image.update(status=status.value).where(Image.id == image.id).execute()


def skip_if_real_rows(status: ImageStatus) -> None:
    """Skip a test that would act on real rows with this status."""
    real: int = (
        Image.select()
        .where(Image.status == status.value, ~Image.file_path.startswith(TEST_PREFIX))
        .count()
    )
    if real:
        pytest.skip(f"Actual database has {real} real '{status.value}' image(s)")


def test_add_image_defaults() -> None:
    image: Image = new_image(room="kitchen")
    loaded: Image | None = get_image(image.id)
    assert loaded is not None
    assert loaded.status == ImageStatus.PENDING.value
    assert loaded.room == "kitchen"
    assert loaded.captured_at is not None


def test_duplicate_file_path_and_sha256_raise() -> None:
    first: Image = new_image()
    with pytest.raises(DuplicateImageError):
        new_image(file_path=first.file_path)
    with pytest.raises(DuplicateImageError):
        new_image(sha256=first.sha256)


@pytest.mark.parametrize(
    "overrides",
    [
        {"file_path": "/abs/a.jpg"},
        {"file_path": "__test__/../a.jpg"},
        {"sha256": "short"},
        {"width": 0},
        {"height": -1},
    ],
)
def test_add_image_rejects_bad_parameters(overrides: dict[str, object]) -> None:
    with pytest.raises(AssertionError):
        new_image(**overrides)


def test_find_image_by_sha256() -> None:
    image: Image = new_image()
    found: Image | None = find_image_by_sha256(image.sha256)
    assert found is not None and found.id == image.id
    assert find_image_by_sha256("0" * 64) is None


def test_get_image_missing_returns_none() -> None:
    assert get_image(2**31) is None


def test_claim_next_pending_takes_oldest_once() -> None:
    skip_if_real_rows(ImageStatus.PENDING)
    assert claim_next_pending() is None
    first: Image = new_image()
    second: Image = new_image()
    claimed: Image | None = claim_next_pending()
    assert claimed is not None and claimed.id == first.id
    assert claimed.status == ImageStatus.PROCESSING.value
    again: Image | None = claim_next_pending()
    assert again is not None and again.id == second.id
    assert claim_next_pending() is None


@pytest.mark.parametrize(
    ("mark", "argument", "status"),
    [
        (mark_image_done, None, ImageStatus.DONE),
        (mark_image_rejected, "too blurry", ImageStatus.REJECTED),
        (mark_image_failed, "model error", ImageStatus.FAILED),
    ],
)
def test_mark_image_from_processing(mark, argument, status: ImageStatus) -> None:
    image: Image = new_image()
    set_status(image, ImageStatus.PROCESSING)
    finished: Image = mark(image.id) if argument is None else mark(image.id, argument)
    assert finished.status == status.value
    assert finished.processed_at is not None
    assert finished.status_detail == argument


def test_mark_image_needs_processing_status() -> None:
    image: Image = new_image()  # status is pending
    with pytest.raises(InvalidStateError):
        mark_image_done(image.id)
    with pytest.raises(InvalidStateError):
        mark_image_done(2**31)


def test_status_detail_is_cut() -> None:
    image: Image = new_image()
    set_status(image, ImageStatus.PROCESSING)
    finished: Image = mark_image_failed(image.id, "x" * 5000)
    assert len(finished.status_detail) == 500


def new_region(**overrides: object) -> NewRegion:
    fields: dict[str, object] = {
        "x0": 10,
        "y0": 20,
        "x1": 110,
        "y1": 120,
        "det_score": 0.31,
        "source": RegionSource.FULL,
        "phrases": ["a bottle"],
    }
    fields.update(overrides)
    return NewRegion(**fields)


def processing_image() -> Image:
    image: Image = new_image()
    set_status(image, ImageStatus.PROCESSING)
    return image


def test_set_image_analysis() -> None:
    image: Image = processing_image()
    updated: Image = set_image_analysis(image.id, "ab12cd34ef56ab12", 153.5)
    assert updated.phash == "ab12cd34ef56ab12"
    assert updated.blur_score == 153.5


def test_set_image_analysis_needs_processing() -> None:
    with pytest.raises(InvalidStateError):
        set_image_analysis(new_image().id, "ab12", 1.0)


def test_add_and_get_regions() -> None:
    image: Image = processing_image()
    rows: list[Region] = add_regions(
        image.id, [new_region(), new_region(x0=0, phrases=["a cup"], in_hand=True)]
    )
    assert len(rows) == 2 and rows[0].id < rows[1].id
    assert all(row.embedding_model is None for row in rows)
    loaded: list[Region] = get_regions(image.id)
    assert [r.id for r in loaded] == [r.id for r in rows]
    assert loaded[1].phrases == ["a cup"] and loaded[1].in_hand is True


def test_add_regions_empty_list_is_allowed() -> None:
    image: Image = processing_image()
    assert add_regions(image.id, []) == []
    assert get_regions(image.id) == []


def test_add_regions_needs_processing() -> None:
    with pytest.raises(InvalidStateError):
        add_regions(new_image().id, [new_region()])
    with pytest.raises(InvalidStateError):
        add_regions(2**31, [new_region()])


@pytest.mark.parametrize("overrides", [{"x1": 5}, {"y0": -1}, {"det_score": 1}])
def test_new_region_rejects_bad_values(overrides: dict[str, object]) -> None:
    with pytest.raises(AssertionError):
        new_region(**overrides)


def test_mark_regions_embedded_and_list() -> None:
    image: Image = processing_image()
    rows: list[Region] = add_regions(image.id, [new_region(), new_region(x0=1)])
    before: set[int] = set(list_embedded_region_ids())
    assert mark_regions_embedded([rows[0].id], "dinov2-base") == 1
    after: set[int] = set(list_embedded_region_ids())
    assert after - before == {rows[0].id}
    assert get_regions(image.id)[1].embedding_model is None


def test_mark_regions_embedded_changes_nothing_on_missing_id() -> None:
    image: Image = processing_image()
    rows: list[Region] = add_regions(image.id, [new_region()])
    with pytest.raises(InvalidStateError):
        mark_regions_embedded([rows[0].id, 2**31], "dinov2-base")
    assert get_regions(image.id)[0].embedding_model is None


def test_recover_interrupted_resets_image_and_reports_stale_vectors() -> None:
    skip_if_real_rows(ImageStatus.PROCESSING)
    image: Image = processing_image()
    set_image_analysis(image.id, "ab12", 1.0)
    rows: list[Region] = add_regions(image.id, [new_region(), new_region(x0=1)])
    mark_regions_embedded([rows[0].id], "dinov2-base")
    Image.update(status_detail="old", processed_at=utc_now()).where(
        Image.id == image.id
    ).execute()

    report: RecoveryReport = recover_interrupted()

    assert report.reset_image_ids == [image.id]
    assert report.stale_vector_ids == [rows[0].id]
    recovered: Image | None = get_image(image.id)
    assert recovered.status == ImageStatus.PENDING.value
    assert recovered.status_detail is None and recovered.processed_at is None
    assert get_regions(image.id) == []
    # A second call has nothing left to reset.
    second: RecoveryReport = recover_interrupted()
    assert second.reset_image_ids == [] and second.stale_vector_ids == []


def test_recover_interrupted_reports_done_regions_without_vector() -> None:
    skip_if_real_rows(ImageStatus.PROCESSING)
    image: Image = processing_image()
    rows: list[Region] = add_regions(image.id, [new_region(), new_region(x0=1)])
    mark_regions_embedded([rows[0].id], "dinov2-base")
    mark_image_done(image.id)

    report: RecoveryReport = recover_interrupted()

    assert rows[1].id in report.unembedded_region_ids
    assert rows[0].id not in report.unembedded_region_ids
    assert report.reset_image_ids == []
    assert get_image(image.id).status == ImageStatus.DONE.value
    assert len(get_regions(image.id)) == 2


def test_list_recent_phashes_returns_done_images_newest_first() -> None:
    skip_if_real_rows(ImageStatus.DONE)
    now = utc_now()
    old: Image = new_image()
    mid: Image = new_image()
    new: Image = new_image()
    pending: Image = new_image()
    for image, phash, age, status in [
        (old, "a" * 16, timedelta(hours=5), ImageStatus.DONE),
        (mid, "b" * 16, timedelta(hours=2), ImageStatus.DONE),
        (new, "c" * 16, timedelta(minutes=1), ImageStatus.DONE),
        (pending, "d" * 16, timedelta(minutes=2), ImageStatus.PENDING),
    ]:
        Image.update(phash=phash, captured_at=now - age, status=status.value).where(
            Image.id == image.id
        ).execute()
    assert list_recent_phashes() == ["c" * 16, "b" * 16, "a" * 16]
    assert list_recent_phashes(limit=2) == ["c" * 16, "b" * 16]
    assert list_recent_phashes(since=now - timedelta(hours=3)) == ["c" * 16, "b" * 16]
    assert list_recent_phashes(since=now) == []


def test_list_recent_phashes_ignores_done_images_without_a_hash() -> None:
    skip_if_real_rows(ImageStatus.DONE)
    image: Image = new_image()
    set_status(image, ImageStatus.DONE)
    assert list_recent_phashes() == []


def test_count_images_by_status_counts_only_that_status() -> None:
    pending_before: int = count_images_by_status(ImageStatus.PENDING)
    done_before: int = count_images_by_status(ImageStatus.DONE)
    new_image()
    done: Image = new_image()
    set_status(done, ImageStatus.DONE)
    assert count_images_by_status(ImageStatus.PENDING) == pending_before + 1
    assert count_images_by_status(ImageStatus.DONE) == done_before + 1
    with pytest.raises(AssertionError):
        count_images_by_status("pending")  # type: ignore[arg-type]
