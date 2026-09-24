"""Tests of the gallery helpers. They use the actual database.

Every object of a test has a name that starts with `__test__ `. The objects are
removed at the end (photos and views go by cascade). No file is written, only paths.
"""

import uuid
from collections.abc import Iterator

import numpy as np
import pytest

import db.api as db
from db.api import gallery
from db.schema import (
    EnrolledObject,
    ObjectStatus,
    ObjectView,
    PhotoStatus,
    ReferencePhoto,
)
from tests.db_support import skip_if_actual_data_has_problems

PREFIX: str = "__test__ "
MODEL: str = "test-model"
OTHER_MODEL: str = "other-model"
DIMENSION: int = 384


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, _ = db.init_db()
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    yield
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()


def make_object(tracked: bool = True, **changes: object) -> EnrolledObject:
    name: str = changes.pop("name", f"{PREFIX}{uuid.uuid4().hex[:8]}")  # type: ignore[assignment]
    return gallery.create_object(name, "a bottle", ["a flask"], tracked)


def make_photo(obj: EnrolledObject, sha: str | None = None) -> ReferencePhoto:
    return gallery.add_reference_photo(
        obj.id, f"{PREFIX}{uuid.uuid4().hex}.jpg", sha or uuid.uuid4().hex * 2, 640, 480
    )


def unit(seed: int, size: int = DIMENSION) -> np.ndarray:
    vector: np.ndarray = (
        np.random.default_rng(seed).normal(size=size).astype(np.float32)
    )
    return vector / np.linalg.norm(vector)


def located(
    obj: EnrolledObject, seeds: tuple[int, ...] = (1, 2), model: str = MODEL
) -> ReferencePhoto:
    """A photo that is `located` with one view for each seed."""
    photo: ReferencePhoto = make_photo(obj)
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()
    return gallery.store_photo_views(
        photo.id, (10, 10, 200, 300), 0.4, [(f"v{s}", unit(s)) for s in seeds], model
    )


def skip_if_real_pending_photos() -> None:
    real = ReferencePhoto.select().where(
        ReferencePhoto.status == PhotoStatus.PENDING.value,
        ~ReferencePhoto.file_path.startswith(PREFIX),
    )
    if real.exists():
        pytest.skip("The actual database has pending reference photos")


# ---- Objects ----


def test_an_object_starts_as_a_draft_and_keeps_its_phrases() -> None:
    obj: EnrolledObject = gallery.create_object(
        f"{PREFIX}bottle", "a bottle", ["a flask", "a jug"], False
    )
    loaded: EnrolledObject | None = gallery.get_object(obj.id)
    assert loaded is not None
    assert loaded.status == ObjectStatus.DRAFT.value and loaded.tau is None
    assert loaded.synonyms == ["a flask", "a jug"] and loaded.tracked is False
    assert obj.id in [o.id for o in gallery.list_objects()]


def test_a_duplicate_name_is_refused() -> None:
    obj: EnrolledObject = make_object()
    with pytest.raises(db.DuplicateObjectError):
        gallery.create_object(obj.name, "a cup")


def test_unknown_object_gives_none() -> None:
    assert gallery.get_object(2**31) is None


def test_the_parameters_are_checked() -> None:
    with pytest.raises(AssertionError):
        gallery.create_object(" ", "a bottle")
    with pytest.raises(AssertionError):
        gallery.get_object(0)
    with pytest.raises(AssertionError):
        gallery.create_object("x", "a bottle", ["ok", ""])


# ---- Photos ----


def test_a_photo_starts_pending() -> None:
    photo: ReferencePhoto = make_photo(make_object())
    assert photo.status == PhotoStatus.PENDING.value and photo.x0 is None


def test_the_same_picture_twice_for_one_object_is_refused_but_not_for_two_objects() -> (
    None
):
    first, second = make_object(), make_object()
    sha: str = uuid.uuid4().hex * 2
    make_photo(first, sha)
    with pytest.raises(db.DuplicateImageError):
        make_photo(first, sha)
    make_photo(second, sha)  # Another object may use the same picture


def test_a_photo_for_an_unknown_object_is_refused() -> None:
    with pytest.raises(db.InvalidStateError):
        gallery.add_reference_photo(2**31, f"{PREFIX}x.jpg", "a" * 64, 10, 10)


def test_the_oldest_pending_photo_is_claimed_once() -> None:
    skip_if_real_pending_photos()
    obj: EnrolledObject = make_object()
    first, second = make_photo(obj), make_photo(obj)
    assert gallery.claim_next_reference_photo().id == first.id  # type: ignore[union-attr]
    assert gallery.claim_next_reference_photo().id == second.id  # type: ignore[union-attr]
    assert gallery.claim_next_reference_photo() is None
    assert gallery.get_photo(first.id).status == PhotoStatus.PROCESSING.value  # type: ignore[union-attr]


def test_views_are_stored_with_the_box_and_the_photo_is_located() -> None:
    obj: EnrolledObject = make_object()
    photo: ReferencePhoto = located(obj, seeds=(1, 2, 3))
    assert photo.status == PhotoStatus.LOCATED.value
    assert (photo.x0, photo.y0, photo.x1, photo.y1) == (
        10,
        10,
        200,
        300,
    ) and photo.det_score == 0.4
    assert ObjectView.select().where(ObjectView.photo == photo.id).count() == 3


def test_storing_views_needs_a_processing_photo_and_leaves_nothing_if_it_fails() -> (
    None
):
    obj: EnrolledObject = make_object()
    pending: ReferencePhoto = make_photo(obj)
    with pytest.raises(db.InvalidStateError):
        gallery.store_photo_views(
            pending.id, (1, 1, 5, 5), 0.5, [("v", unit(1))], MODEL
        )
    assert ObjectView.select().where(ObjectView.photo == pending.id).count() == 0


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.full(DIMENSION, np.nan, dtype=np.float32),
        np.full(DIMENSION, 2.0, dtype=np.float32),  # Not unit length
        np.ones(10, dtype=np.float32) / np.sqrt(10),  # Wrong size
    ],
    ids=["not finite", "not unit length", "wrong size"],
)
def test_a_bad_view_is_refused_and_the_photo_stays_processing(
    bad_vector: np.ndarray,
) -> None:
    photo: ReferencePhoto = make_photo(make_object())
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()
    with pytest.raises(db.InvalidRecordError):
        gallery.store_photo_views(
            photo.id, (1, 1, 5, 5), 0.5, [("v", bad_vector)], MODEL
        )
    assert gallery.get_photo(photo.id).status == PhotoStatus.PROCESSING.value  # type: ignore[union-attr]


def test_a_bad_box_is_refused() -> None:
    photo: ReferencePhoto = make_photo(make_object())
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()
    with pytest.raises(db.InvalidRecordError):
        gallery.store_photo_views(
            photo.id, (10, 10, 5, 5), 0.5, [("v", unit(1))], MODEL
        )
    assert ObjectView.select().where(ObjectView.photo == photo.id).count() == 0


def test_a_photo_can_fail_with_a_cut_reason_and_be_retried() -> None:
    photo: ReferencePhoto = make_photo(make_object())
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()
    failed: ReferencePhoto = gallery.mark_photo_failed(photo.id, "x" * 5000)
    assert (
        failed.status == PhotoStatus.FAILED.value and len(failed.status_detail) == 500
    )
    retried: ReferencePhoto = gallery.retry_photo(photo.id)
    assert retried.status == PhotoStatus.PENDING.value and retried.status_detail is None
    with pytest.raises(db.InvalidStateError):
        gallery.retry_photo(photo.id)  # It is not failed any more


def test_confirm_needs_a_located_photo() -> None:
    obj: EnrolledObject = make_object()
    with pytest.raises(db.InvalidStateError):
        gallery.confirm_photo(make_photo(obj).id)
    assert gallery.confirm_photo(located(obj).id).status == PhotoStatus.CONFIRMED.value


def test_removing_a_photo_returns_its_path_and_removes_its_views() -> None:
    photo: ReferencePhoto = located(make_object())
    assert gallery.remove_photo(photo.id) == photo.file_path
    assert gallery.get_photo(photo.id) is None
    assert ObjectView.select().where(ObjectView.photo == photo.id).count() == 0
    assert gallery.remove_photo(photo.id) is None  # Unknown id


def test_a_photo_in_use_cannot_be_removed() -> None:
    photo: ReferencePhoto = make_photo(make_object())
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()
    with pytest.raises(db.InvalidStateError):
        gallery.remove_photo(photo.id)


def test_photo_counts_by_status() -> None:
    obj: EnrolledObject = make_object()
    make_photo(obj)
    gallery.confirm_photo(located(obj).id)
    counts = gallery.count_photos_by_status(obj.id)
    assert counts[PhotoStatus.PENDING] == 1 and counts[PhotoStatus.CONFIRMED] == 1
    assert counts[PhotoStatus.FAILED] == 0


# ---- Views ----


def test_only_confirmed_views_of_one_model_are_loaded_with_their_photo_ids() -> None:
    obj: EnrolledObject = make_object()
    confirmed: ReferencePhoto = gallery.confirm_photo(located(obj, (1, 2)).id)
    located(obj, (3,))  # Not confirmed
    other_model: ReferencePhoto = gallery.confirm_photo(
        located(obj, (4,), model=OTHER_MODEL).id
    )
    vectors, photo_ids = gallery.load_views(obj.id, MODEL)
    assert vectors.shape == (2, DIMENSION) and vectors.dtype == np.float32
    assert photo_ids == [confirmed.id, confirmed.id]
    assert np.allclose(vectors[0], unit(1)) and np.allclose(vectors[1], unit(2))
    assert gallery.load_views(obj.id, OTHER_MODEL)[1] == [other_model.id]


def test_no_views_give_an_empty_array_of_the_right_width() -> None:
    vectors, photo_ids = gallery.load_views(make_object().id, MODEL)
    assert vectors.shape == (0, DIMENSION) and photo_ids == []


def test_the_views_of_other_objects_are_the_negatives() -> None:
    mine, other = make_object(), make_object(tracked=False)
    gallery.confirm_photo(located(mine, (1,)).id)
    gallery.confirm_photo(located(other, (7, 8)).id)
    located(other, (9,))  # Not confirmed
    negatives: np.ndarray = gallery.load_other_views(mine.id, MODEL)
    assert negatives.shape[1] == DIMENSION
    assert any(np.allclose(row, unit(7)) for row in negatives)
    assert not any(np.allclose(row, unit(1)) for row in negatives)  # Not its own views
    assert not any(np.allclose(row, unit(9)) for row in negatives)  # Not confirmed


# ---- Calibration and phrases ----


def test_calibration_makes_an_object_ready_and_can_be_cleared() -> None:
    obj: EnrolledObject = make_object()
    ready: EnrolledObject = gallery.set_calibration(
        obj.id, 0.6, 0.65, 0.03, MODEL, "a warning"
    )
    assert ready.status == ObjectStatus.READY.value
    assert (ready.tau, ready.tau_strong, ready.margin_min) == (0.6, 0.65, 0.03)
    assert ready.calibration_model == MODEL and ready.calibration_note == "a warning"
    draft: EnrolledObject = gallery.clear_calibration(obj.id, "needs 3 photos")
    assert draft.status == ObjectStatus.DRAFT.value and draft.tau is None
    assert draft.calibration_note == "needs 3 photos"
    with pytest.raises(db.InvalidStateError):
        gallery.set_calibration(2**31, 0.6, 0.65, 0.03, MODEL)


def test_the_detector_phrases_come_from_ready_objects_only_without_repeats() -> None:
    skip_if_actual_data_has_problems()
    before: list[str] = gallery.gallery_phrases()
    ready = gallery.create_object(
        f"{PREFIX}a", "a test mug zzz", ["a test cup zzz", "a test mug zzz"]
    )
    look_alike = gallery.create_object(
        f"{PREFIX}b", "a test cup zzz", [], tracked=False
    )
    gallery.create_object(f"{PREFIX}c", "a test draft zzz")  # A draft: not listed
    gallery.set_calibration(ready.id, 0.6, 0.65, 0.03, MODEL)
    gallery.set_calibration(look_alike.id, 0.6, 0.65, 0.03, MODEL)
    added: list[str] = [p for p in gallery.gallery_phrases() if p not in before]
    assert added == [
        "a test mug zzz",
        "a test cup zzz",
    ]  # Once each, the look-alike included


def test_deleting_an_object_returns_the_paths_and_removes_everything() -> None:
    obj: EnrolledObject = make_object()
    photos: list[ReferencePhoto] = [located(obj), make_photo(obj)]
    expected: list[str] = sorted(p.file_path for p in photos)
    assert sorted(gallery.delete_object(obj.id)) == expected
    assert gallery.get_object(obj.id) is None
    assert (
        ReferencePhoto.select().where(ReferencePhoto.enrolled_object == obj.id).count()
        == 0
    )
    assert ObjectView.select().where(ObjectView.enrolled_object == obj.id).count() == 0
    assert gallery.delete_object(obj.id) == []  # Unknown id


def test_an_object_with_a_photo_in_use_cannot_be_deleted() -> None:
    obj: EnrolledObject = make_object()
    photo: ReferencePhoto = make_photo(obj)
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()
    with pytest.raises(db.InvalidStateError):
        gallery.delete_object(obj.id)
    assert gallery.get_object(obj.id) is not None


# ---- Paths and recovery ----


def test_all_file_paths_include_the_reference_photos() -> None:
    photo: ReferencePhoto = make_photo(make_object())
    paths: list[str] = db.list_all_file_paths()
    assert photo.file_path in paths and paths == sorted(paths)
    assert set(db.list_image_paths()) <= set(
        paths
    )  # The image files are in the list too


def test_the_recovery_sets_photos_in_process_back_to_pending() -> None:
    skip_if_actual_data_has_problems()
    photo: ReferencePhoto = make_photo(make_object())
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()
    assert db.recover_interrupted().reset_photo_ids == [photo.id]
    assert gallery.get_photo(photo.id).status == PhotoStatus.PENDING.value  # type: ignore[union-attr]
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()
    report: db.StartupReport = db.startup_recovery()
    assert report.interrupted_photo_ids == [photo.id]
    assert db.startup_recovery() == db.StartupReport(
        [], [], 0, 0
    )  # Nothing left to repair
