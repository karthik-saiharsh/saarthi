"""Tests of the enrollment service. They use the actual database and storage, and a fake ai engine.

Every object of a test has a name that starts with `__test__ `. The objects and
their files are removed at the end.
"""

import dataclasses
import uuid
from collections.abc import Iterator
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.calibration import calibrate
from ai.settings import load_config as load_ai_config
from backend.enrollment import (
    MAX_SYNONYMS,
    EnrollmentService,
    ObjectDetail,
    ObjectSummary,
    PhotoInfo,
    clean_name,
    clean_phrase,
)
from backend.errors import InvalidInputError, LimitReachedError, NotFoundError
from backend.settings import BackendConfig, load_config
from db.schema import PhotoStatus, ReferencePhoto
from tests.ai_support import make_engine

PREFIX: str = "__test__ "
MODEL: str = "fake-embedder@00000000"
DIMENSION: int = 384


def picture(seed: int) -> bytes:
    rng: np.random.Generator = np.random.default_rng(seed)
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
        buffer, "JPEG"
    )
    return buffer.getvalue()


def unique_name() -> str:
    return f"{PREFIX}{uuid.uuid4().hex[:8]}"


def near(base_seed: int, noise_seed: int, scale: float = 0.3) -> np.ndarray:
    """A unit vector near the vector of `base_seed`."""
    base: np.ndarray = np.random.default_rng(base_seed).normal(size=DIMENSION)
    noisy: np.ndarray = base + np.random.default_rng(noise_seed).normal(
        scale=scale, size=DIMENSION
    )
    return (noisy / np.linalg.norm(noisy)).astype(np.float32)


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    yield
    for obj in db.gallery.list_objects():
        if obj.name.startswith(PREFIX):
            storage.delete_images(db.gallery.delete_object(obj.id))
    database.close()


class Woken:
    def __init__(self) -> None:
        self.count: int = 0

    def __call__(self) -> None:
        self.count += 1


@pytest.fixture()
def woken() -> Woken:
    return Woken()


@pytest.fixture()
def service(woken: Woken) -> EnrollmentService:
    return EnrollmentService(load_config(), make_engine(), on_photo_added=woken)


def new_object(service: EnrollmentService, **kwargs: object) -> ObjectSummary:
    return service.create_object(unique_name(), kwargs.pop("phrase", "a bottle"), **kwargs)  # type: ignore[arg-type]


def located_photo(
    service: EnrollmentService, obj: ObjectSummary, seed: int, vector: np.ndarray
) -> PhotoInfo:
    """Add a photo, and give it a view as the worker would. The photo is `located`."""
    info: PhotoInfo = service.add_photo(obj.id, picture(seed))
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == info.id
    ).execute()
    db.gallery.store_photo_views(
        info.id, (5, 5, 40, 40), 0.5, [("original", vector)], MODEL
    )
    return PhotoInfo(info.id, PhotoStatus.LOCATED.value, None, (5, 5, 40, 40), 0.5)


def ready_object(
    service: EnrollmentService, base_seed: int, first_photo_seed: int
) -> ObjectSummary:
    obj: ObjectSummary = new_object(service)
    for k in range(3):
        photo: PhotoInfo = located_photo(
            service, obj, first_photo_seed + k, near(base_seed, k)
        )
        service.confirm_photo(obj.id, photo.id)
    return service.get_object(obj.id).summary


# ---- The words of the carer ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a bottle", "a bottle"),
        ("  A   Water  Bottle ", "a water bottle"),
        ("a pair of glasses", "a pair of glasses"),
        ("mum's keys", "mum's keys"),
        ("a t-shirt", "a t-shirt"),
        ("a 5 liter jug", "a 5 liter jug"),
        ("bottle\ncup\tlid", "bottle cup lid"),  # Any white space becomes one space
    ],
)
def test_a_good_phrase_is_cleaned(text: str, expected: str) -> None:
    assert clean_phrase(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "a bottle.",
        "a bottle. a cup",
        "a bottle!",
        "a über bottle",
        "a very long phrase with too many words",
        "a" * 41,
        "-bottle",
        "a  - b",
        "bottle'",
    ],
)
def test_a_bad_phrase_is_refused(text: str) -> None:
    with pytest.raises(InvalidInputError):
        clean_phrase(text)


def test_names() -> None:
    assert clean_name("  my   blue bottle ") == "my blue bottle"
    assert clean_name("Mum’s résumé") == "Mum’s résumé"
    for bad in ("", "  ", "x" * 81, "a\x00b", "a\x1bb"):
        with pytest.raises(InvalidInputError):
            clean_name(bad)


# ---- Objects ----


def test_a_new_object_is_a_draft_that_needs_photos(service: EnrollmentService) -> None:
    summary: ObjectSummary = service.create_object(
        unique_name(), " A Bottle ", ["a flask", "A  flask", "a bottle"], False
    )
    assert summary.status == "draft" and summary.category_phrase == "a bottle"
    assert (
        summary.synonyms == ["a flask"] and summary.tracked is False
    )  # Cleaned, no repeat
    assert summary.note == "Needs 3 more confirmed photo(s)."
    assert summary.photo_counts["pending"] == 0 and summary.tau is None


def test_too_many_synonyms_are_refused(service: EnrollmentService) -> None:
    many: list[str] = [f"a thing{chr(97 + i)}" for i in range(MAX_SYNONYMS + 1)]
    with pytest.raises(InvalidInputError):
        service.create_object(unique_name(), "a bottle", many)


def test_a_duplicate_name_is_refused(service: EnrollmentService) -> None:
    name: str = unique_name()
    service.create_object(name, "a bottle")
    with pytest.raises(db.DuplicateObjectError):
        service.create_object(
            f"  {name} ", "a cup"
        )  # The spaces do not make a new name


def test_the_number_of_objects_is_limited() -> None:
    base: BackendConfig = load_config()
    count: int = db.gallery.count_objects()
    limited: BackendConfig = dataclasses.replace(
        base, enrollment=dataclasses.replace(base.enrollment, max_objects=count + 1)
    )
    service: EnrollmentService = EnrollmentService(limited, make_engine())
    service.create_object(unique_name(), "a bottle")
    with pytest.raises(LimitReachedError):
        service.create_object(unique_name(), "a cup")


def test_unknown_objects_and_photos_are_not_found(service: EnrollmentService) -> None:
    with pytest.raises(NotFoundError):
        service.get_object(2**31)
    with pytest.raises(NotFoundError):
        service.add_photo(2**31, picture(1))
    obj: ObjectSummary = new_object(service)
    with pytest.raises(NotFoundError):
        service.confirm_photo(obj.id, 2**31)


def test_the_list_and_the_detail_show_the_photos(service: EnrollmentService) -> None:
    obj: ObjectSummary = new_object(service)
    pending: PhotoInfo = service.add_photo(obj.id, picture(2))
    located: PhotoInfo = located_photo(service, obj, 3, near(1, 1))
    detail: ObjectDetail = service.get_object(obj.id)
    assert [p.id for p in detail.photos] == [pending.id, located.id]
    assert (
        detail.summary.photo_counts["pending"] == 1
        and detail.summary.photo_counts["located"] == 1
    )
    assert detail.photos[1].box == (5, 5, 40, 40) and detail.photos[0].box is None
    assert obj.id in [o.id for o in service.list_objects()]


# ---- Photos ----


def test_a_photo_is_stored_pending_and_wakes_the_worker(
    service: EnrollmentService, woken: Woken
) -> None:
    obj: ObjectSummary = new_object(service)
    info: PhotoInfo = service.add_photo(obj.id, picture(10))
    row: ReferencePhoto = db.gallery.get_photo(info.id)  # type: ignore[assignment]
    assert info.status == "pending" and woken.count == 1
    assert row.file_path in storage.list_stored_paths()
    assert (
        storage.read_image(row.file_path)[:2] == b"\xff\xd8"
    )  # A JPEG without metadata


def test_the_same_picture_twice_for_one_object_leaves_one_file(
    service: EnrollmentService,
) -> None:
    obj: ObjectSummary = new_object(service)
    service.add_photo(obj.id, picture(11))
    files_before: set[str] = set(storage.list_stored_paths())
    with pytest.raises(db.DuplicateImageError):
        service.add_photo(obj.id, picture(11))
    assert set(storage.list_stored_paths()) == files_before


def test_a_file_that_is_not_an_image_leaves_nothing(
    service: EnrollmentService, woken: Woken
) -> None:
    obj: ObjectSummary = new_object(service)
    files_before: set[str] = set(storage.list_stored_paths())
    with pytest.raises(storage.UnsupportedImageError):
        service.add_photo(obj.id, b"not an image")
    assert set(storage.list_stored_paths()) == files_before and woken.count == 0
    assert service.get_object(obj.id).photos == []


def test_a_failed_row_deletes_the_file_again(
    service: EnrollmentService, monkeypatch: pytest.MonkeyPatch
) -> None:
    obj: ObjectSummary = new_object(service)
    files_before: set[str] = set(storage.list_stored_paths())

    def fail(*args: object, **kwargs: object) -> None:
        raise db.DbError("database error")

    monkeypatch.setattr(db.gallery, "add_reference_photo", fail)
    with pytest.raises(db.DbError):
        service.add_photo(obj.id, picture(12))
    assert set(storage.list_stored_paths()) == files_before


def test_the_number_of_photos_for_one_object_is_limited() -> None:
    base: BackendConfig = load_config()
    limited: BackendConfig = dataclasses.replace(
        base, enrollment=dataclasses.replace(base.enrollment, max_photos_per_object=2)
    )
    service: EnrollmentService = EnrollmentService(limited, make_engine())
    obj: ObjectSummary = new_object(service)
    service.add_photo(obj.id, picture(13))
    service.add_photo(obj.id, picture(14))
    with pytest.raises(LimitReachedError):
        service.add_photo(obj.id, picture(15))


def test_a_failed_photo_can_be_tried_again(
    service: EnrollmentService, woken: Woken
) -> None:
    obj: ObjectSummary = new_object(service)
    info: PhotoInfo = service.add_photo(obj.id, picture(16))
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == info.id
    ).execute()
    db.gallery.mark_photo_failed(info.id, "object not found")
    wakes: int = woken.count
    assert (
        service.retry_photo(obj.id, info.id).status == "pending"
        and woken.count == wakes + 1
    )
    with pytest.raises(db.InvalidStateError):
        service.retry_photo(obj.id, info.id)  # It is not failed any more


# ---- Crops ----


def test_the_crop_of_a_located_photo_is_a_jpeg(service: EnrollmentService) -> None:
    obj: ObjectSummary = new_object(service)
    photo: PhotoInfo = located_photo(service, obj, 20, near(1, 1))
    crop: bytes = service.get_crop(obj.id, photo.id)
    image: PilImage.Image = PilImage.open(BytesIO(crop))
    assert image.format == "JPEG" and image.size == (35, 35)  # The box (5, 5, 40, 40)
    assert PilImage.open(
        BytesIO(service.get_crop(obj.id, photo.id, max_side=10))
    ).size == (10, 10)


def test_there_is_no_crop_before_the_object_is_located_or_for_another_object(
    service: EnrollmentService,
) -> None:
    obj, other = new_object(service), new_object(service)
    pending: PhotoInfo = service.add_photo(obj.id, picture(21))
    with pytest.raises(NotFoundError):
        service.get_crop(obj.id, pending.id)
    located: PhotoInfo = located_photo(service, obj, 22, near(1, 2))
    with pytest.raises(NotFoundError):
        service.get_crop(other.id, located.id)  # The photo belongs to another object


# ---- Confirmation and calibration ----


def test_an_object_is_ready_with_three_confirmed_photos(
    service: EnrollmentService,
) -> None:
    obj: ObjectSummary = new_object(service)
    vectors: list[np.ndarray] = [near(1, k) for k in range(3)]
    photos: list[PhotoInfo] = [
        located_photo(service, obj, 30 + k, vectors[k]) for k in range(3)
    ]
    for count, photo in enumerate(photos, start=1):
        assert service.confirm_photo(obj.id, photo.id).status == "confirmed"
        summary: ObjectSummary = service.get_object(obj.id).summary
        if count < 3:
            assert (
                summary.status == "draft"
                and summary.note == f"Needs {3 - count} more confirmed photo(s)."
            )
        else:
            expected = calibrate(
                np.stack(vectors),
                [p.id for p in photos],
                np.empty((0, DIMENSION), np.float32),
                load_ai_config().enrollment,
            )
            assert summary.status == "ready" and summary.note is None
            assert summary.tau == pytest.approx(expected.tau, abs=1e-6)
            assert summary.tau_strong == pytest.approx(expected.tau_strong, abs=1e-6)


def test_confirming_needs_a_located_photo(service: EnrollmentService) -> None:
    obj: ObjectSummary = new_object(service)
    pending: PhotoInfo = service.add_photo(obj.id, picture(40))
    with pytest.raises(db.InvalidStateError):
        service.confirm_photo(obj.id, pending.id)


def test_removing_a_photo_deletes_its_file_and_calibrates_again(
    service: EnrollmentService,
) -> None:
    obj: ObjectSummary = ready_object(service, 1, 50)
    assert obj.status == "ready"
    photo: PhotoInfo = service.get_object(obj.id).photos[0]
    path: str = db.gallery.get_photo(photo.id).file_path  # type: ignore[union-attr]
    service.remove_photo(obj.id, photo.id)
    assert path not in storage.list_stored_paths()
    after: ObjectSummary = service.get_object(obj.id).summary
    assert after.status == "draft" and after.note == "Needs 1 more confirmed photo(s)."
    assert len(service.get_object(obj.id).photos) == 2


def test_a_look_alike_raises_the_threshold_of_the_first_object_and_deleting_it_brings_it_back(
    service: EnrollmentService,
) -> None:
    first: ObjectSummary = ready_object(service, 1, 60)
    alone: float = first.tau  # type: ignore[assignment]
    twin: ObjectSummary = new_object(service, tracked=False)
    for k in range(3):  # The look-alike has views very near those of the first object
        photo: PhotoInfo = located_photo(
            service, twin, 70 + k, near(1, 10 + k, scale=0.05)
        )
        service.confirm_photo(twin.id, photo.id)
    with_twin: ObjectSummary = service.get_object(first.id).summary
    assert (
        with_twin.status == "ready" and with_twin.tau > alone
    )  # The negatives raise tau
    assert with_twin.note is not None  # A warning about the look-alike
    service.delete_object(twin.id)
    back: ObjectSummary = service.get_object(first.id).summary
    assert back.tau == pytest.approx(alone, abs=1e-6) and back.note is None


def test_views_of_another_model_are_not_used(service: EnrollmentService) -> None:
    obj: ObjectSummary = new_object(service)
    for k in range(3):
        photo: PhotoInfo = service.add_photo(obj.id, picture(80 + k))
        ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
            ReferencePhoto.id == photo.id
        ).execute()
        db.gallery.store_photo_views(
            photo.id, (5, 5, 40, 40), 0.5, [("original", near(1, k))], "an-older-model"
        )
        service.confirm_photo(obj.id, photo.id)
    summary: ObjectSummary = service.get_object(obj.id).summary
    assert (
        summary.status == "draft" and summary.note == "Needs 3 more confirmed photo(s)."
    )


# ---- Delete ----


def test_deleting_an_object_deletes_its_photos_and_files(
    service: EnrollmentService,
) -> None:
    obj: ObjectSummary = new_object(service)
    info: PhotoInfo = service.add_photo(obj.id, picture(90))
    path: str = db.gallery.get_photo(info.id).file_path  # type: ignore[union-attr]
    service.delete_object(obj.id)
    assert (
        db.gallery.get_object(obj.id) is None
        and path not in storage.list_stored_paths()
    )
    with pytest.raises(NotFoundError):
        service.delete_object(obj.id)


def test_the_parameters_are_checked(service: EnrollmentService) -> None:
    with pytest.raises(AssertionError):
        EnrollmentService("config", make_engine())  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        service.create_object(unique_name(), "a bottle", "a flask")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        service.add_photo(1, "text")  # type: ignore[arg-type]
