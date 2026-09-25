"""Tests of the UI adapter for enrollment. They use the actual database and storage, and a fake ai engine.

Every object has a name that starts with `__test__ `. The objects and their
files are removed at the end.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterator

import pytest

import db.api as db
import storage.api as storage
from ai.api import AiError
from backend import ui_adapter
from backend.enrollment import EnrollmentService
from backend.errors import (
    BackendError,
    InvalidInputError,
    LimitReachedError,
    NotFoundError,
)
from backend.settings import load_config
from backend.ui_adapter import UiEnrollmentService
from tests.ai_support import make_engine
from tests.test_backend_enrollment import (
    PREFIX,
    located_photo,
    near,
    picture,
    unique_name,
)
from ui.ports import EnrollmentError, PhotoState


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    yield
    for obj in db.gallery.list_objects():
        if obj.name.startswith(PREFIX):
            storage.delete_images(db.gallery.delete_object(obj.id))
    database.close()


@pytest.fixture()
def backend_service() -> EnrollmentService:
    return EnrollmentService(load_config(), make_engine())


@pytest.fixture()
def adapter(backend_service: EnrollmentService) -> UiEnrollmentService:
    return UiEnrollmentService(backend_service)


def run(scenario: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(scenario())


def test_an_object_goes_through_the_adapter() -> None:
    async def scenario() -> None:
        service: EnrollmentService = EnrollmentService(load_config(), make_engine())
        adapter: UiEnrollmentService = UiEnrollmentService(service)
        name: str = unique_name()
        info = await adapter.create_object(name, "A Bottle", ("a flask",), False)
        assert (info.name, info.category_phrase, info.synonyms) == (
            name,
            "a bottle",
            ("a flask",),
        )
        assert not info.tracked and not info.ready
        assert info.note is not None and info.note.startswith("Needs")
        assert [o.id for o in await adapter.list_objects() if o.id == info.id] == [
            info.id
        ]
        detail = await adapter.get_object(info.id)
        assert detail.info.id == info.id and detail.photos == ()
        await adapter.delete_object(info.id)
        assert not [o for o in await adapter.list_objects() if o.id == info.id]

    run(scenario)


def test_a_photo_is_added_confirmed_and_cropped(
    backend_service: EnrollmentService, adapter: UiEnrollmentService
) -> None:
    async def scenario() -> None:
        info = await adapter.create_object(unique_name(), "a bottle", (), True)
        photo = await adapter.add_photo(info.id, picture(1))
        assert photo.state == PhotoState.PENDING
        with pytest.raises(EnrollmentError):  # Not located yet: there is no crop
            await adapter.get_crop(info.id, photo.id, 200)

    run(scenario)
    # The worker would locate the photo. The test does it with a helper.
    summary = backend_service.list_objects()[-1]
    located = located_photo(backend_service, summary, 2, near(1, 1))

    async def after() -> None:
        crop: bytes = await adapter.get_crop(summary.id, located.id, 200)
        assert crop.startswith(b"\xff\xd8")
        await adapter.confirm_photo(summary.id, located.id)
        detail = await adapter.get_object(summary.id)
        states = {p.id: p.state for p in detail.photos}
        assert states[located.id] == PhotoState.CONFIRMED
        await adapter.remove_photo(summary.id, located.id)
        assert located.id not in {
            p.id for p in (await adapter.get_object(summary.id)).photos
        }

    run(after)


def test_three_accepted_photos_make_the_object_ready(
    backend_service: EnrollmentService, adapter: UiEnrollmentService
) -> None:
    summary = backend_service.create_object(unique_name(), "a bottle")
    for k in range(3):
        photo = located_photo(backend_service, summary, 10 + k, near(5, k))
        backend_service.confirm_photo(summary.id, photo.id)

    async def scenario() -> None:
        detail = await adapter.get_object(summary.id)
        assert detail.info.ready
        assert detail.info.photo_counts[PhotoState.CONFIRMED] == 3
        assert detail.info.photo_counts[PhotoState.LOCATED] == 0

    run(scenario)


def test_errors_become_fixed_texts(adapter: UiEnrollmentService) -> None:
    async def scenario() -> None:
        name: str = unique_name()
        await adapter.create_object(name, "a bottle", (), True)
        with pytest.raises(EnrollmentError) as exists:
            await adapter.create_object(name, "a bottle", (), True)
        assert str(exists.value) == ui_adapter.MESSAGE_NAME_EXISTS
        with pytest.raises(EnrollmentError) as phrase:
            await adapter.create_object(unique_name(), "a bottle!", (), True)
        assert "phrase" in str(phrase.value).lower()
        with pytest.raises(EnrollmentError) as missing:
            await adapter.get_object(99999999)
        assert str(missing.value) == ui_adapter.MESSAGE_NOT_FOUND
        info = await adapter.create_object(unique_name(), "a cup", (), True)
        await adapter.add_photo(info.id, picture(3))
        with pytest.raises(EnrollmentError) as same:
            await adapter.add_photo(info.id, picture(3))
        assert str(same.value) == ui_adapter.MESSAGE_PHOTO_EXISTS
        with pytest.raises(EnrollmentError) as bad:
            await adapter.add_photo(info.id, b"not an image")
        assert str(bad.value) == ui_adapter.MESSAGE_UNSUPPORTED

    run(scenario)


SECRET: str = "/home/secret/path SELECT * FROM x"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (NotFoundError(SECRET), ui_adapter.MESSAGE_NOT_FOUND),
        (db.DuplicateObjectError(SECRET), ui_adapter.MESSAGE_NAME_EXISTS),
        (db.DuplicateImageError(SECRET), ui_adapter.MESSAGE_PHOTO_EXISTS),
        (db.InvalidStateError(SECRET), ui_adapter.MESSAGE_WRONG_STATE),
        (storage.ImageTooLargeError(SECRET), ui_adapter.MESSAGE_TOO_LARGE),
        (storage.UnsupportedImageError(SECRET), ui_adapter.MESSAGE_UNSUPPORTED),
        (storage.StorageFullError(SECRET), ui_adapter.MESSAGE_DISK_FULL),
        (storage.StorageNotFoundError(SECRET), ui_adapter.MESSAGE_FILE_MISSING),
        (storage.StorageIOError(SECRET), ui_adapter.MESSAGE_FAILED),
        (db.DbError(SECRET), ui_adapter.MESSAGE_FAILED),
        (BackendError(SECRET), ui_adapter.MESSAGE_FAILED),
        (AiError(SECRET), ui_adapter.MESSAGE_FAILED),
        (RuntimeError(SECRET), ui_adapter.MESSAGE_FAILED),
    ],
)
def test_internal_details_never_reach_the_user(
    backend_service: EnrollmentService,
    adapter: UiEnrollmentService,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    def raises(*arguments: object) -> None:
        raise error

    monkeypatch.setattr(backend_service, "list_objects", raises)

    async def scenario() -> None:
        with pytest.raises(EnrollmentError) as caught:
            await adapter.list_objects()
        assert str(caught.value) == message
        assert SECRET not in str(caught.value)

    run(scenario)


@pytest.mark.parametrize(
    "error", [InvalidInputError("A fixed text."), LimitReachedError("A fixed limit.")]
)
def test_service_texts_of_input_and_limit_errors_pass_through(
    backend_service: EnrollmentService,
    adapter: UiEnrollmentService,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def raises(*arguments: object) -> None:
        raise error

    monkeypatch.setattr(backend_service, "create_object", raises)

    async def scenario() -> None:
        with pytest.raises(EnrollmentError) as caught:
            await adapter.create_object("n", "a bottle", (), True)
        assert str(caught.value) == str(error)

    run(scenario)


def test_the_backend_gives_the_adapter() -> None:
    from backend.api import Backend

    assert Backend.enrollment_service.fget is not None  # type: ignore[attr-defined]
