import asyncio
from collections.abc import Awaitable, Callable

import pytest

from ui.enroll_session import (
    CROP_MAX_SIDE,
    MESSAGE_CATEGORY_REQUIRED,
    MESSAGE_NAME_REQUIRED,
    MESSAGE_OBJECT_CREATED,
    EnrollSession,
    parse_synonyms,
)
from ui.ports import (
    EnrollmentError,
    ObjectDetail,
    ObjectInfo,
    PhotoInfo,
    PhotoState,
)
from ui.settings import MAX_FILE_SIZE_BYTES
from ui.upload_session import MESSAGE_EMPTY, MESSAGE_TOO_LARGE, MESSAGE_UNSUPPORTED_TYPE

JPEG: str = "image/jpeg"


class FakeEnrollment:
    """Keeps objects in memory. Records calls. Photo data selects the state."""

    def __init__(self) -> None:
        self.objects: dict[int, ObjectInfo] = {}
        self.photos: dict[int, list[PhotoInfo]] = {}
        self.calls: list[str] = []
        self.crop_calls: list[int] = []
        self.active: int = 0
        self.max_active: int = 0
        self._next: int = 1

    def _record(self, call: str) -> None:
        self.calls.append(call)

    async def _busy(self) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1

    def _find(self, object_id: int) -> ObjectInfo:
        if object_id not in self.objects:
            raise EnrollmentError("gone")
        return self.objects[object_id]

    def set_state(self, object_id: int, photo_id: int, state: PhotoState) -> None:
        photos: list[PhotoInfo] = self.photos[object_id]
        self.photos[object_id] = [
            PhotoInfo(p.id, state, p.detail) if p.id == photo_id else p for p in photos
        ]

    async def create_object(
        self,
        name: str,
        category_phrase: str,
        synonyms: tuple[str, ...],
        tracked: bool,
    ) -> ObjectInfo:
        self._record(f"create {name}|{category_phrase}|{synonyms}|{tracked}")
        await self._busy()
        if any(o.name == name for o in self.objects.values()):
            raise EnrollmentError("An object with this name exists.")
        info: ObjectInfo = ObjectInfo(
            self._next, name, category_phrase, synonyms, tracked, False, None, {}
        )
        self.objects[info.id] = info
        self.photos[info.id] = []
        self._next += 1
        return info

    async def list_objects(self) -> list[ObjectInfo]:
        self._record("list")
        await self._busy()
        return list(self.objects.values())

    async def get_object(self, object_id: int) -> ObjectDetail:
        self._record(f"get {object_id}")
        await self._busy()
        return ObjectDetail(self._find(object_id), tuple(self.photos[object_id]))

    async def delete_object(self, object_id: int) -> None:
        self._record(f"delete {object_id}")
        self._find(object_id)
        del self.objects[object_id]

    async def add_photo(self, object_id: int, data: bytes) -> PhotoInfo:
        self._record(f"add {object_id}")
        await self._busy()
        self._find(object_id)
        if data == b"duplicate":
            raise EnrollmentError("This object has this photo already.")
        photo: PhotoInfo = PhotoInfo(self._next, PhotoState.LOCATED, None)
        self._next += 1
        self.photos[object_id].append(photo)
        return photo

    async def confirm_photo(self, object_id: int, photo_id: int) -> None:
        self._record(f"confirm {photo_id}")
        self.set_state(object_id, photo_id, PhotoState.CONFIRMED)

    async def remove_photo(self, object_id: int, photo_id: int) -> None:
        self._record(f"remove {photo_id}")
        self.photos[object_id] = [p for p in self.photos[object_id] if p.id != photo_id]

    async def retry_photo(self, object_id: int, photo_id: int) -> None:
        self._record(f"retry {photo_id}")
        self.set_state(object_id, photo_id, PhotoState.PENDING)

    async def get_crop(self, object_id: int, photo_id: int, max_side: int) -> bytes:
        self.crop_calls.append(max_side)
        if photo_id == 9999:
            raise EnrollmentError("no crop")
        return f"crop{photo_id}".encode()


def run(scenario: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(scenario())


class Listener:
    def __init__(self) -> None:
        self.changes: int = 0
        self.notices: list[tuple[str, bool]] = []

    def on_changed(self) -> None:
        self.changes += 1

    def on_notice(self, text: str, is_error: bool) -> None:
        self.notices.append((text, is_error))


def make_session() -> tuple[EnrollSession, FakeEnrollment, Listener]:
    service: FakeEnrollment = FakeEnrollment()
    listener: Listener = Listener()
    session: EnrollSession = EnrollSession(
        service, listener.on_changed, listener.on_notice
    )
    return session, service, listener


def test_parse_synonyms_splits_and_drops_empty_parts() -> None:
    assert parse_synonyms("a flask, a water bottle ,, ") == (
        "a flask",
        "a water bottle",
    )
    assert parse_synonyms("") == ()
    assert parse_synonyms(" , ") == ()


def test_create_object_opens_it_and_tells_the_user() -> None:
    async def scenario() -> None:
        session, service, listener = make_session()
        assert await session.create_object("Blue bottle", "a bottle", "a flask", False)
        assert service.calls[0] == "create Blue bottle|a bottle|('a flask',)|False"
        assert session.selected_id == 1
        assert session.detail is not None and session.detail.info.name == "Blue bottle"
        assert listener.notices == [(MESSAGE_OBJECT_CREATED, False)]
        assert [o.name for o in session.objects] == ["Blue bottle"]

    run(scenario)


def test_create_object_needs_a_name_and_a_kind() -> None:
    async def scenario() -> None:
        session, service, listener = make_session()
        assert not await session.create_object("  ", "a bottle", "", True)
        assert not await session.create_object("Bottle", " ", "", True)
        assert service.calls == []
        assert listener.notices == [
            (MESSAGE_NAME_REQUIRED, True),
            (MESSAGE_CATEGORY_REQUIRED, True),
        ]

    run(scenario)


def test_create_object_error_is_shown_and_nothing_opens() -> None:
    async def scenario() -> None:
        session, _, listener = make_session()
        assert await session.create_object("Bottle", "a bottle", "", True)
        session.close()
        assert not await session.create_object("Bottle", "a bottle", "", True)
        assert listener.notices[-1] == ("An object with this name exists.", True)
        assert session.selected_id is None and session.detail is None

    run(scenario)


@pytest.mark.parametrize(
    ("content_type", "data", "message"),
    [
        (JPEG, b"", MESSAGE_EMPTY),
        ("application/pdf", b"x", MESSAGE_UNSUPPORTED_TYPE),
        (JPEG, b"x" * (MAX_FILE_SIZE_BYTES + 1), MESSAGE_TOO_LARGE),
    ],
)
def test_bad_photo_is_not_sent(content_type: str, data: bytes, message: str) -> None:
    async def scenario() -> None:
        session, service, listener = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        listener.notices.clear()
        service.calls.clear()
        await session.add_photo("a.jpg", content_type, data)
        assert not any(call.startswith("add") for call in service.calls)
        assert listener.notices == [(f"a.jpg: {message}", True)]

    run(scenario)


def test_photo_without_open_object_is_ignored() -> None:
    async def scenario() -> None:
        session, service, listener = make_session()
        await session.add_photo("a.jpg", JPEG, b"data")
        assert service.calls == [] and listener.notices == []

    run(scenario)


def test_photo_error_names_the_file() -> None:
    async def scenario() -> None:
        session, _, listener = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        listener.notices.clear()
        await session.add_photo("same.jpg", JPEG, b"duplicate")
        assert listener.notices == [
            ("same.jpg: This object has this photo already.", True)
        ]

    run(scenario)


def test_photos_are_sent_one_at_a_time_and_crops_are_loaded_once() -> None:
    async def scenario() -> None:
        session, service, _ = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        await asyncio.gather(
            *(session.add_photo(f"{n}.jpg", JPEG, b"data") for n in range(4))
        )
        assert service.max_active == 1
        assert session.detail is not None and len(session.detail.photos) == 4
        photo_ids: list[int] = [p.id for p in session.detail.photos]
        assert all(session.crop(i) == f"crop{i}".encode() for i in photo_ids)
        assert len(service.crop_calls) == 4  # Not loaded again after each reload
        assert set(service.crop_calls) == {CROP_MAX_SIDE}

    run(scenario)


def test_no_crop_for_a_waiting_photo_and_a_removed_photo_loses_its_crop() -> None:
    async def scenario() -> None:
        session, service, _ = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        await session.add_photo("a.jpg", JPEG, b"data")
        photo_id: int = session.detail.photos[0].id  # type: ignore[union-attr]
        service.set_state(1, photo_id, PhotoState.PENDING)
        await session.refresh()
        assert session.crop(photo_id) is None
        service.set_state(1, photo_id, PhotoState.LOCATED)
        await session.refresh()
        assert session.crop(photo_id) is not None
        await session.remove(photo_id)
        assert session.crop(photo_id) is None
        assert session.detail.photos == ()  # type: ignore[union-attr]

    run(scenario)


def test_a_failed_crop_does_not_stop_the_reload() -> None:
    async def scenario() -> None:
        session, service, _ = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        service.photos[1].append(PhotoInfo(9999, PhotoState.LOCATED, None))
        await session.refresh()
        assert session.crop(9999) is None
        assert session.detail is not None and len(session.detail.photos) == 1

    run(scenario)


def test_confirm_and_retry_call_the_service_for_the_open_object() -> None:
    async def scenario() -> None:
        session, service, _ = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        await session.add_photo("a.jpg", JPEG, b"data")
        photo_id: int = session.detail.photos[0].id  # type: ignore[union-attr]
        await session.confirm(photo_id)
        assert session.detail.photos[0].state == PhotoState.CONFIRMED  # type: ignore[union-attr]
        service.set_state(1, photo_id, PhotoState.FAILED)
        await session.retry(photo_id)
        assert session.detail.photos[0].state == PhotoState.PENDING  # type: ignore[union-attr]
        assert (
            f"confirm {photo_id}" in service.calls
            and f"retry {photo_id}" in service.calls
        )

    run(scenario)


def test_polling_runs_only_while_the_server_works() -> None:
    async def scenario() -> None:
        session, service, _ = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        await session.add_photo("a.jpg", JPEG, b"data")
        photo_id: int = session.detail.photos[0].id  # type: ignore[union-attr]
        assert not session.needs_polling
        before: int = service.calls.count("list")
        await session.poll()
        assert service.calls.count("list") == before  # Idle: no request

        service.set_state(1, photo_id, PhotoState.PROCESSING)
        await session.refresh()
        assert session.needs_polling
        before = service.calls.count("list")
        await session.poll()
        assert service.calls.count("list") == before + 1

        service.set_state(1, photo_id, PhotoState.LOCATED)
        await session.poll()
        assert not session.needs_polling

    run(scenario)


def test_the_view_changes_only_when_the_state_changed() -> None:
    async def scenario() -> None:
        session, service, listener = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        await session.add_photo("a.jpg", JPEG, b"data")
        photo_id: int = session.detail.photos[0].id  # type: ignore[union-attr]
        service.set_state(1, photo_id, PhotoState.PROCESSING)
        await session.refresh()
        changes: int = listener.changes
        await session.refresh()
        await session.refresh()
        assert listener.changes == changes  # Same state: the page is not drawn again
        service.set_state(1, photo_id, PhotoState.LOCATED)
        await session.refresh()
        assert listener.changes == changes + 1

    run(scenario)


def test_delete_closes_the_object_and_a_failed_delete_keeps_it() -> None:
    async def scenario() -> None:
        session, service, listener = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        original = service.delete_object

        async def refuse(object_id: int) -> None:
            raise EnrollmentError("refused")

        service.delete_object = refuse  # type: ignore[method-assign]
        await session.delete_selected()
        assert session.selected_id == 1
        assert listener.notices[-1] == ("refused", True)

        service.delete_object = original  # type: ignore[method-assign]
        await session.delete_selected()
        assert session.selected_id is None and session.detail is None
        assert session.objects == []

    run(scenario)


def test_an_object_that_disappears_is_closed() -> None:
    async def scenario() -> None:
        session, service, _ = make_session()
        await session.create_object("Bottle", "a bottle", "", True)
        service.objects.pop(1)
        await session.refresh()
        assert session.selected_id is None and session.detail is None

    run(scenario)


def test_select_shows_the_other_object() -> None:
    async def scenario() -> None:
        session, _, _ = make_session()
        await session.create_object("One", "a cup", "", True)
        await session.create_object("Two", "a key", "", True)
        await session.select(1)
        assert session.detail is not None and session.detail.info.name == "One"
        session.close()
        assert session.selected_id is None and session.detail is None

    run(scenario)


def test_listener_errors_do_not_stop_the_session() -> None:
    async def scenario() -> None:
        def broken_changed() -> None:
            raise RuntimeError("view error")

        def broken_notice(text: str, is_error: bool) -> None:
            raise RuntimeError("view error")

        session: EnrollSession = EnrollSession(
            FakeEnrollment(), broken_changed, broken_notice
        )
        assert await session.create_object("Bottle", "a bottle", "", True)
        assert session.selected_id == 1

    run(scenario)
