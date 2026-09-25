import asyncio
from collections.abc import Awaitable, Callable

import pytest

import ui.stub_service as stub_module
from ui.ports import EnrollmentError, PhotoState
from ui.stub_service import StubEnrollmentService


class Clock:
    def __init__(self) -> None:
        self.now: float = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake: Clock = Clock()
    monkeypatch.setattr(stub_module.time, "monotonic", fake)

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(stub_module.asyncio, "sleep", no_sleep)
    return fake


def run(scenario: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(scenario())


def test_a_photo_goes_through_the_states(clock: Clock) -> None:
    async def scenario() -> None:
        service: StubEnrollmentService = StubEnrollmentService()
        info = await service.create_object("Bottle", "a bottle", (), True)
        photo = await service.add_photo(info.id, b"data")
        states: list[PhotoState] = []
        for step in (0.0, 1.2, 3.0):
            clock.now = 1000.0 + step
            states.append((await service.get_object(info.id)).photos[0].state)
        assert states == [PhotoState.PENDING, PhotoState.PROCESSING, PhotoState.LOCATED]
        crop: bytes = await service.get_crop(info.id, photo.id, 200)
        assert crop.startswith(b"\xff\xd8")

    run(scenario)


def test_the_object_is_ready_after_three_accepted_photos(clock: Clock) -> None:
    async def scenario() -> None:
        service: StubEnrollmentService = StubEnrollmentService()
        info = await service.create_object("Bottle", "a bottle", (), True)
        for _ in range(3):
            await service.add_photo(info.id, b"data")
        clock.now += 10
        detail = await service.get_object(info.id)
        assert [p.state for p in detail.photos] == [PhotoState.LOCATED] * 3
        for photo in detail.photos:
            await service.confirm_photo(info.id, photo.id)
        assert (await service.list_objects())[0].ready

    run(scenario)


def test_the_fourth_photo_fails_and_can_be_tried_again(clock: Clock) -> None:
    async def scenario() -> None:
        service: StubEnrollmentService = StubEnrollmentService()
        info = await service.create_object("Bottle", "a bottle", (), True)
        for _ in range(4):
            await service.add_photo(info.id, b"data")
        clock.now += 10
        failed = (await service.get_object(info.id)).photos[3]
        assert failed.state == PhotoState.FAILED and failed.detail == "object not found"
        await service.retry_photo(info.id, failed.id)
        clock.now += 10
        assert (await service.get_object(info.id)).photos[3].state == PhotoState.LOCATED

    run(scenario)


def test_wrong_actions_raise_enrollment_errors(clock: Clock) -> None:
    async def scenario() -> None:
        service: StubEnrollmentService = StubEnrollmentService()
        info = await service.create_object("Bottle", "a bottle", (), True)
        with pytest.raises(EnrollmentError):
            await service.create_object("Bottle", "a bottle", (), True)
        with pytest.raises(EnrollmentError):
            await service.get_object(99)
        photo = await service.add_photo(info.id, b"data")
        with pytest.raises(EnrollmentError):
            await service.confirm_photo(info.id, photo.id)  # Still pending
        with pytest.raises(EnrollmentError):
            await service.get_crop(info.id, photo.id, 100)
        await service.delete_object(info.id)
        assert await service.list_objects() == []

    run(scenario)
