"""Tests of the stub for the find page. It is for work on the UI only."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from ui import stub_service
from ui.ports import FindError, FindResult, FindStatus
from ui.stub_service import StubFindService


@pytest.fixture(autouse=True)
def no_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(seconds: float) -> None:
        return None

    monkeypatch.setattr(stub_service.asyncio, "sleep", instant)


def run(scenario: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(scenario())


def test_the_stub_lists_its_objects() -> None:
    async def scenario() -> None:
        choices = await StubFindService().list_choices()
        assert [c.name for c in choices] == [
            "Blue bottle",
            "House keys",
            "Tea cup",
            "Grey mug",
        ]

    run(scenario)


@pytest.mark.parametrize(
    ("text", "status", "name"),
    [
        ("where is my bottle", FindStatus.FOUND, "Blue bottle"),
        ("the thing I drink from", FindStatus.FOUND, "Blue bottle"),
        ("my KEYS", FindStatus.NOT_SEEN, "House keys"),
        ("my cup", FindStatus.CHOOSE, None),
        ("xylophone", FindStatus.NO_MATCH, None),
    ],
)
def test_the_words_select_the_result(
    text: str, status: FindStatus, name: str | None
) -> None:
    async def scenario() -> None:
        result: FindResult = await StubFindService().ask(text)
        assert result.status == status and result.object_name == name

    run(scenario)


def test_each_fixed_object_shows_one_case_of_the_page() -> None:
    async def scenario() -> None:
        service: StubFindService = StubFindService()
        bottle = (await service.find(1)).sighting
        cup = (await service.find(3)).sighting
        mug = (await service.find(4)).sighting
        keys: FindResult = await service.find(2)
        assert bottle is not None and bottle.strong and not bottle.stale
        assert bottle.picture is not None and bottle.picture.startswith(b"\xff\xd8")
        assert cup is not None and cup.in_hand and not cup.strong
        assert mug is not None and mug.stale
        assert keys.status == FindStatus.NOT_SEEN and keys.complete is False

    run(scenario)


def test_a_choice_has_two_objects_and_an_unknown_object_is_refused() -> None:
    async def scenario() -> None:
        service: StubFindService = StubFindService()
        assert len((await service.ask("a mug")).choices) == 2
        with pytest.raises(FindError):
            await service.find(99)

    run(scenario)
