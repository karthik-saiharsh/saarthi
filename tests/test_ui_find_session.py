"""Tests of the state of the find page. The session has no NiceGUI code, so no browser is needed."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from ui.find_session import (
    MESSAGE_FAILED,
    MESSAGE_LIST_FAILED,
    MESSAGE_QUERY_REQUIRED,
    MESSAGE_QUERY_TOO_LONG,
    FindSession,
)
from ui.ports import FindChoice, FindError, FindResult, FindStatus
from ui.settings import MAX_QUERY_CHARS

BOTTLE: FindChoice = FindChoice(1, "Blue bottle")
FOUND: FindResult = FindResult(FindStatus.FOUND, object_name="Blue bottle")


class FakeFind:
    """Records the calls. A test sets the answer, an error, or a gate that holds a call."""

    def __init__(self) -> None:
        self.choices: list[FindChoice] = [BOTTLE, FindChoice(2, "House keys")]
        self.result: FindResult = FOUND
        self.error: Exception | None = None
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None

    async def _answer(self) -> FindResult:
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def list_choices(self) -> list[FindChoice]:
        if self.error is not None:
            raise self.error
        return list(self.choices)

    async def ask(self, text: str) -> FindResult:
        self.calls.append(f"ask {text}")
        return await self._answer()

    async def find(self, object_id: int) -> FindResult:
        self.calls.append(f"find {object_id}")
        return await self._answer()


class Recorder:
    """Collects what the session tells the view."""

    def __init__(self, session_getter: Callable[[], FindSession]) -> None:
        self._get = session_getter
        self.states: list[tuple[bool, FindResult | None]] = []
        self.notices: list[tuple[str, bool]] = []

    def changed(self) -> None:
        session: FindSession = self._get()
        self.states.append((session.busy, session.result))

    def notice(self, text: str, is_error: bool) -> None:
        self.notices.append((text, is_error))


def make(service: FakeFind | None = None) -> tuple[FindSession, FakeFind, Recorder]:
    fake: FakeFind = service or FakeFind()
    holder: list[FindSession] = []
    recorder: Recorder = Recorder(lambda: holder[0])
    session: FindSession = FindSession(fake, recorder.changed, recorder.notice)
    holder.append(session)
    return session, fake, recorder


def run(scenario: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(scenario())


def test_load_gives_the_objects_and_tells_the_view() -> None:
    async def scenario() -> None:
        session, _, recorder = make()
        await session.load()
        assert [c.name for c in session.choices] == ["Blue bottle", "House keys"]
        assert len(recorder.states) == 1

    run(scenario)


def test_a_failed_load_tells_the_user_and_keeps_the_page_usable() -> None:
    async def scenario() -> None:
        fake: FakeFind = FakeFind()
        fake.error = FindError("internal detail")
        session, _, recorder = make(fake)
        await session.load()
        assert session.choices == []
        assert recorder.notices == [(MESSAGE_LIST_FAILED, True)]

    run(scenario)


def test_ask_cleans_the_spaces_and_shows_the_result() -> None:
    async def scenario() -> None:
        session, fake, _ = make()
        await session.ask("  where   is\nmy bottle ")
        assert fake.calls == ["ask where is my bottle"]
        assert session.result == FOUND and not session.busy

    run(scenario)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_words_are_not_sent(text: str) -> None:
    async def scenario() -> None:
        session, fake, recorder = make()
        await session.ask(text)
        assert fake.calls == [] and session.result is None
        assert recorder.notices == [(MESSAGE_QUERY_REQUIRED, True)]

    run(scenario)


def test_words_that_are_too_long_are_not_sent() -> None:
    async def scenario() -> None:
        session, fake, recorder = make()
        await session.ask("a" * (MAX_QUERY_CHARS + 1))
        assert fake.calls == []
        assert recorder.notices == [(MESSAGE_QUERY_TOO_LONG, True)]
        await session.ask("a" * MAX_QUERY_CHARS)  # The limit itself is allowed
        assert len(fake.calls) == 1

    run(scenario)


def test_pick_finds_the_object_by_its_id() -> None:
    async def scenario() -> None:
        session, fake, _ = make()
        await session.pick(2)
        assert fake.calls == ["find 2"] and session.result == FOUND

    run(scenario)


def test_the_view_sees_busy_first_and_then_the_result() -> None:
    async def scenario() -> None:
        session, _, recorder = make()
        await session.pick(1)
        assert recorder.states == [(True, None), (False, FOUND)]

    run(scenario)


def test_an_old_result_is_removed_when_a_new_search_starts() -> None:
    async def scenario() -> None:
        session, fake, _ = make()
        await session.pick(1)
        fake.gate = asyncio.Event()
        task = asyncio.create_task(session.pick(2))
        await asyncio.sleep(0)
        assert session.busy and session.result is None  # The old answer is gone
        fake.gate.set()
        await task
        assert session.result == FOUND

    run(scenario)


def test_a_request_while_a_search_runs_is_ignored() -> None:
    async def scenario() -> None:
        session, fake, _ = make()
        fake.gate = asyncio.Event()
        first = asyncio.create_task(session.pick(1))
        await asyncio.sleep(0)
        await session.pick(2)  # A second tap, for example a double tap
        await session.ask("keys")
        fake.gate.set()
        await first
        assert fake.calls == ["find 1"]

    run(scenario)


def test_a_refused_search_shows_its_text_and_frees_the_page() -> None:
    async def scenario() -> None:
        fake: FakeFind = FakeFind()
        fake.error = FindError("This object cannot be searched yet.")
        session, _, recorder = make(fake)
        await session.pick(1)
        assert recorder.notices == [("This object cannot be searched yet.", True)]
        assert session.result is None and not session.busy
        fake.error = None
        await session.pick(1)  # The page still works
        assert session.result == FOUND

    run(scenario)


def test_an_unexpected_error_gives_a_plain_text_and_frees_the_page() -> None:
    async def scenario() -> None:
        fake: FakeFind = FakeFind()
        fake.error = RuntimeError("secret internal text")
        session, _, recorder = make(fake)
        await session.ask("my bottle")
        assert recorder.notices == [(MESSAGE_FAILED, True)]
        assert "secret" not in recorder.notices[0][0]
        assert not session.busy and session.result is None

    run(scenario)


def test_clear_removes_the_result_and_tells_the_view_only_when_it_changed() -> None:
    async def scenario() -> None:
        session, _, recorder = make()
        session.clear()
        assert recorder.states == []  # Nothing to clear
        await session.pick(1)
        before: int = len(recorder.states)
        session.clear()
        assert session.result is None and len(recorder.states) == before + 1

    run(scenario)


def test_a_view_error_does_not_stop_the_session() -> None:
    async def scenario() -> None:
        fake: FakeFind = FakeFind()

        def broken() -> None:
            raise RuntimeError("view error")

        def broken_notice(text: str, is_error: bool) -> None:
            raise RuntimeError("view error")

        session: FindSession = FindSession(fake, broken, broken_notice)
        await session.load()
        await session.ask("")  # The notice fails, and nothing breaks
        await session.pick(1)
        assert session.result == FOUND and not session.busy

    run(scenario)
