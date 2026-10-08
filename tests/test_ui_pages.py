"""Tests that build the pages in a NiceGUI client without a browser.

They check what the page contains: the elements and their handlers. They would have found the broken Talk card
(a click handler that was not a JavaScript function). The sound and the real rendering need a browser.
"""

import itertools
from collections.abc import Iterator

import pytest
from nicegui import Client, ui

from ui.pages.find_page import build_find_page
from ui.pages.memory_page import build_memory_page
from ui.ports import FramesPage
from ui.stub_service import StubFindService

_paths = itertools.count(1)


class FakeVoice:
    max_bytes = 100
    max_seconds = 5

    async def transcribe(self, wav: bytes) -> str:
        return "text"


class FakeBrowse:
    async def list_frames(self, status, offset, limit):
        return FramesPage([], 0, offset, limit, {"done": 0})

    async def get_frame(
        self, image_id
    ):  # pragma: no cover - the page does not open a frame here
        raise AssertionError

    async def get_picture(self, image_id, max_side):  # pragma: no cover
        raise AssertionError


@pytest.fixture()
def client() -> Iterator[Client]:
    yield Client(ui.page(f"/probe-{next(_paths)}"))


def texts_of(client: Client) -> list[str]:
    return [
        t for t in (getattr(e, "text", None) for e in client.elements.values()) if t
    ]


def test_the_find_page_has_the_talk_button_with_a_label_and_a_function_handler(
    client: Client,
) -> None:
    with client:
        build_find_page(StubFindService(), FakeVoice())  # type: ignore[arg-type]
    assert "Say what you look for" in texts_of(client) and "Talk" in texts_of(client)
    buttons = [
        e
        for e in client.elements.values()
        if getattr(e, "tag", "") == "button" and e._props.get("id") == "saarthi-talk"
    ]
    assert len(buttons) == 1
    handlers = [
        listener.js_handler for listener in buttons[0]._event_listeners.values()
    ]
    assert handlers == ["() => window.saarthiTalk.toggle()"]


def test_the_find_page_has_no_talk_card_without_a_voice_service(client: Client) -> None:
    with client:
        build_find_page(StubFindService(), None)  # type: ignore[arg-type]
    assert "Say what you look for" not in texts_of(client) and "Talk" not in texts_of(
        client
    )
    assert "Or write it in your own words" in texts_of(client)


def test_the_memory_page_builds_and_is_in_the_navigation(client: Client) -> None:
    with client:
        build_memory_page(FakeBrowse())  # type: ignore[arg-type]
    texts = texts_of(client)
    assert "Memory" in texts and "Find" in texts
