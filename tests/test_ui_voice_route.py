"""The route of the Talk button: size limit, fixed errors, no caching."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui.ports import VoiceError
from ui.voice_route import VOICE_ROUTE, register_voice_route


class FakeVoice:
    max_bytes = 100
    max_seconds = 5

    def __init__(self) -> None:
        self.error: VoiceError | None = None
        self.received: list[int] = []

    async def transcribe(self, wav: bytes) -> str:
        if self.error is not None:
            raise self.error
        self.received.append(len(wav))
        return "where is my bottle"


def make_client() -> tuple[TestClient, FakeVoice]:
    app = FastAPI()
    voice = FakeVoice()
    register_voice_route(app, voice)  # type: ignore[arg-type]
    return TestClient(app), voice


def test_a_recording_gives_the_text() -> None:
    client, voice = make_client()
    answer = client.post(VOICE_ROUTE, content=b"x" * 50)
    assert answer.status_code == 200 and answer.json() == {"text": "where is my bottle"}
    assert answer.headers["cache-control"] == "no-store"
    assert voice.received == [50]


def test_a_big_body_is_refused() -> None:
    client, voice = make_client()
    answer = client.post(VOICE_ROUTE, content=b"x" * 101)
    assert answer.status_code == 413 and voice.received == []


def test_an_empty_body_is_refused() -> None:
    client, _ = make_client()
    assert client.post(VOICE_ROUTE, content=b"").status_code == 400


def test_a_voice_error_gives_its_fixed_text() -> None:
    client, voice = make_client()
    voice.error = VoiceError("The recording is too short.")
    answer = client.post(VOICE_ROUTE, content=b"x")
    assert answer.status_code == 422
    assert answer.json() == {"error": "The recording is too short."}


def test_the_talk_button_handler_is_a_javascript_function() -> None:
    # NiceGUI gives the text to the browser as a handler. A bare expression such as `window.x()` runs when the page
    # is built, returns nothing, and breaks the whole card ("Cannot read properties of undefined (reading '_withMods')").
    from ui.pages.find_page import TALK_SCRIPT

    assert TALK_SCRIPT.startswith("() =>") and TALK_SCRIPT.endswith(")")
