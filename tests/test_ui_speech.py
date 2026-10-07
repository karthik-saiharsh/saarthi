"""The voice of the page. The browser plays the sound, so these tests check the script and the text."""

from ui import speech


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_javascript(self, code: str) -> None:
        self.calls.append(code)


def test_the_script_uses_local_english_voices_only() -> None:
    assert "v.localService" in speech.SPEECH_SCRIPT
    assert "startsWith('en')" in speech.SPEECH_SCRIPT
    # A device without a local voice must stay silent.
    assert "return false" in speech.SPEECH_SCRIPT


def test_an_empty_text_is_not_spoken() -> None:
    browser = FakeBrowser()
    speech.speak("   ", browser)
    assert browser.calls == []


def test_the_text_is_quoted_and_cut() -> None:
    browser = FakeBrowser()
    speech.speak('He said "hi"\n' + "a" * 2000, browser)
    call: str = browser.calls[0]
    assert call.startswith("window.saarthiSpeak(") and '\\"hi\\"' in call
    assert len(call) < speech.MAX_SPOKEN_CHARS + 60


def test_the_text_goes_to_the_given_client_and_not_to_the_current_element() -> None:
    # The old code called `ui.run_javascript`. It failed when the element of the click was deleted.
    browser = FakeBrowser()
    speech.speak("Your bottle was last seen 2 minutes ago.", browser)
    assert len(browser.calls) == 1 and "2 minutes ago" in browser.calls[0]
