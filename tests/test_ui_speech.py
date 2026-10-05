"""The voice of the page. The browser plays the sound, so these tests check the script and the text."""

from unittest.mock import patch

from ui import speech


def test_the_script_uses_local_english_voices_only() -> None:
    assert "v.localService" in speech.SPEECH_SCRIPT
    assert "startsWith('en')" in speech.SPEECH_SCRIPT
    # A device without a local voice must stay silent.
    assert "return false" in speech.SPEECH_SCRIPT


def test_an_empty_text_is_not_spoken() -> None:
    with patch.object(speech.ui, "run_javascript") as run:
        speech.speak("   ")
    run.assert_not_called()


def test_the_text_is_quoted_and_cut() -> None:
    with patch.object(speech.ui, "run_javascript") as run:
        speech.speak('He said "hi"\n' + "a" * 2000)
    call: str = run.call_args.args[0]
    assert call.startswith("window.saarthiSpeak(") and '\\"hi\\"' in call
    assert len(call) < speech.MAX_SPOKEN_CHARS + 60
