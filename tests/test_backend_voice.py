"""The voice input: a WAV recording becomes text. The recognizer is a fake that records what it gets."""

import asyncio
import io
import logging
import wave

import numpy as np
import pytest

from backend.errors import InvalidAudioError, VoiceUnavailableError
from backend.settings import load_config
from backend.ui_adapter import UiVoiceService
from backend.voice import VoiceService
from tests.ai_support import FakeSpeechRecognizer, make_engine
from ui.ports import VoiceError

RATE = 16000


def wav_bytes(
    seconds: float = 1.0,
    amplitude: float = 0.3,
    rate: int = RATE,
    channels: int = 1,
    width: int = 2,
) -> bytes:
    count = int(seconds * rate)
    wave_data = (np.sin(np.arange(count) / 20.0) * amplitude * 32767).astype("<i2")
    if channels == 2:
        wave_data = np.repeat(wave_data, 2)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(
            wave_data.tobytes() if width == 2 else bytes(count * channels)
        )
    return buffer.getvalue()


def make_service(
    text: str = "where is my bottle",
) -> tuple[VoiceService, FakeSpeechRecognizer]:
    fake = FakeSpeechRecognizer(text)
    return VoiceService(load_config(), make_engine(speech_recognizer=fake)), fake


def test_a_good_recording_gives_the_text_and_float_samples() -> None:
    service, fake = make_service()
    assert service.transcribe(wav_bytes(1.0)) == "where is my bottle"
    assert fake.calls == [RATE]


@pytest.mark.parametrize(
    "wav",
    [
        b"not a wav file",
        b"",
        wav_bytes(rate=44100),
        wav_bytes(channels=2),
        wav_bytes(width=1),
    ],
)
def test_a_wrong_format_is_refused(wav: bytes) -> None:
    service, fake = make_service()
    with pytest.raises(InvalidAudioError):
        service.transcribe(wav)
    assert fake.calls == []


def test_a_short_recording_is_refused() -> None:
    service, fake = make_service()
    with pytest.raises(InvalidAudioError, match="short"):
        service.transcribe(wav_bytes(0.2))
    assert fake.calls == []


def test_a_long_recording_is_refused() -> None:
    service, fake = make_service()
    seconds = service.max_seconds + 2
    with pytest.raises(InvalidAudioError, match="long"):
        service.transcribe(wav_bytes(seconds))
    assert fake.calls == []


def test_a_silent_recording_gets_no_text_and_no_model_call() -> None:
    service, fake = make_service()
    assert service.transcribe(wav_bytes(1.0, amplitude=0.0)) == ""
    assert fake.calls == []


def test_a_switched_off_recognizer_is_reported() -> None:
    engine = make_engine()
    engine._speech = None  # The speech model is switched off
    off = VoiceService(load_config(), engine)
    assert not off.enabled
    with pytest.raises(VoiceUnavailableError):
        off.transcribe(wav_bytes())


def test_the_ui_service_maps_errors_to_fixed_texts() -> None:
    service, fake = make_service()
    ui_service = UiVoiceService(service)
    assert asyncio.run(ui_service.transcribe(wav_bytes())) == "where is my bottle"
    with pytest.raises(VoiceError, match="not a usable WAV"):
        asyncio.run(ui_service.transcribe(b"junk"))
    fake.error = RuntimeError("secret detail")
    with pytest.raises(VoiceError) as caught:
        asyncio.run(ui_service.transcribe(wav_bytes()))
    assert "secret" not in str(caught.value)


def test_the_log_never_shows_the_words(caplog: pytest.LogCaptureFixture) -> None:
    service, _ = make_service("my secret bottle")
    with caplog.at_level(logging.DEBUG):
        service.transcribe(wav_bytes())
    assert "secret" not in caplog.text


def test_speak_gives_a_wav_file_of_the_voice() -> None:
    from tests.ai_support import FakeSpeechSynthesizer

    fake = FakeSpeechSynthesizer()
    service = VoiceService(load_config(), make_engine(speech_synthesizer=fake))
    data = service.speak("Your bottle was last seen just now.")
    assert data is not None and fake.texts == ["Your bottle was last seen just now."]
    with wave.open(io.BytesIO(data), "rb") as reader:
        assert reader.getframerate() == 22050 and reader.getnchannels() == 1
        assert reader.getsampwidth() == 2 and reader.getnframes() == 2205


def test_speak_gives_none_when_the_voice_is_off_fails_or_has_no_text() -> None:
    from ai.errors import InferenceError
    from tests.ai_support import FakeSpeechSynthesizer

    fake = FakeSpeechSynthesizer()
    engine = make_engine(speech_synthesizer=fake)
    service = VoiceService(load_config(), engine)
    assert service.speak("  ") is None and not fake.texts
    fake.error = InferenceError("broke")
    assert service.speak("Hello.") is None
    engine._tts = None  # The switch "voice off" of the engine
    assert not service.speaking_enabled and service.speak("Hello.") is None
