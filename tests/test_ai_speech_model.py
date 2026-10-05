"""Tests of the real speech recognizer. They need the model.

Run them with: uv run pytest -m slow tests/test_ai_speech_model.py
For a real test of the words, record a short WAV file (16 kHz, 16-bit, mono) of a person who says a sentence.
Set SAARTHI_TEST_SPEECH_WAV to its path, and SAARTHI_TEST_SPEECH_WORDS to words that the text must contain.
"""

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from ai.settings import load_config
from ai.speech import SAMPLE_RATE, SpeechRecognizer

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def recognizer() -> SpeechRecognizer:
    return SpeechRecognizer(load_config())


def test_the_model_reads_audio_and_gives_text(recognizer: SpeechRecognizer) -> None:
    time_axis = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    tone = (0.2 * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)
    assert isinstance(recognizer.transcribe(tone), str)


@pytest.mark.skipif(
    "SAARTHI_TEST_SPEECH_WAV" not in os.environ,
    reason="no recording (set SAARTHI_TEST_SPEECH_WAV)",
)
def test_a_real_recording_gives_the_expected_words(
    recognizer: SpeechRecognizer,
) -> None:
    with wave.open(str(Path(os.environ["SAARTHI_TEST_SPEECH_WAV"])), "rb") as reader:
        assert reader.getframerate() == SAMPLE_RATE and reader.getnchannels() == 1
        raw = reader.readframes(reader.getnframes())
    samples = (np.frombuffer(raw, dtype="<i2") / 32768.0).astype(np.float32)
    text = recognizer.transcribe(samples).lower()
    for word in os.environ.get("SAARTHI_TEST_SPEECH_WORDS", "").lower().split():
        assert word in text
