"""Tests of the real voice of the wearable (Piper). They need the voice files (a download of about 63 MB).

Run them with: uv run pytest -m slow tests/test_ai_tts_model.py
"""

import numpy as np
import pytest

from ai.settings import load_config
from ai.tts import SpeechSynthesizer

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def synthesizer() -> SpeechSynthesizer:
    return SpeechSynthesizer(load_config())


def test_a_sentence_becomes_speech(synthesizer: SpeechSynthesizer) -> None:
    samples: np.ndarray = synthesizer.synthesize(
        "Your bottle was last seen 12 minutes ago. It was on a table."
    )
    assert synthesizer.sample_rate == 22050 and samples.dtype == np.int16
    seconds: float = samples.size / synthesizer.sample_rate
    assert 2.0 < seconds < 15.0 and np.abs(samples).max() > 1000


def test_an_empty_text_gives_no_samples(synthesizer: SpeechSynthesizer) -> None:
    assert synthesizer.synthesize("   ").size == 0


def test_a_long_text_is_cut(synthesizer: SpeechSynthesizer) -> None:
    short = synthesizer.synthesize("one two three. " * 40)
    longer = synthesizer.synthesize("one two three. " * 400)
    assert longer.size < short.size * 1.5
