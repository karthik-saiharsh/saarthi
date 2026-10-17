"""The real SigLIP vectors for the open search, on real photos. They need the model and the photos.

Run them with: uv run pytest -m slow tests/test_ai_open_vectors_model.py
The boxes are from the broad detector on the photos 14 and 10 of the folder `Dataset`.
"""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ai.api import AiEngine
from ai.settings import load_config
from ai.types import Box
from tests.ai_support import (
    FakeDetector,
    FakeEmbedder,
    FakeLanguageModel,
    FakeOpenDetector,
    FakeQueryEncoder,
    FakeSpeechRecognizer,
)

DATASET = Path(
    os.environ.get(
        "SAARTHI_TEST_DATASET", Path(__file__).resolve().parents[2] / "Dataset"
    )
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not DATASET.is_dir(), reason="no photos (set SAARTHI_TEST_DATASET)"
    ),
]

LAPTOP_BOX = Box(462, 132, 916, 591)  # Photo 14
BOTTLE_BOX = Box(424, 80, 709, 993)  # Photo 10, a bottle in a hand


@pytest.fixture(scope="module")
def engine() -> AiEngine:
    base = load_config()
    config = dataclasses.replace(
        base,
        verification=dataclasses.replace(base.verification, enabled=False),
    )
    return AiEngine(
        config,
        detector=FakeDetector(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        query_encoder=FakeQueryEncoder(),  # type: ignore[arg-type]
        speech_recognizer=FakeSpeechRecognizer(),  # type: ignore[arg-type]
        open_detector=FakeOpenDetector(),  # type: ignore[arg-type]
        language_model=FakeLanguageModel(),  # type: ignore[arg-type]
    )


def score(engine: AiEngine, photo: str, box: Box, phrases: list[str]) -> np.ndarray:
    image = Image.open(DATASET / photo).convert("RGB")
    crops = engine.embed_evidence_crops(image, [box])
    return engine.open_pair_scores(crops, engine.embed_open_phrases(phrases))[0]


def test_a_laptop_crop_fits_laptop_better_than_other_phrases(engine: AiEngine) -> None:
    scores = score(
        engine, "14.jpeg", LAPTOP_BOX, ["laptop", "bottle", "person", "wall"]
    )
    # The size of the score depends on the crop (0.08 here, 0.89 for a laptop box of another photo).
    # The order is what the search uses, so the test checks the order and a margin.
    assert scores[0] > scores[1:].max() + 0.05


def test_a_bottle_crop_fits_bottle_better_than_laptop(engine: AiEngine) -> None:
    scores = score(engine, "10.jpeg", BOTTLE_BOX, ["bottle", "laptop"])
    assert scores[0] > scores[1]


def test_the_vectors_have_unit_length_and_the_model_size(engine: AiEngine) -> None:
    image = Image.open(DATASET / "14.jpeg").convert("RGB")
    crops = engine.embed_evidence_crops(image, [LAPTOP_BOX, Box(1, 1, 50, 50)])
    phrases = engine.embed_open_phrases(["laptop"])
    assert crops.shape == (2, engine.open_vector_dimension)
    assert np.allclose(np.linalg.norm(crops, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(phrases, axis=1), 1.0, atol=1e-3)
