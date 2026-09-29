"""Tests of the real query encoder. They download the weights on the first run.

Run them with: uv run pytest -m slow tests/test_ai_query_model.py
"""

import numpy as np
import pytest

from ai.query import QueryEncoder
from ai.settings import AiConfig, load_config

pytestmark = pytest.mark.slow

DESCRIPTIONS: list[str] = [
    "Blue bottle: a bottle, a water bottle",
    "Reading glasses: a pair of glasses",
    "House keys: a set of keys",
    "Tea cup: a cup, a mug",
    "My phone: a mobile phone",
]


@pytest.fixture(scope="module")
def config() -> AiConfig:
    return load_config()


@pytest.fixture(scope="module")
def encoder(config: AiConfig) -> QueryEncoder:
    return QueryEncoder(config)


def test_the_vectors_have_unit_length_and_the_same_order_as_the_texts(
    encoder: QueryEncoder,
) -> None:
    vectors: np.ndarray = encoder.encode(["a bottle", "a set of keys"])
    assert vectors.shape == (2, encoder.dimension) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert encoder.encode([]).shape == (0, encoder.dimension)
    assert encoder.name.endswith("@" + load_config().models.query_encoder_revision[:8])


def test_the_padding_does_not_change_a_vector(encoder: QueryEncoder) -> None:
    alone: np.ndarray = encoder.encode(["a bottle"])[0]
    batched: np.ndarray = encoder.encode(["a bottle", "a very long text " * 10])[0]
    assert float(alone @ batched) > 0.999


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("I can't find this blue thing I use to drink stuff", 0),
        ("where are my glasses", 1),
        ("I need to unlock the front door", 2),
        ("something I drink tea from", 3),
    ],
)
def test_a_vague_question_ranks_the_right_object_first(
    encoder: QueryEncoder, question: str, expected: int
) -> None:
    objects: np.ndarray = encoder.encode(DESCRIPTIONS)
    scores: np.ndarray = objects @ encoder.encode([question])[0]
    assert int(scores.argmax()) == expected


def test_a_text_longer_than_the_limit_is_cut_and_still_works(
    encoder: QueryEncoder,
) -> None:
    vector: np.ndarray = encoder.encode(["word " * 5000])
    assert vector.shape == (1, encoder.dimension) and np.isfinite(vector).all()
