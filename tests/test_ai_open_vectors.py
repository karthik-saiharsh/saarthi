"""The vectors for the open search: crops and phrases in one space. The model is a fake or a small stub."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from ai.scene import SceneClassifier, phrase_with_article
from ai.types import Box
from tests.ai_support import FakeSceneClassifier, make_engine

IMAGE = Image.new("RGB", (400, 300))


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("laptop", "a laptop"),
        ("umbrella", "an umbrella"),
        ("  coffee   cup ", "a coffee cup"),
        ("a bottle", "a bottle"),
        ("the remote", "the remote"),
        ("An apple", "An apple"),
        ("my keys", "my keys"),
        ("alarm clock", "an alarm clock"),
    ],
)
def test_phrase_with_article(label: str, expected: str) -> None:
    assert phrase_with_article(label) == expected


# ---- The engine ----


def test_the_boxes_are_cut_with_a_little_space_and_stay_in_the_frame() -> None:
    scene = FakeSceneClassifier()
    engine = make_engine(scene_classifier=scene)
    # Padding 0.05: a box of 100 x 200 gets 5 and 10 px on each side.
    vectors = engine.embed_open_crops(
        IMAGE, [Box(100, 50, 200, 250), Box(0, 0, 400, 300), Box(390, 290, 400, 300)]
    )
    assert vectors.shape == (3, scene.dimension)
    assert scene.crop_calls == [[(110, 220), (400, 300), (10, 10)]]


def test_no_box_gives_no_vector() -> None:
    scene = FakeSceneClassifier()
    engine = make_engine(scene_classifier=scene)
    assert engine.embed_open_crops(IMAGE, []).shape == (0, scene.dimension)


def test_phrases_get_vectors_and_the_same_phrase_gives_the_same_vector() -> None:
    engine = make_engine(scene_classifier=FakeSceneClassifier())
    vectors = engine.embed_open_phrases(["laptop", "bottle", "laptop"])
    assert vectors.shape[0] == 3 and np.allclose(vectors[0], vectors[2])
    assert not np.allclose(vectors[0], vectors[1])


def test_a_switched_off_scene_model_is_reported() -> None:
    engine = make_engine()
    engine._scene = None
    for call in (
        lambda: engine.embed_open_crops(IMAGE, []),
        lambda: engine.embed_open_phrases(["laptop"]),
        lambda: engine.open_vector_dimension,
        lambda: engine.open_vector_model,
    ):
        with pytest.raises(AssertionError, match="switched off"):
            call()


def test_the_model_name_and_the_size_come_from_the_scene_model() -> None:
    scene = FakeSceneClassifier()
    engine = make_engine(scene_classifier=scene)
    assert engine.open_vector_model == scene.name
    assert engine.open_vector_dimension == scene.dimension


# ---- The score of a pair ----


def make_classifier(scale: float, bias: float, dimension: int) -> SceneClassifier:
    classifier = object.__new__(SceneClassifier)  # No model is loaded
    classifier._lock = threading.Lock()
    classifier._room_texts = torch.zeros(1, dimension)
    classifier._model = SimpleNamespace(
        logit_scale=torch.tensor(float(np.log(scale))), logit_bias=torch.tensor(bias)
    )
    return classifier


def test_the_pair_score_is_a_scaled_cosine_with_a_bias_through_a_sigmoid() -> None:
    classifier = make_classifier(scale=10.0, bias=-5.0, dimension=2)
    crops = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    phrases = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = classifier.pair_probabilities(crops, phrases)
    assert scores.shape == (2, 1)
    assert scores[0, 0] == pytest.approx(1 / (1 + np.exp(-5.0)), abs=1e-4)  # cosine 1
    assert scores[1, 0] == pytest.approx(1 / (1 + np.exp(5.0)), abs=1e-4)  # cosine 0
    assert 0.0 <= scores.min() and scores.max() <= 1.0


def test_a_wrong_vector_size_is_refused() -> None:
    classifier = make_classifier(scale=10.0, bias=0.0, dimension=2)
    with pytest.raises(AssertionError, match="vector size"):
        classifier.pair_probabilities(
            np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)
        )
