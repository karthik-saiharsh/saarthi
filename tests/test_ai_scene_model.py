"""Tests of the real scene classifier on real photos. They need the model and the photos.

Run them with: uv run pytest -m slow tests/test_ai_scene_model.py
The photos are the 14 photos of the folder `Dataset` (set SAARTHI_TEST_DATASET, or keep the folder next to the project).
"""

import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, PreTrainedTokenizerFast

from ai.scene import SceneClassifier, context_box
from ai.settings import AiConfig, load_config
from ai.types import Box

DATASET: Path = Path(
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


@pytest.fixture(scope="module")
def config() -> AiConfig:
    return load_config()


@pytest.fixture(scope="module")
def classifier(config: AiConfig) -> SceneClassifier:
    return SceneClassifier(config)


def photo(number: int) -> Image.Image:
    return Image.open(DATASET / f"{number}.jpeg").convert("RGB")


def test_the_scores_are_probabilities_of_the_words_of_the_lists(
    classifier: SceneClassifier, config: AiConfig
) -> None:
    rooms = classifier.room_scores(photo(6))
    assert list(rooms) == list(config.scene.rooms)
    assert sum(rooms.values()) == pytest.approx(1.0, abs=1e-4)
    surfaces = classifier.surface_scores(photo(2))
    assert {name for name, _ in surfaces} == set(config.scene.surfaces)
    assert sum(score for _, score in surfaces) == pytest.approx(1.0, abs=1e-4)
    assert [score for _, score in surfaces] == sorted(
        (score for _, score in surfaces), reverse=True
    )


def test_the_scores_equal_those_of_the_full_model(
    classifier: SceneClassifier, config: AiConfig
) -> None:
    """The classifier encodes the sentences once. The result must equal a full run of the model."""
    models = config.models
    kwargs = {
        "revision": models.scene_revision,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    processor = AutoImageProcessor.from_pretrained(models.scene_id, **kwargs)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(models.scene_id, **kwargs)
    model = AutoModel.from_pretrained(
        models.scene_id, use_safetensors=True, **kwargs
    ).eval()
    sentences = [config.scene.prompt.format(label=r) for r in config.scene.rooms]
    tokens = tokenizer(
        sentences,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    image = photo(14)
    with torch.inference_mode():
        logits = model(
            input_ids=tokens["input_ids"],
            pixel_values=processor(images=image, return_tensors="pt")["pixel_values"],
        ).logits_per_image[0]
    expected = torch.softmax(logits, dim=0).tolist()
    found = list(classifier.room_scores(image).values())
    assert found == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    ("number", "room"),
    [(6, "a corridor"), (7, "a corridor"), (4, "a balcony"), (14, "a study room")],
)
def test_the_room_of_a_clear_photo(
    classifier: SceneClassifier, number: int, room: str
) -> None:
    rooms = classifier.room_scores(photo(number))
    assert max(rooms, key=rooms.get) == room


@pytest.mark.parametrize(
    ("number", "box", "surface"),
    [
        (2, Box(803, 518, 1029, 949), "a table"),  # The bottle in a hand, above a table
        (9, Box(650, 501, 817, 1072), "a window sill"),  # The bottle on a ledge
        (14, Box(324, 18, 472, 326), "a desk"),  # The small bottle on a desk
    ],
)
def test_the_surface_under_the_bottle(
    classifier: SceneClassifier,
    config: AiConfig,
    number: int,
    box: Box,
    surface: str,
) -> None:
    image = photo(number)
    # The same crop that `AiEngine.score_surfaces` makes: the box with space around it.
    around = context_box(box, image.width, image.height, config.scene.surface_context)
    crop = image.crop((around.x0, around.y0, around.x1, around.y1))
    assert classifier.surface_scores(crop)[0][0] == surface
