"""Tests of the real keypoint verifier on real photos. They need the model and the photos.

Run them with: uv run pytest -m slow tests/test_ai_verifier_model.py
The photos are 10 first-person photos in which a speckled bottle is in photos 2, 3, 8, 9 and 10.
Set SAARTHI_TEST_DATASET to their folder. The default is the folder `Dataset` next to the project.
The boxes are fixed, so the tests do not depend on the detector.
"""

import os
from pathlib import Path

import pytest
from PIL import Image

from ai.settings import AiConfig, load_config
from ai.types import Box
from ai.verifier import KeypointVerifier

DATASET: Path = Path(
    os.environ.get(
        "SAARTHI_TEST_DATASET", Path(__file__).resolve().parents[2] / "Dataset"
    )
)
# The boxes of the bottle that the detector found in the photos 2, 3 and 9 (enrollment), and in 10.
REFERENCES: dict[int, Box] = {
    2: Box(802, 490, 1029, 949),
    3: Box(872, 440, 1594, 1320),
    9: Box(650, 499, 817, 1071),
}
BOTTLE_IN_HAND: tuple[int, Box] = (10, Box(412, 70, 731, 998))
# Parts of the photos with no bottle: a notebook, a balcony, a backpack, a corridor.
OTHER_THINGS: list[tuple[int, Box]] = [
    (1, Box(130, 780, 640, 1000)),
    (4, Box(1000, 800, 1500, 1400)),
    (5, Box(1050, 300, 1500, 800)),
    (6, Box(300, 400, 900, 1200)),
]

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
def verifier(config: AiConfig) -> KeypointVerifier:
    return KeypointVerifier(config)


def crop(photo: int, box: Box, config: AiConfig) -> Image.Image:
    image: Image.Image = Image.open(DATASET / f"{photo}.jpeg").convert("RGB")
    pad: float = config.verification.crop_padding
    pad_x, pad_y = round(box.width * pad), round(box.height * pad)
    return image.crop(
        (
            max(0, box.x0 - pad_x),
            max(0, box.y0 - pad_y),
            min(image.width, box.x1 + pad_x),
            min(image.height, box.y1 + pad_y),
        )
    )


def test_the_same_bottle_has_many_matches_with_a_reference(
    verifier: KeypointVerifier, config: AiConfig
) -> None:
    photo, box = BOTTLE_IN_HAND
    candidate: Image.Image = crop(photo, box, config)
    best: int = max(
        verifier.count_matches(crop(n, b, config), candidate)
        for n, b in REFERENCES.items()
    )
    assert best >= config.verification.min_matches


def test_other_things_have_few_matches_with_every_reference(
    verifier: KeypointVerifier, config: AiConfig
) -> None:
    for photo, box in OTHER_THINGS:
        candidate: Image.Image = crop(photo, box, config)
        for number, reference_box in REFERENCES.items():
            count: int = verifier.count_matches(
                crop(number, reference_box, config), candidate
            )
            assert count < config.verification.min_matches, (photo, number, count)


def test_the_count_is_the_same_in_both_orders_within_a_small_difference(
    verifier: KeypointVerifier, config: AiConfig
) -> None:
    first: Image.Image = crop(9, REFERENCES[9], config)
    second: Image.Image = crop(*BOTTLE_IN_HAND, config)
    forward: int = verifier.count_matches(first, second)
    backward: int = verifier.count_matches(second, first)
    assert abs(forward - backward) <= max(10, forward // 5)
