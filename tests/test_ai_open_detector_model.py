"""The real broad detector on real photos. It needs the weights (about 28 MB) and the photos.

Run it with: uv run pytest -m slow tests/test_ai_open_detector_model.py
The photos are the folder `Dataset` (set SAARTHI_TEST_DATASET, or keep the folder next to the project).
"""

import os
from pathlib import Path

import pytest
from PIL import Image

from ai.open_detector import OpenDetector
from ai.settings import load_config

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


@pytest.fixture(scope="module")
def detector() -> OpenDetector:
    return OpenDetector(load_config())


def test_the_laptop_on_the_desk_is_found(detector: OpenDetector) -> None:
    found = detector.detect(Image.open(DATASET / "14.jpeg").convert("RGB"))
    assert any(d.label == "laptop" and d.confidence > 0.5 for d in found)


def test_the_bottle_in_the_hand_is_found(detector: OpenDetector) -> None:
    found = detector.detect(Image.open(DATASET / "10.jpeg").convert("RGB"))
    assert any("bottle" in d.label and d.confidence > 0.5 for d in found)


def test_the_boxes_stay_inside_the_frame(detector: OpenDetector) -> None:
    image = Image.open(DATASET / "13.jpeg").convert("RGB")
    for d in detector.detect(image):
        assert 0 <= d.box.x0 < d.box.x1 <= image.width
        assert 0 <= d.box.y0 < d.box.y1 <= image.height
