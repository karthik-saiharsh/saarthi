"""Tests with the real models. They download the weights on the first run.

Run them with: uv run pytest -m slow
Set SAARTHI_TEST_IMAGE to the path of a photo to also test detection on a real picture.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ai.api import AiEngine
from ai.detector import CandidateDetector
from ai.embedder import RegionEmbedder
from ai.postprocess import select_candidates
from ai.settings import AiConfig, load_config
from ai.types import Box, RawDetection
from db.api import load_config as load_db_config
from tests.ai_support import FakeQueryEncoder, FakeVerifier
from tests.image_support import TEST_IMAGE

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def config() -> AiConfig:
    return load_config()


@pytest.fixture(scope="module")
def embedder(config: AiConfig) -> RegionEmbedder:
    return RegionEmbedder(config)


@pytest.fixture(scope="module")
def detector(config: AiConfig) -> CandidateDetector:
    return CandidateDetector(config)


def synthetic_image(size: tuple[int, int] = (640, 480)) -> Image.Image:
    rng: np.random.Generator = np.random.default_rng(0)
    image: Image.Image = Image.fromarray(
        rng.integers(80, 160, (size[1], size[0], 3), dtype=np.uint8)
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 300, 300), fill=(200, 30, 30))
    draw.ellipse((400, 200, 520, 320), fill=(30, 30, 200))
    return image


def test_embedder_output_shape_and_values(embedder: RegionEmbedder) -> None:
    boxes: list[Box] = [
        Box(100, 100, 300, 300),  # Large: pooled from the frame
        Box(400, 200, 520, 320),  # Large
        Box(10, 10, 50, 50),  # Small: embedded from a crop
    ]
    vectors: np.ndarray = embedder.embed(synthetic_image(), boxes)
    assert embedder.dimension == 384
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert embedder.name.startswith("dinov2-small@")


def test_embedder_is_repeatable_and_tells_regions_apart(
    embedder: RegionEmbedder,
) -> None:
    image: Image.Image = synthetic_image()
    boxes: list[Box] = [Box(100, 100, 300, 300), Box(400, 200, 520, 320)]
    first: np.ndarray = embedder.embed(image, boxes)
    second: np.ndarray = embedder.embed(image, boxes)
    assert np.allclose(first, second, atol=1e-5)
    assert (
        float(first[0] @ first[1]) < 0.99
    )  # Different regions are not the same vector


def test_embedder_without_boxes(embedder: RegionEmbedder) -> None:
    assert embedder.embed(synthetic_image(), []).shape == (0, 384)


def test_embedder_refuses_a_box_outside_the_image(embedder: RegionEmbedder) -> None:
    with pytest.raises(AssertionError):
        embedder.embed(synthetic_image(), [Box(600, 400, 700, 500)])


def test_detector_runs_all_three_passes(
    detector: CandidateDetector, config: AiConfig
) -> None:
    image: Image.Image = synthetic_image()
    detections: list[RawDetection] = detector.detect(
        image,
        config.detection.default_phrases,
        config.detection.small_phrases,
        include_generic=True,
        tiling=True,
    )
    for detection in detections:
        box: Box = detection.box
        assert 0 <= box.x0 < box.x1 <= image.width
        assert 0 <= box.y0 < box.y1 <= image.height
        assert detection.source in ("full", "tile", "generic")


def test_token_count(detector: CandidateDetector) -> None:
    assert detector.count_tokens("a pair of glasses") >= 4


@pytest.mark.skipif(TEST_IMAGE is None, reason="no test photo (set SAARTHI_TEST_IMAGE)")
def test_real_photo_end_to_end(
    detector: CandidateDetector, embedder: RegionEmbedder, config: AiConfig
) -> None:
    image: Image.Image = Image.open(TEST_IMAGE).convert("RGB")
    raw: list[RawDetection] = detector.detect(image, config.detection.default_phrases)
    candidates = select_candidates(
        raw,
        image.width,
        image.height,
        config.detection.default_phrases,
        config.detection,
    )
    vectors: np.ndarray = embedder.embed(image, [c.box for c in candidates])
    assert vectors.shape == (len(candidates), 384)


def test_engine_runs_the_stages_and_matches_the_db_vector_size(
    config: AiConfig,
) -> None:
    engine: AiEngine = AiEngine(config)
    assert engine.embedding_dimension == load_db_config().vector_dimension
    image: Image.Image = synthetic_image()
    report = engine.assess_quality(image)
    assert not engine.is_near_duplicate(report.phash, [])
    candidates = engine.detect_candidates(image)
    vectors: np.ndarray = engine.embed_regions(image, [c.box for c in candidates])
    assert vectors.shape == (len(candidates), engine.embedding_dimension)


@pytest.mark.skipif(TEST_IMAGE is None, reason="no test photo (set SAARTHI_TEST_IMAGE)")
def test_the_phone_is_located_in_the_real_photo(
    detector: CandidateDetector, embedder: RegionEmbedder, config: AiConfig
) -> None:
    engine: AiEngine = AiEngine(config, detector=detector, embedder=embedder, query_encoder=FakeQueryEncoder(), verifier=FakeVerifier())  # type: ignore[arg-type]
    image: Image.Image = Image.open(TEST_IMAGE).convert("RGB")
    found = engine.locate_object(image, "a mobile phone")
    assert found is not None and found.phrase == "a mobile phone"
    box: Box = found.box
    assert 0 <= box.x0 < box.x1 <= image.width and 0 <= box.y0 < box.y1 <= image.height
    # The test photo shows a phone, with its pink cover, at about (351, 264) to (478, 408).
    # The detector may box the phone with or without the cover. Both lie inside that area.
    phone: Box = Box(351, 264, 478, 408)
    inter_w: int = min(box.x1, phone.x1) - max(box.x0, phone.x0)
    inter_h: int = min(box.y1, phone.y1) - max(box.y0, phone.y0)
    assert inter_w > 0 and inter_h > 0
    assert (inter_w * inter_h) / box.area > 0.8  # The box lies on the phone
    assert box.area > 0.4 * phone.area  # And it is not a small part of the phone
    # An open-vocabulary detector gives a box even for an object that is not in the photo
    # (here: "a violin" gets the tall pink phone). That is why the carer confirms every crop.
    # The real object must score clearly higher than the invented one.
    invented = engine.locate_object(image, "a violin")
    assert invented is None or invented.det_score < found.det_score - 0.1


@pytest.mark.skipif(TEST_IMAGE is None, reason="no test photo (set SAARTHI_TEST_IMAGE)")
def test_the_views_of_the_phone_look_alike_and_differ_from_the_pen(
    detector: CandidateDetector, embedder: RegionEmbedder, config: AiConfig
) -> None:
    engine: AiEngine = AiEngine(config, detector=detector, embedder=embedder, query_encoder=FakeQueryEncoder(), verifier=FakeVerifier())  # type: ignore[arg-type]
    image: Image.Image = Image.open(TEST_IMAGE).convert("RGB")
    found = engine.locate_object(image, "a mobile phone")
    assert found is not None
    views = engine.embed_views(image, found.box)
    assert [name for name, _ in views] == list(config.enrollment.augmentations)
    vectors: np.ndarray = np.stack([v for _, v in views])
    assert vectors.shape == (4, 384) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    pen: np.ndarray = engine.embed_regions(image, [Box(719, 499, 763, 688)])[
        0
    ]  # The green pen
    phone_similarity: float = float(vectors[0] @ vectors[1:].T.mean(axis=1))
    assert (
        phone_similarity > float(vectors[0] @ pen) + 0.1
    )  # The views of one object are closer
