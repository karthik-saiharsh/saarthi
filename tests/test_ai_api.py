"""Tests of the AiEngine with fake models, and of open_image. No weights are loaded."""

import dataclasses
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageFilter

from ai.api import AiEngine, open_image
from ai.errors import ImageError
from ai.settings import AiConfig, load_config
from ai.types import Box, MatchObject, MatchRegion, RawDetection
from tests.ai_support import FakeQueryEncoder

CONFIG: AiConfig = load_config()


def encode(image: Image.Image, fmt: str = "JPEG", **kwargs: object) -> bytes:
    buffer: BytesIO = BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def noise_image(size: tuple[int, int] = (640, 480), seed: int = 0) -> Image.Image:
    rng: np.random.Generator = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_open_image_decodes_to_rgb() -> None:
    image: Image.Image = open_image(encode(noise_image()), 10_000_000)
    assert image.mode == "RGB" and image.size == (640, 480)


def test_open_image_converts_other_modes_to_rgb() -> None:
    rgba: Image.Image = noise_image((50, 40)).convert("RGBA")
    assert open_image(encode(rgba, "PNG"), 10_000_000).mode == "RGB"
    assert (
        open_image(encode(noise_image((50, 40)).convert("L"), "PNG"), 10_000_000).mode
        == "RGB"
    )


def test_open_image_applies_the_exif_orientation() -> None:
    source: Image.Image = noise_image((200, 100))
    exif: Image.Exif = Image.Exif()
    exif[0x0112] = 6  # The camera was rotated: show it turned by 90 degrees
    rotated: Image.Image = open_image(encode(source, exif=exif.tobytes()), 10_000_000)
    assert rotated.size == (100, 200)


def test_open_image_refuses_a_large_image() -> None:
    with pytest.raises(ImageError):
        open_image(encode(noise_image((100, 100))), 5_000)


@pytest.mark.parametrize(
    "data", [b"", b"not an image", b"\xff\xd8\xff\xe0" + b"x" * 50]
)
def test_open_image_refuses_bad_files(data: bytes) -> None:
    with pytest.raises(ImageError):
        open_image(data, 10_000_000)


def test_open_image_refuses_a_truncated_file() -> None:
    data: bytes = encode(noise_image())
    with pytest.raises(ImageError):
        open_image(data[: len(data) // 2], 10_000_000)


class FakeDetector:
    name = "fake-detector@00000000"

    def __init__(self, detections: list[RawDetection]) -> None:
        self.detections: list[RawDetection] = detections
        self.calls: list[tuple] = []

    def detect(
        self,
        image,
        phrases,
        small_phrases=(),
        include_generic=False,
        tiling=None,
        context_phrases=True,
    ):
        self.calls.append((list(phrases), list(small_phrases), include_generic, tiling))
        return self.detections


class FakeEmbedder:
    name = "fake-embedder@00000000"
    dimension = 384

    def embed(self, image, boxes):
        vectors: np.ndarray = np.zeros((len(boxes), self.dimension), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def make_engine(
    detections: list[RawDetection] | None = None,
) -> tuple[AiEngine, FakeDetector]:
    detector: FakeDetector = FakeDetector(detections or [])
    engine: AiEngine = AiEngine(CONFIG, detector=detector, embedder=FakeEmbedder(), query_encoder=FakeQueryEncoder())  # type: ignore[arg-type]
    return engine, detector


def test_engine_reports_the_embedding_model() -> None:
    engine, _ = make_engine()
    assert engine.embedding_model == "fake-embedder@00000000"
    assert engine.embedding_dimension == 384


def test_quality_assessment_and_blur_gate() -> None:
    # The shipped config has the gate off (blur_min = 0), so this test sets a limit.
    config: AiConfig = dataclasses.replace(
        CONFIG, quality=dataclasses.replace(CONFIG.quality, blur_min=100.0)
    )
    engine: AiEngine = AiEngine(config, detector=FakeDetector([]), embedder=FakeEmbedder(), query_encoder=FakeQueryEncoder())  # type: ignore[arg-type]
    sharp: Image.Image = noise_image()
    blurred: Image.Image = sharp.filter(ImageFilter.GaussianBlur(10))
    sharp_report = engine.assess_quality(sharp)
    blurred_report = engine.assess_quality(blurred)
    assert len(sharp_report.phash) == 16
    assert not engine.is_blurry(sharp_report)
    assert engine.is_blurry(blurred_report)


def test_near_duplicate_check() -> None:
    engine, _ = make_engine()
    base: str = "0" * 16
    assert engine.is_near_duplicate(base, ["f" * 16, "0" * 15 + "7"])  # 3 bits apart
    assert not engine.is_near_duplicate(
        base, ["f" * 16, "0" * 14 + "ff"]
    )  # 8 bits apart
    assert not engine.is_near_duplicate(base, [])


def test_detect_candidates_uses_the_config_phrases_by_default() -> None:
    engine, detector = make_engine()
    assert engine.detect_candidates(noise_image()) == []
    phrases, small, generic, tiling = detector.calls[0]
    assert phrases == list(CONFIG.detection.default_phrases)
    assert small == list(CONFIG.detection.small_phrases)
    assert generic is False and tiling is None


def test_detect_candidates_passes_custom_phrases_and_the_generic_switch() -> None:
    engine, detector = make_engine()
    engine.detect_candidates(
        noise_image(),
        phrases=["a bottle", "a set of keys"],
        include_generic=True,
        tiling=True,
    )
    phrases, small, generic, tiling = detector.calls[0]
    assert phrases == ["a bottle", "a set of keys"]
    assert small == ["a set of keys"]  # Only the small phrases that are in use
    assert generic is True and tiling is True


def test_detect_candidates_cleans_up_and_marks_quality() -> None:
    detections: list[RawDetection] = [
        RawDetection(Box(100, 100, 260, 260), 0.6, "bottle", "full"),
        RawDetection(Box(101, 101, 261, 259), 0.4, "a flask", "full"),  # Duplicate
        RawDetection(Box(0, 300, 120, 400), 0.5, "keys", "full"),  # Touches the edge
    ]
    engine, _ = make_engine(detections)
    result = engine.detect_candidates(noise_image())
    assert [c.box for c in result] == [Box(100, 100, 260, 260), Box(0, 300, 120, 400)]
    assert result[0].phrases[0] == "a bottle" and result[0].quality_ok
    assert not result[1].quality_ok


def test_embed_regions_passes_through() -> None:
    engine, _ = make_engine()
    vectors = engine.embed_regions(
        noise_image(), [Box(0, 0, 10, 10), Box(5, 5, 50, 50)]
    )
    assert vectors.shape == (2, 384)


def test_the_shipped_config_has_the_blur_gate_off() -> None:
    # The owner chose to switch the gate off until real wearable frames exist.
    engine, _ = make_engine()
    flat: Image.Image = Image.new("RGB", (200, 200), (120, 120, 120))
    assert not engine.is_blurry(engine.assess_quality(flat))


def test_the_engine_matches_regions_with_the_config_values() -> None:
    engine, _ = make_engine()
    views: np.ndarray = np.zeros((1, 384), dtype=np.float32)
    views[0, 0] = 1.0
    bottle = MatchObject(7, True, ("a bottle",), views, 0.8, 0.9, 0.03)
    result = engine.match_regions(
        views.copy(), [MatchRegion(("a bottle",), True)], [bottle]
    )
    assert [(s.object_id, s.strong) for s in result] == [(7, True)]


def test_the_engine_encodes_texts_with_the_query_encoder() -> None:
    engine, _ = make_engine()
    vectors: np.ndarray = engine.encode_texts(["a blue bottle", "my keys"])
    assert vectors.shape == (2, 64) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert engine.query_model == "fake-query-encoder@00000000"
