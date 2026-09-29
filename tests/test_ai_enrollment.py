"""Tests of the helpers for the views of a reference photo, and of the engine functions E2 to E4."""

import dataclasses

import numpy as np
import pytest
from PIL import Image

from ai.api import AiEngine, CalibrationError
from ai.enrollment import MIN_BOX_SIDE, brighten, scale_box
from ai.settings import AiConfig, load_config
from ai.types import Box, RawDetection
from tests.ai_support import FakeDetector, FakeEmbedder, FakeQueryEncoder

CONFIG: AiConfig = load_config()


def engine_with(
    detector: FakeDetector | None = None,
    embedder: FakeEmbedder | None = None,
    augmentations: tuple[str, ...] | None = None,
) -> AiEngine:
    config: AiConfig = CONFIG
    if augmentations is not None:
        config = dataclasses.replace(
            CONFIG,
            enrollment=dataclasses.replace(
                CONFIG.enrollment, augmentations=augmentations
            ),
        )
    return AiEngine(config, detector=detector or FakeDetector(), embedder=embedder or FakeEmbedder(), query_encoder=FakeQueryEncoder())  # type: ignore[arg-type]


def photo(width: int = 200, height: int = 150, grey: int = 100) -> Image.Image:
    return Image.new("RGB", (width, height), (grey, grey, grey))


# ---- Pure helpers ----


def test_scale_box_grows_and_shrinks_around_the_centre() -> None:
    box: Box = Box(50, 40, 150, 120)  # 100 x 80
    bigger: Box = scale_box(box, 0.1, 300, 300)
    smaller: Box = scale_box(box, -0.1, 300, 300)
    assert (bigger.width, bigger.height) == (110, 88) and (
        smaller.width,
        smaller.height,
    ) == (90, 72)
    for scaled in (bigger, smaller):
        assert (scaled.x0 + scaled.x1) / 2 == pytest.approx(100, abs=1)
        assert (scaled.y0 + scaled.y1) / 2 == pytest.approx(80, abs=1)


def test_scale_box_stays_inside_the_image_and_keeps_a_minimum_size() -> None:
    assert scale_box(Box(0, 0, 100, 100), 0.5, 120, 110) == Box(0, 0, 120, 110)
    tiny: Box = scale_box(Box(10, 10, 12, 12), -0.9, 50, 50)
    assert tiny.width >= MIN_BOX_SIDE and tiny.height >= MIN_BOX_SIDE
    corner: Box = scale_box(Box(0, 0, 2, 2), -0.5, 10, 10)
    assert 0 <= corner.x0 < corner.x1 <= 10 and 0 <= corner.y0 < corner.y1 <= 10


def test_brighten_makes_a_brighter_copy() -> None:
    original: Image.Image = photo(grey=100)
    brighter: Image.Image = brighten(original, 0.15)
    assert brighter is not original
    assert np.asarray(brighter).mean() == pytest.approx(115, abs=1)
    assert np.asarray(original).mean() == 100  # The original is not changed


# ---- E2: locate the object ----


def test_the_best_box_of_the_category_is_the_object() -> None:
    detector: FakeDetector = FakeDetector()
    detector.detections = [
        RawDetection(Box(10, 10, 60, 90), 0.40, "bottle", "full"),
        RawDetection(Box(80, 20, 150, 120), 0.62, "bottle", "full"),
        RawDetection(Box(5, 100, 40, 140), 0.90, "cup", "full"),  # Another category
    ]
    found = engine_with(detector).locate_object(photo(), "a bottle")
    assert found is not None
    assert (
        found.box == Box(80, 20, 150, 120)
        and found.det_score == 0.62
        and found.phrase == "a bottle"
    )


def test_a_synonym_also_finds_the_object() -> None:
    detector: FakeDetector = FakeDetector()
    detector.detections = [RawDetection(Box(10, 10, 80, 100), 0.5, "flask", "full")]
    found = engine_with(detector).locate_object(photo(), "a bottle", ["a flask"])
    assert found is not None and found.phrase == "a flask"


def test_nothing_found_gives_none() -> None:
    detector: FakeDetector = FakeDetector()
    assert engine_with(detector).locate_object(photo(), "a bottle") is None
    detector.detections = [RawDetection(Box(10, 10, 80, 100), 0.7, "cup", "full")]
    assert (
        engine_with(detector).locate_object(photo(), "a bottle") is None
    )  # Only another category


def test_a_hand_is_not_the_object_and_the_detector_gets_one_plain_pass() -> None:
    detector: FakeDetector = FakeDetector()
    detector.detections = [RawDetection(Box(10, 10, 80, 100), 0.9, "a hand", "full")]
    assert engine_with(detector).locate_object(photo(), "a bottle") is None
    assert detector.generic_flags == [False]  # No generic pass


def test_the_detector_gets_only_the_phrases_of_the_object() -> None:
    # In a real photo, a book cover got the label "a picture frame", and the depiction rule
    # dropped the phone that lies on it. The enrollment prompt has no hand and no depiction phrases.
    detector: FakeDetector = FakeDetector()
    engine_with(detector).locate_object(photo(), "a bottle")
    assert detector.context_flags == [False]


def test_a_detector_error_is_not_hidden() -> None:
    detector: FakeDetector = FakeDetector()
    detector.error = RuntimeError("model broke")
    with pytest.raises(RuntimeError):
        engine_with(detector).locate_object(photo(), "a bottle")


# ---- E3: the views ----


def test_one_view_for_each_augmentation_in_the_order_of_the_config() -> None:
    embedder: FakeEmbedder = FakeEmbedder(one_hot_by_box=True)
    views = engine_with(embedder=embedder).embed_views(
        photo(300, 300), Box(100, 100, 200, 180)
    )
    assert [name for name, _ in views] == [
        "original",
        "box_smaller",
        "box_larger",
        "brighter",
    ]
    assert all(
        vector.shape == (384,) and vector.dtype == np.float32 for _, vector in views
    )
    original, smaller, larger, brighter = (vector for _, vector in views)
    assert not np.array_equal(original, smaller) and not np.array_equal(
        original, larger
    )
    assert np.array_equal(
        original, brighter
    )  # The brighter view uses the box of the original


def test_the_box_views_share_one_model_run_and_the_brighter_view_has_its_own() -> None:
    embedder: FakeEmbedder = FakeEmbedder()
    engine_with(embedder=embedder).embed_views(
        photo(300, 300, grey=100), Box(100, 100, 200, 180)
    )
    assert len(embedder.calls) == 2
    box_call, bright_call = embedder.calls
    assert len(box_call[1]) == 3 and box_call[0] == pytest.approx(100, abs=1)
    assert len(bright_call[1]) == 1 and bright_call[0] == pytest.approx(115, abs=1)


def test_the_config_chooses_the_views() -> None:
    embedder: FakeEmbedder = FakeEmbedder()
    views = engine_with(
        embedder=embedder, augmentations=("original", "brighter")
    ).embed_views(photo(300, 300), Box(100, 100, 200, 180))
    assert [name for name, _ in views] == ["original", "brighter"]
    views = engine_with(
        embedder=FakeEmbedder(), augmentations=("box_larger",)
    ).embed_views(photo(300, 300), Box(100, 100, 200, 180))
    assert [name for name, _ in views] == ["box_larger"]


# ---- E4 through the engine ----


def test_the_engine_calibrates_and_knows_the_minimum_number_of_photos() -> None:
    engine: AiEngine = engine_with()
    assert engine.min_photos == CONFIG.enrollment.min_photos == 3
    rng = np.random.default_rng(0)
    base: np.ndarray = rng.normal(size=384)
    views: np.ndarray = np.stack(
        [base + rng.normal(scale=0.3, size=384) for _ in range(3)]
    )
    views = (views / np.linalg.norm(views, axis=1, keepdims=True)).astype(np.float32)
    result = engine.calibrate_object(
        views, [1, 2, 3], np.empty((0, 384), dtype=np.float32)
    )
    assert 0 < result.tau < result.tau_strong <= 1.0 and result.n_photos == 3
    with pytest.raises(CalibrationError):
        engine.calibrate_object(views[:2], [1, 1], np.empty((0, 384), dtype=np.float32))


def test_the_parameters_are_checked() -> None:
    engine: AiEngine = engine_with()
    with pytest.raises(AssertionError):
        engine.locate_object("photo", "a bottle")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        engine.embed_views(photo(), (1, 2, 3, 4))  # type: ignore[arg-type]
