"""Tests of the verification (stage I5) in the engine. The keypoint model is a fake."""

import dataclasses

import pytest
from PIL import Image

from ai.api import AiEngine
from ai.errors import InferenceError
from ai.settings import load_config
from ai.types import Box, VerifyResult
from tests.ai_support import (
    FakeDetector,
    FakeEmbedder,
    FakeQueryEncoder,
    FakeVerifier,
    make_engine,
)

NEEDED: int = load_config().verification.min_matches


def picture(width: int = 40, height: int = 30) -> Image.Image:
    return Image.new("RGB", (width, height), (120, 120, 120))


def engine_with(verifier: FakeVerifier) -> AiEngine:
    return make_engine(verifier=verifier)


def test_a_candidate_that_matches_a_reference_is_verified() -> None:
    verifier: FakeVerifier = FakeVerifier(answer=NEEDED + 20)
    result: VerifyResult = engine_with(verifier).verify_match(
        [picture(), picture()], picture()
    )
    assert result == VerifyResult(True, NEEDED + 20, 1)
    assert len(verifier.calls) == 1  # It stops at the first reference that passes


def test_the_limit_itself_passes_and_one_less_does_not() -> None:
    assert (
        engine_with(FakeVerifier(NEEDED)).verify_match([picture()], picture()).verified
    )
    below: VerifyResult = engine_with(FakeVerifier(NEEDED - 1)).verify_match(
        [picture()], picture()
    )
    assert not below.verified and below.matches == NEEDED - 1


def test_every_reference_is_tried_when_none_passes_and_the_best_count_is_kept() -> None:
    verifier: FakeVerifier = FakeVerifier()
    verifier.answers = [3, 11, 7]
    result: VerifyResult = engine_with(verifier).verify_match(
        [picture(), picture(), picture()], picture()
    )
    assert result == VerifyResult(False, 11, 3)


def test_a_later_reference_can_pass() -> None:
    verifier: FakeVerifier = FakeVerifier()
    verifier.answers = [4, NEEDED + 1, 99]
    result: VerifyResult = engine_with(verifier).verify_match(
        [picture(), picture(), picture()], picture()
    )
    assert result == VerifyResult(True, NEEDED + 1, 2)
    assert len(verifier.calls) == 2  # The third reference is not needed


def test_no_reference_means_not_verified() -> None:
    verifier: FakeVerifier = FakeVerifier()
    result: VerifyResult = engine_with(verifier).verify_match([], picture())
    assert result == VerifyResult(False, 0, 0) and verifier.calls == []


def test_the_candidate_is_the_second_picture_of_each_pair() -> None:
    verifier: FakeVerifier = FakeVerifier(answer=0)
    engine_with(verifier).verify_match([picture(10, 20)], picture(30, 40))
    assert verifier.calls == [((10, 20), (30, 40))]


def test_an_error_of_the_model_reaches_the_caller() -> None:
    verifier: FakeVerifier = FakeVerifier()
    verifier.error = InferenceError("test")
    with pytest.raises(InferenceError):
        engine_with(verifier).verify_match([picture()], picture())


def test_the_crop_has_space_around_the_box_inside_the_picture() -> None:
    engine: AiEngine = make_engine()
    pad: float = load_config().verification.crop_padding
    image: Image.Image = picture(400, 300)
    box: Box = Box(100, 100, 200, 160)  # 100 x 60
    crop: Image.Image = engine.verification_crop(image, box)
    assert crop.size == (100 + 2 * round(100 * pad), 60 + 2 * round(60 * pad))


def test_the_crop_is_cut_at_the_edge_of_the_picture() -> None:
    engine: AiEngine = make_engine()
    image: Image.Image = picture(400, 300)
    corner: Image.Image = engine.verification_crop(image, Box(0, 0, 100, 100))
    far: Image.Image = engine.verification_crop(image, Box(300, 200, 400, 300))
    pad: int = round(100 * load_config().verification.crop_padding)
    assert corner.size == (100 + pad, 100 + pad) and far.size == (100 + pad, 100 + pad)


def test_a_switched_off_verification_loads_no_model_and_refuses_calls() -> None:
    base = load_config()
    config = dataclasses.replace(
        base, verification=dataclasses.replace(base.verification, enabled=False)
    )
    engine: AiEngine = AiEngine(
        config,
        detector=FakeDetector(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        query_encoder=FakeQueryEncoder(),  # type: ignore[arg-type]
    )  # The real verifier would be loaded here if the switch did not work
    assert not engine.verification_enabled
    with pytest.raises(AssertionError):
        engine.verify_match([picture()], picture())
    assert make_engine().verification_enabled
