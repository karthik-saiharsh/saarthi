"""Tests of the evidence-based matching (ai/evidence.py) and of its engine methods. No real model."""

import dataclasses

import numpy as np
import pytest
from PIL import Image

from ai.errors import MatchingError
from ai.evidence import (
    class_shares,
    evidence_score,
    photo_agreement,
    score_regions,
    select_regions,
)
from ai.settings import MatchingConfig, load_config
from ai.types import Box, MatchObject, MatchRegion
from tests.ai_support import FakeSceneClassifier, make_engine

CONFIG: MatchingConfig = load_config().matching
DINO: int = 8
SIGLIP: int = 6


def unit(*values: float, size: int = DINO) -> np.ndarray:
    vector: np.ndarray = np.zeros(size, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def axis(index: int, size: int = DINO) -> np.ndarray:
    vector: np.ndarray = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def make_object(
    object_id: int,
    views: list[np.ndarray],
    references: list[np.ndarray],
    tracked: bool = True,
    phrases: tuple[str, ...] = ("a watch",),
) -> MatchObject:
    """One view and one reference for each photo, photo ids 1, 2, ..."""
    return MatchObject(
        object_id=object_id,
        tracked=tracked,
        phrases=phrases,
        views=np.stack(views),
        tau=0.9,
        tau_strong=0.95,
        margin_min=0.03,
        view_photos=np.arange(1, len(views) + 1),
        references=np.stack(references),
        reference_photos=np.arange(1, len(references) + 1),
    )


def logits_for(n_regions: int, n_classes: int, winner: int) -> np.ndarray:
    """Logits where the class `winner` clearly wins for every region."""
    logits: np.ndarray = np.full((n_regions, n_classes), -5.0, dtype=np.float32)
    logits[:, winner] = 5.0
    return logits


# ---- Pure functions ----


def test_photo_agreement_is_the_mean_of_the_best_photos() -> None:
    scores = np.array([0.9, 0.2, 0.5, 0.6, 0.1])
    photos = np.array([1, 1, 2, 2, 3])
    assert photo_agreement(scores, photos, 2) == pytest.approx((0.9 + 0.6) / 2)
    assert photo_agreement(scores, photos, 1) == pytest.approx(0.9)
    # More photos than exist: the mean of all photos.
    assert photo_agreement(scores, photos, 9) == pytest.approx((0.9 + 0.6 + 0.1) / 3)


def test_class_shares_add_up_to_one() -> None:
    shares = class_shares(np.array([[1.0, 2.0, 3.0], [100.0, 0.0, 0.0]]))
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert shares[1, 0] == pytest.approx(1.0)


def test_the_meaning_can_stop_a_match_but_cannot_make_one() -> None:
    centre = {
        "dino_agree": CONFIG.dino_centre,
        "margin": 0.0,
        "visual": CONFIG.visual_centre,
    }
    certain: float = evidence_score(meaning=0.999999, config=CONFIG, **centre)
    unsure: float = evidence_score(meaning=0.5, config=CONFIG, **centre)
    wrong: float = evidence_score(meaning=0.000001, config=CONFIG, **centre)
    assert certain - unsure == pytest.approx(CONFIG.weight_meaning * CONFIG.meaning_cap)
    assert wrong < unsure - 3 * CONFIG.weight_meaning


def test_the_margin_is_capped() -> None:
    base = {"dino_agree": 0.5, "visual": 0.8, "meaning": 0.9, "config": CONFIG}
    at_cap: float = evidence_score(margin=CONFIG.margin_cap, **base)
    above: float = evidence_score(margin=CONFIG.margin_cap + 0.5, **base)
    assert above == pytest.approx(at_cap)


# ---- Region selection ----


def test_select_regions_keeps_good_similar_regions_in_order() -> None:
    obj = make_object(1, [axis(0)], [axis(0, SIGLIP)])
    vectors = np.stack([unit(0.3, 1.0), unit(1.0, 0.1), axis(3), unit(1.0, 0.5)])
    regions = [
        MatchRegion((), True),
        MatchRegion((), True),
        MatchRegion((), True),  # Not similar: cosine 0
        MatchRegion((), False),  # Similar, but not a good crop
    ]
    assert select_regions(vectors, regions, [obj], CONFIG) == [1, 0]


def test_select_regions_ignores_look_alikes_and_respects_the_limit() -> None:
    look_alike = make_object(2, [axis(1)], [axis(1, SIGLIP)], tracked=False)
    vectors = np.stack([axis(1)] * 3)
    regions = [MatchRegion((), True)] * 3
    assert select_regions(vectors, regions, [look_alike], CONFIG) == []

    obj = make_object(1, [axis(1)], [axis(1, SIGLIP)])
    small = dataclasses.replace(CONFIG, max_regions=2)
    assert len(select_regions(vectors, regions, [obj], small)) == 2


# ---- Scoring ----


def test_a_region_that_agrees_everywhere_is_a_strong_sighting() -> None:
    watch = make_object(1, [axis(0), axis(0)], [axis(0, SIGLIP), axis(0, SIGLIP)])
    vectors = np.stack([axis(0)])
    found = score_regions(
        [0], vectors, np.stack([axis(0, SIGLIP)]), logits_for(1, 2, 0), [watch], CONFIG
    )
    assert len(found) == 1
    assert found[0].object_id == 1 and found[0].strong
    assert found[0].score is not None and found[0].score >= CONFIG.score_sure


def test_a_low_dino_score_is_rescued_by_the_other_signals() -> None:
    """The watch on a wrist: DINOv2 is unsure, SigLIP and the meaning agree."""
    watch = make_object(
        1, [unit(1.0, 1.0, 1.0), unit(1.0, 1.0, 0.0, 1.0)], [axis(0, SIGLIP)] * 2
    )
    region = unit(1.0, 0.0, 0.0, 0.0, 3.0)  # Cosine about 0.2 with the views
    found = score_regions(
        [0],
        np.stack([region]),
        np.stack([axis(0, SIGLIP)]),
        logits_for(1, 2, 0),
        [watch],
        CONFIG,
    )
    assert len(found) == 1


def test_the_wrong_kind_of_thing_is_refused() -> None:
    """Realistic pictures (SigLIP cosine 0.74, DINOv2 0.78). Only the meaning differs: the object, or "a hand"."""
    watch = make_object(1, [axis(0)] * 2, [unit(1.0, 0.6, size=SIGLIP)] * 2)
    crop = unit(0.3, 1.0, size=SIGLIP)
    region = unit(1.0, 0.8)
    args = ([0], np.stack([region]), np.stack([crop]))
    assert [
        s.object_id for s in score_regions(*args, logits_for(1, 2, 0), [watch], CONFIG)
    ] == [1]
    assert score_regions(*args, logits_for(1, 2, 1), [watch], CONFIG) == []


def test_one_lucky_view_scores_lower_than_agreement() -> None:
    lucky = make_object(1, [axis(0), axis(1)], [axis(0, SIGLIP), axis(1, SIGLIP)])
    agreed = make_object(1, [axis(0), axis(0)], [axis(0, SIGLIP), axis(0, SIGLIP)])
    args = ([0], np.stack([axis(0)]), np.stack([axis(0, SIGLIP)]), logits_for(1, 2, 0))
    low = dataclasses.replace(CONFIG, score_maybe=-100.0)
    score_lucky = score_regions(*args, [lucky], low)[0].score
    score_agreed = score_regions(*args, [agreed], low)[0].score
    assert score_lucky is not None and score_agreed is not None
    assert score_lucky < score_agreed


def test_a_look_alike_blocks_the_match_and_is_never_reported() -> None:
    watch = make_object(1, [axis(0)] * 2, [axis(0, SIGLIP)] * 2)
    twin = make_object(2, [axis(0)] * 2, [axis(0, SIGLIP)] * 2, tracked=False)
    other = make_object(3, [axis(2)] * 2, [axis(2, SIGLIP)] * 2, tracked=False)
    args = ([0], np.stack([axis(0)]), np.stack([axis(0, SIGLIP)]), logits_for(1, 3, 0))
    assert score_regions(*args, [watch, twin], CONFIG) == []
    found = score_regions(*args, [watch, other], CONFIG)
    assert [s.object_id for s in found] == [1]
    assert found[0].margin == pytest.approx(1.0)


def test_a_small_margin_refuses_even_a_high_score() -> None:
    watch = make_object(1, [axis(0)] * 2, [axis(0, SIGLIP)] * 2)
    near = make_object(
        2, [unit(1.0, 0.25)] * 2, [axis(3, SIGLIP)] * 2
    )  # Cosine 0.97 with the region
    args = ([0], np.stack([axis(0)]), np.stack([axis(0, SIGLIP)]), logits_for(1, 3, 0))
    assert [s.object_id for s in score_regions(*args, [watch, near], CONFIG)] == []


def test_one_sighting_for_each_object_the_best_region() -> None:
    watch = make_object(1, [axis(0)] * 2, [axis(0, SIGLIP)] * 2)
    vectors = np.stack([unit(1.0, 0.3), axis(0)])
    crops = np.stack([axis(0, SIGLIP), axis(0, SIGLIP)])
    found = score_regions([0, 1], vectors, crops, logits_for(2, 2, 0), [watch], CONFIG)
    assert len(found) == 1 and found[0].region_index == 1


def test_bad_data_raises_a_matching_error() -> None:
    watch = make_object(1, [axis(0)], [axis(0, SIGLIP)])
    no_refs = dataclasses.replace(watch, references=None)
    vectors = np.stack([axis(0)])
    crops = np.stack([axis(0, SIGLIP)])
    with pytest.raises(MatchingError):
        score_regions([0], vectors, crops, logits_for(1, 2, 0), [no_refs], CONFIG)
    with pytest.raises(MatchingError):
        score_regions([0], vectors * 2, crops, logits_for(1, 2, 0), [watch], CONFIG)
    with pytest.raises(MatchingError):
        score_regions([0], vectors, crops, np.zeros((1, 0)), [watch], CONFIG)


# ---- Engine ----


def test_the_engine_uses_evidence_only_with_the_scene_model() -> None:
    engine = make_engine()
    assert engine.matching_mode == load_config().matching.mode
    engine._scene = None  # The scene model is switched off
    assert engine.matching_mode == "tau"


def test_the_engine_scores_a_frame_and_keeps_the_phrase_vectors() -> None:
    scene = FakeSceneClassifier()
    engine = make_engine(scene_classifier=scene)
    picture = Image.new("RGB", (100, 100), (10, 20, 30))
    scene.crop_labels[(10, 20, 30)] = "a watch"
    crops = engine.embed_evidence_crops(picture, [Box(10, 10, 40, 40)])
    assert crops.shape == (1, scene.dimension)
    watch = MatchObject(
        object_id=1,
        tracked=True,
        phrases=("a watch",),
        views=np.stack([axis(0, 384)] * 2),
        tau=0.9,
        tau_strong=0.95,
        margin_min=0.03,
        view_photos=np.array([1, 2]),
        references=np.stack([crops[0], crops[0]]),
        reference_photos=np.array([1, 2]),
    )
    vectors = np.stack([axis(0, 384)])
    first = engine.score_evidence([0], vectors, crops, [watch])
    second = engine.score_evidence([0], vectors, crops, [watch])
    assert [s.object_id for s in first] == [1] and first[0].strong
    assert first == second
    phrase_runs = [call for call in scene.phrase_calls if "a watch" in call]
    assert len(phrase_runs) == 1  # The second frame used the kept vectors
    assert engine.evidence_regions(vectors, [MatchRegion((), True)], [watch]) == [0]
