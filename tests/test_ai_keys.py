"""Tests of the matching with two keys (ai/keys.py) and of its engine methods. No real model."""

import numpy as np
import pytest
from PIL import Image

from ai.errors import MatchingError
from ai.keys import NONE, STRONG, WEAK, KeyBars, key_level, match_keys, word_score
from ai.settings import load_config
from ai.types import Box, MatchObject, MatchRegion, OpenDetection
from tests.ai_support import FakeEmbedder, make_engine

BARS: KeyBars = KeyBars(
    top_photos=2,
    sig_strong=0.77,
    dino_strong=0.42,
    sig_weak=0.72,
    dino_weak=0.40,
    word_vote=0.30,
)
SIG: int = 6
DINO: int = 5
GOOD: MatchRegion = MatchRegion(("a watch",), True, 0.0)


def with_cosine(cosine: float, size: int, axis: int = 0) -> np.ndarray:
    """A unit vector whose cosine with the unit vector on `axis` is `cosine`."""
    vector: np.ndarray = np.zeros(size, dtype=np.float32)
    vector[axis] = cosine
    vector[size - 1 if axis != size - 1 else 0] = np.sqrt(max(0.0, 1.0 - cosine**2))
    return vector


def axis(index: int, size: int) -> np.ndarray:
    vector: np.ndarray = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def make_object(
    object_id: int,
    sig_axis: int,
    dino_axis: int,
    tracked: bool = True,
    phrases: tuple[str, ...] = ("a watch",),
    photos: int = 2,
) -> MatchObject:
    """An object whose reference photos all point at one axis in each space."""
    return MatchObject(
        object_id=object_id,
        tracked=tracked,
        phrases=phrases,
        views=np.stack([axis(0, 8)] * photos),
        tau=0.9,
        tau_strong=0.95,
        margin_min=0.03,
        view_photos=np.arange(1, photos + 1),
        references=np.stack([axis(sig_axis, SIG)] * photos),
        reference_photos=np.arange(1, photos + 1),
        key_references=np.stack([axis(dino_axis, DINO)] * photos),
    )


# ---- The rule for one box ----


@pytest.mark.parametrize(
    ("s", "d", "word", "expected"),
    [
        (0.80, 0.45, 0.0, STRONG),  # Both keys at the strong bars
        (0.80, 0.41, 0.0, WEAK),  # DINOv2 at the weak bar only
        (0.74, 0.45, 0.0, WEAK),  # SigLIP at the weak bar only
        (0.74, 0.10, 0.5, WEAK),  # A word vote replaces the DINOv2 vote
        (0.74, 0.10, 0.2, NONE),  # A weak word is no vote
        (
            0.70,
            0.90,
            0.9,
            NONE,
        ),  # Words and DINOv2 never make a sighting without SigLIP
    ],
)
def test_key_level_follows_the_rule(
    s: float, d: float, word: float, expected: int
) -> None:
    assert key_level(s, d, word, GOOD, BARS) == expected


def test_a_bad_crop_is_weak_at_most() -> None:
    blurred: MatchRegion = MatchRegion(("a watch",), False, 0.0)
    assert key_level(0.90, 0.90, 0.0, blurred, BARS) == WEAK
    assert key_level(0.60, 0.90, 0.0, blurred, BARS) == NONE


def test_a_depicted_box_is_never_a_sighting() -> None:
    screen: MatchRegion = MatchRegion(("a watch",), False, 0.9, depicted=True)
    assert key_level(0.99, 0.99, 0.9, screen, BARS) == NONE


def test_word_score_needs_a_phrase_of_the_object() -> None:
    obj = make_object(1, 0, 0)
    assert word_score(MatchRegion(("a watch",), True, 0.6), obj) == pytest.approx(0.6)
    assert word_score(MatchRegion(("a book",), True, 0.6), obj) == 0.0
    assert word_score(MatchRegion((), True, 0.6), obj) == 0.0


def test_bars_check_their_order() -> None:
    with pytest.raises(AssertionError):
        KeyBars(2, 0.70, 0.42, 0.75, 0.40, 0.30)


# ---- A frame ----


def test_match_keys_reports_the_keys() -> None:
    watch = make_object(1, sig_axis=0, dino_axis=0)
    sig = np.stack([with_cosine(0.80, SIG)])
    dino = np.stack([with_cosine(0.45, DINO)])
    found = match_keys(sig, dino, [GOOD], [watch], BARS)
    assert len(found) == 1
    assert found[0].strong is True  # A plain bool, so the db and JSON take it
    assert found[0].similarity == pytest.approx(0.80, abs=1e-5)
    assert found[0].score == pytest.approx(0.45, abs=1e-5)
    assert found[0].margin == pytest.approx(0.80, abs=1e-5)  # No other object


def test_one_box_is_one_object() -> None:
    """Two objects pass on the same box. The object with the higher SigLIP key keeps it."""
    watch = make_object(1, sig_axis=0, dino_axis=0)
    bracelet = make_object(2, sig_axis=1, dino_axis=0)
    vector: np.ndarray = np.zeros(SIG, dtype=np.float32)
    vector[0], vector[1] = (
        0.62,
        0.78,
    )  # cosine 0.62 with the watch, 0.78 with the bracelet
    vector[2] = np.sqrt(1 - 0.62**2 - 0.78**2)
    found = match_keys(
        np.stack([vector]),
        np.stack([axis(0, DINO)]),
        [MatchRegion((), True, 0.0)],
        [watch, bracelet],
        BARS,
    )
    assert [f.object_id for f in found] == [2]
    assert found[0].margin == pytest.approx(0.78 - 0.62, abs=1e-5)


def test_a_look_alike_takes_the_box_and_hides_it() -> None:
    watch = make_object(1, sig_axis=0, dino_axis=0)
    twin = make_object(2, sig_axis=1, dino_axis=0, tracked=False)
    vector: np.ndarray = np.zeros(SIG, dtype=np.float32)
    vector[0], vector[1] = 0.55, 0.83
    vector[2] = np.sqrt(1 - 0.55**2 - 0.83**2)
    found = match_keys(
        np.stack([vector]),
        np.stack([axis(0, DINO)]),
        [GOOD],
        [watch, twin],
        BARS,
    )
    assert found == []


def test_one_sighting_per_object_prefers_strong() -> None:
    watch = make_object(1, sig_axis=0, dino_axis=0)
    sig = np.stack([with_cosine(0.85, SIG), with_cosine(0.80, SIG)])
    dino = np.stack([with_cosine(0.10, DINO), with_cosine(0.50, DINO)])
    regions = [MatchRegion(("a watch",), True, 0.9), GOOD]
    found = match_keys(sig, dino, regions, [watch], BARS)
    assert len(found) == 1
    assert found[0].region_index == 1 and found[0].strong


def test_match_keys_rejects_an_object_without_key_vectors() -> None:
    watch = make_object(1, 0, 0)
    broken = MatchObject(
        object_id=1,
        tracked=True,
        phrases=watch.phrases,
        views=watch.views,
        tau=0.9,
        tau_strong=0.95,
        margin_min=0.03,
        view_photos=watch.view_photos,
        references=watch.references,
        reference_photos=watch.reference_photos,
    )
    with pytest.raises(MatchingError):
        match_keys(
            np.stack([axis(0, SIG)]),
            np.stack([axis(0, DINO)]),
            [GOOD],
            [broken],
            BARS,
        )


def test_match_keys_rejects_vectors_that_are_not_unit() -> None:
    watch = make_object(1, 0, 0)
    with pytest.raises(MatchingError):
        match_keys(
            np.ones((1, SIG), dtype=np.float32),
            np.stack([axis(0, DINO)]),
            [GOOD],
            [watch],
            BARS,
        )


def test_no_region_gives_no_sighting() -> None:
    watch = make_object(1, 0, 0)
    empty_sig = np.empty((0, SIG), dtype=np.float32)
    empty_dino = np.empty((0, DINO), dtype=np.float32)
    assert match_keys(empty_sig, empty_dino, [], [watch], BARS) == []


# ---- The engine ----


def test_the_mode_keys_needs_the_scene_model() -> None:
    assert load_config().matching.mode == "keys"  # The shipped default
    engine = make_engine(matching_mode="keys")
    assert engine.matching_mode == "keys"
    engine._scene = None  # The scene model is switched off
    assert engine.matching_mode == "tau"


def test_embed_key_crops_uses_the_crop_size_of_the_config() -> None:
    embedder = FakeEmbedder()
    engine = make_engine(embedder=embedder)
    image = Image.new("RGB", (200, 100))
    vectors = engine.embed_key_crops(image, [Box(0, 0, 50, 50)])
    assert vectors.shape == (1, embedder.dimension)
    assert embedder.cls_calls == [
        ([Box(0, 0, 50, 50)], load_config().matching.key_crop_size)
    ]


def test_broad_candidates_have_no_phrase_and_pass_the_crop_gate() -> None:
    engine = make_engine()
    image = Image.new("RGB", (400, 300), (120, 120, 120))
    found = engine.broad_candidates(
        image,
        [
            OpenDetection(Box(100, 100, 200, 200), "handcuff", 0.8),
            OpenDetection(Box(0, 0, 60, 60), "cup", 0.5),  # Touches the edge
        ],
    )
    assert [c.phrases for c in found] == [(), ()]
    assert [c.det_score for c in found] == [0.8, 0.5]
    assert not found[0].in_hand and not found[0].depicted
    assert not found[1].quality_ok


def test_engine_match_keys_uses_the_bars_of_the_config() -> None:
    engine = make_engine(matching_mode="keys")
    watch = make_object(1, 0, 0)
    cfg = load_config().matching
    sig = np.stack([with_cosine(cfg.sig_strong + 0.01, SIG)])
    dino = np.stack([with_cosine(cfg.dino_strong + 0.01, DINO)])
    found = engine.match_keys(sig, dino, [GOOD], [watch])
    assert len(found) == 1 and found[0].strong


# ---- The boxes that the carer refused ----


def test_a_box_closer_to_a_refused_box_than_to_the_object_is_refused() -> None:
    import dataclasses

    watch = make_object(1, sig_axis=0, dino_axis=0)
    box_sig = np.stack(
        [with_cosine(0.80, SIG)]
    )  # Passes for the watch without a negative
    dino = np.stack([with_cosine(0.45, DINO)])
    assert match_keys(box_sig, dino, [GOOD], [watch], BARS)
    no_crop = box_sig.copy()  # The carer said "no" to this very box
    refused = dataclasses.replace(watch, negative_crops=no_crop)
    assert match_keys(box_sig, dino, [GOOD], [refused], BARS) == []


def test_a_far_refused_box_changes_nothing() -> None:
    import dataclasses

    watch = make_object(1, sig_axis=0, dino_axis=0)
    box_sig = np.stack([with_cosine(0.80, SIG)])
    dino = np.stack([with_cosine(0.45, DINO)])
    far = dataclasses.replace(
        watch,
        negative_crops=np.stack([axis(2, SIG)]),
        negative_keys=np.stack([axis(2, DINO)]),
    )
    assert len(match_keys(box_sig, dino, [GOOD], [far], BARS)) == 1


def test_negatives_of_another_size_are_refused() -> None:
    import dataclasses

    watch = dataclasses.replace(
        make_object(1, 0, 0), negative_crops=np.stack([axis(0, SIG + 1)])
    )
    with pytest.raises(MatchingError):
        match_keys(
            np.stack([axis(0, SIG)]), np.stack([axis(0, DINO)]), [GOOD], [watch], BARS
        )
