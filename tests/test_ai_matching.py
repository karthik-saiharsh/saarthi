"""Tests of the matching math (stage I4). Pure numpy, no model."""

import numpy as np
import pytest

from ai.errors import MatchingError
from ai.matching import match_regions
from ai.types import MatchObject, MatchRegion, Sighting

DIMENSION: int = 8
GENERIC: str = "an object"
MARGIN_STRONG: float = 0.08


def vector(*numbers: float) -> np.ndarray:
    """A unit vector. The first numbers are given, the rest is zero."""
    padded: np.ndarray = np.zeros(DIMENSION, dtype=np.float32)
    padded[: len(numbers)] = numbers
    return padded / np.linalg.norm(padded)


def stack(*vectors: np.ndarray) -> np.ndarray:
    return np.stack(vectors)


def make_object(
    object_id: int,
    *views: np.ndarray,
    phrases: tuple[str, ...] = ("a bottle",),
    tracked: bool = True,
    tau: float = 0.8,
    tau_strong: float = 0.9,
    margin_min: float = 0.03,
) -> MatchObject:
    return MatchObject(
        object_id, tracked, phrases, stack(*views), tau, tau_strong, margin_min
    )


def region(*phrases: str, quality_ok: bool = True) -> MatchRegion:
    return MatchRegion(tuple(phrases), quality_ok)


def run(
    vectors: list[np.ndarray],
    regions: list[MatchRegion],
    objects: list[MatchObject],
) -> list[Sighting]:
    return match_regions(stack(*vectors), regions, objects, GENERIC, MARGIN_STRONG)


def test_a_region_close_to_the_object_is_a_sighting() -> None:
    bottle = make_object(1, vector(1, 0), vector(1, 0.1))
    result = run([vector(1, 0.05)], [region("a bottle")], [bottle])
    assert len(result) == 1
    sighting: Sighting = result[0]
    assert (sighting.object_id, sighting.region_index) == (1, 0)
    assert sighting.similarity == pytest.approx(1.0, abs=0.01)
    # Without another object, the margin is the similarity itself.
    assert sighting.margin == pytest.approx(sighting.similarity)
    assert sighting.strong


def test_the_similarity_is_the_best_view_not_the_mean() -> None:
    bottle = make_object(1, vector(1, 0), vector(0, 1))
    result = run([vector(1, 0.02)], [region("a bottle")], [bottle])
    assert len(result) == 1 and result[0].similarity > 0.99


def test_a_region_below_tau_is_not_a_sighting() -> None:
    bottle = make_object(1, vector(1, 0))
    far: np.ndarray = vector(1, 1)  # Cosine 0.71
    assert run([far], [region("a bottle")], [bottle]) == []


def test_a_sighting_between_tau_and_tau_strong_is_weak() -> None:
    bottle = make_object(1, vector(1, 0), tau=0.8, tau_strong=0.95)
    between: np.ndarray = vector(1, 0.5)  # Cosine 0.894
    result = run([between], [region("a bottle")], [bottle])
    assert len(result) == 1 and not result[0].strong


def test_a_high_similarity_with_a_small_margin_is_weak() -> None:
    # Two objects that the region matches almost equally. The margin is 0.06.
    first = make_object(1, vector(1, 0), margin_min=0.03)
    second = make_object(2, vector(1, 0.35), phrases=("a bottle",), margin_min=0.03)
    point: np.ndarray = vector(1, 0.05)
    result = run([point], [region("a bottle")], [first, second])
    by_object = {s.object_id: s for s in result}
    assert 1 in by_object and 0.03 <= by_object[1].margin < MARGIN_STRONG
    assert not by_object[1].strong


def test_a_look_alike_that_is_as_close_blocks_the_owner_object() -> None:
    owner = make_object(1, vector(1, 0))
    twin = make_object(2, vector(1, 0.02), tracked=False)
    assert run([vector(1, 0.01)], [region("a bottle")], [owner, twin]) == []


def test_a_look_alike_is_never_reported() -> None:
    owner = make_object(1, vector(1, 0))
    twin = make_object(2, vector(0, 1), tracked=False)
    result = run([vector(0, 1)], [region("a bottle")], [owner, twin])
    assert result == []


def test_the_margin_uses_the_nearest_other_object() -> None:
    owner = make_object(1, vector(1, 0))
    twin = make_object(2, vector(1, 0.3), tracked=False)  # Cosine 0.958 to the owner
    result = run([vector(1, 0)], [region("a bottle")], [owner, twin])
    assert len(result) == 1
    assert result[0].margin == pytest.approx(1.0 - 0.958, abs=0.001)


def test_the_phrase_of_the_region_chooses_the_objects() -> None:
    bottle = make_object(1, vector(1, 0), phrases=("a bottle", "a flask"))
    keys = make_object(2, vector(0, 1), phrases=("a set of keys",))
    # The vectors match the bottle, but the phrase names keys only.
    assert run([vector(1, 0)], [region("a set of keys")], [bottle, keys]) == []
    result = run([vector(1, 0)], [region("a flask")], [bottle, keys])
    assert [s.object_id for s in result] == [1]


def test_a_generic_or_unlabelled_region_may_show_any_object() -> None:
    bottle = make_object(1, vector(1, 0))
    keys = make_object(2, vector(0, 1), phrases=("a set of keys",))
    result = run(
        [vector(1, 0), vector(0, 1)],
        [region(GENERIC), region()],
        [bottle, keys],
    )
    assert {(s.object_id, s.region_index) for s in result} == {(1, 0), (2, 1)}


def test_a_region_that_is_not_good_is_skipped() -> None:
    bottle = make_object(1, vector(1, 0))
    assert run([vector(1, 0)], [region("a bottle", quality_ok=False)], [bottle]) == []


def test_a_skipped_region_does_not_take_the_place_of_a_good_region() -> None:
    # The first region is the better match, but it is not good. The second one wins.
    bottle = make_object(1, vector(1, 0))
    result = run(
        [vector(1, 0), vector(1, 0.01)],
        [region("a bottle", quality_ok=False), region("a bottle")],
        [bottle],
    )
    assert [s.region_index for s in result] == [1]


def test_each_object_keeps_its_best_region() -> None:
    bottle = make_object(1, vector(1, 0))
    result = run(
        [vector(1, 0.3), vector(1, 0.02), vector(1, 0.2)],
        [region("a bottle")] * 3,
        [bottle],
    )
    assert len(result) == 1 and result[0].region_index == 1


def test_two_objects_can_each_have_a_sighting_sorted_by_id() -> None:
    bottle = make_object(5, vector(1, 0))
    keys = make_object(2, vector(0, 1), phrases=("a set of keys",))
    result = run(
        [vector(1, 0), vector(0, 1)],
        [region("a bottle"), region("a set of keys")],
        [bottle, keys],
    )
    assert [(s.object_id, s.region_index) for s in result] == [(2, 1), (5, 0)]


def test_no_regions_or_no_objects_give_no_sightings() -> None:
    bottle = make_object(1, vector(1, 0))
    empty: np.ndarray = np.empty((0, DIMENSION), dtype=np.float32)
    assert match_regions(empty, [], [bottle], GENERIC, MARGIN_STRONG) == []
    assert run([vector(1, 0)], [region("a bottle")], []) == []


def test_the_margin_limit_of_the_object_is_used() -> None:
    first = make_object(1, vector(1, 0), margin_min=0.2)
    second = make_object(2, vector(1, 0.35), margin_min=0.01)
    result = run([vector(1, 0.05)], [region("a bottle")], [first, second])
    # Object 1 has the best similarity, but its margin (0.04) is below its own limit (0.2).
    assert [s.object_id for s in result] == []


def test_a_vector_that_is_not_unit_length_is_refused() -> None:
    bottle = make_object(1, vector(1, 0))
    bad: np.ndarray = np.zeros((1, DIMENSION), dtype=np.float32)
    bad[0, 0] = 2.0
    with pytest.raises(MatchingError, match="unit length"):
        match_regions(bad, [region("a bottle")], [bottle], GENERIC, MARGIN_STRONG)


def test_a_vector_that_is_not_finite_is_refused() -> None:
    bottle = make_object(1, vector(1, 0))
    bad: np.ndarray = np.full((1, DIMENSION), np.nan, dtype=np.float32)
    with pytest.raises(MatchingError, match="not finite"):
        match_regions(bad, [region("a bottle")], [bottle], GENERIC, MARGIN_STRONG)


def test_views_of_another_size_are_refused() -> None:
    wrong = MatchObject(
        1, True, ("a bottle",), np.ones((1, 4), dtype=np.float32) / 2, 0.8, 0.9, 0.03
    )
    with pytest.raises(MatchingError, match="another size"):
        run([vector(1, 0)], [region("a bottle")], [wrong])


def test_an_object_without_views_is_refused() -> None:
    empty = MatchObject(
        1,
        True,
        ("a bottle",),
        np.empty((0, DIMENSION), dtype=np.float32),
        0.8,
        0.9,
        0.03,
    )
    with pytest.raises(MatchingError, match="no views"):
        run([vector(1, 0)], [region("a bottle")], [empty])
