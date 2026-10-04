"""Tests of the calibration math. Pure numpy, no model."""

import dataclasses

import numpy as np
import pytest

from ai.calibration import NOTE_LOOK_ALIKE, NOTE_PHOTOS_DIFFER, calibrate
from ai.errors import CalibrationError
from ai.settings import EnrollmentConfig, load_config

CONFIG: EnrollmentConfig = load_config().enrollment
DIMENSION: int = 8


def vector(*numbers: float) -> np.ndarray:
    """A unit vector. The first numbers are given, the rest is zero."""
    padded: np.ndarray = np.zeros(DIMENSION, dtype=np.float32)
    padded[: len(numbers)] = numbers
    return padded / np.linalg.norm(padded)


def stack(*vectors: np.ndarray) -> np.ndarray:
    return np.stack(vectors) if vectors else np.empty((0, DIMENSION), dtype=np.float32)


def expected_pairs(views: np.ndarray, photo_ids: list[int]) -> np.ndarray:
    """Slow reference: the similarities between views of different photos."""
    found: list[float] = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            if photo_ids[i] != photo_ids[j]:
                found.append(float(views[i] @ views[j]))
    return np.asarray(found)


def test_tau_is_mean_minus_a_part_of_the_std_of_the_similarities_between_photos() -> (
    None
):
    views: np.ndarray = stack(
        vector(1, 0.0), vector(1, 0.2), vector(1, 0.4), vector(1, 0.7)
    )
    photo_ids: list[int] = [1, 2, 3, 4]
    result = calibrate(views, photo_ids, stack(), CONFIG)
    pairs: np.ndarray = expected_pairs(views, photo_ids)
    assert result.own_mean == pytest.approx(pairs.mean(), abs=1e-6)
    assert result.own_std == pytest.approx(pairs.std(), abs=1e-6)
    assert result.tau == pytest.approx(
        pairs.mean() - CONFIG.tau_std_factor * pairs.std(), abs=1e-6
    )
    assert result.tau_strong == pytest.approx(
        min(result.tau + CONFIG.tau_strong_offset, 1.0), abs=1e-6
    )
    assert result.margin_min == CONFIG.margin_min
    assert (result.n_views, result.n_photos) == (4, 4)
    assert result.negative_max is None and result.note is None


def test_views_of_the_same_photo_are_not_compared_with_each_other() -> None:
    # Two views of photo 1 are the same picture. Their similarity of 1.0 must not count.
    views: np.ndarray = stack(vector(1, 0), vector(1, 0), vector(1, 1), vector(1, 1))
    photo_ids: list[int] = [1, 1, 2, 2]
    result = calibrate(views, photo_ids, stack(), CONFIG)
    cross: float = float(vector(1, 0) @ vector(1, 1))
    assert result.own_mean == pytest.approx(
        cross, abs=1e-6
    ) and result.own_std == pytest.approx(0, abs=1e-6)
    assert result.tau == pytest.approx(cross, abs=1e-6)


def test_the_negatives_raise_tau_to_the_nearest_look_alike_plus_the_margin() -> None:
    views: np.ndarray = stack(vector(1, 0.1), vector(1, 0.0), vector(1, 0.2))
    others: np.ndarray = stack(
        vector(0, 1, 0), vector(0.2, 0.9, 0)
    )  # Not close to the views
    plain = calibrate(views, [1, 2, 3], stack(), CONFIG)
    far = calibrate(views, [1, 2, 3], others, CONFIG)
    assert far.negative_max == pytest.approx(float((views @ others.T).max()), abs=1e-6)
    assert far.tau == pytest.approx(
        plain.tau, abs=1e-6
    )  # The negatives are far away: no change
    near_other: np.ndarray = stack(vector(1, 0.05, 0.1))  # Almost the same as the views
    near = calibrate(views, [1, 2, 3], near_other, CONFIG)
    assert near.negative_max is not None
    assert near.tau == pytest.approx(
        min(near.negative_max + CONFIG.margin_min, 1.0), abs=1e-6
    )
    assert near.tau > plain.tau


def test_a_look_alike_that_is_too_close_gives_a_note() -> None:
    views: np.ndarray = stack(vector(1, 0.3), vector(1, 0.35), vector(1, 0.4))
    identical: np.ndarray = stack(vector(1, 0.35))
    result = calibrate(views, [1, 2, 3], identical, CONFIG)
    assert result.note == NOTE_LOOK_ALIKE and result.tau > result.own_mean


def test_tau_and_the_strong_threshold_never_pass_one() -> None:
    views: np.ndarray = stack(vector(1, 0), vector(1, 0), vector(1, 0))
    result = calibrate(views, [1, 2, 3], stack(vector(1, 0)), CONFIG)
    assert result.tau == 1.0 and result.tau_strong == 1.0


def test_fewer_than_two_photos_are_refused() -> None:
    with pytest.raises(CalibrationError):
        calibrate(stack(vector(1, 0), vector(1, 1)), [1, 1], stack(), CONFIG)
    with pytest.raises(CalibrationError):
        calibrate(stack(), [], stack(), CONFIG)


@pytest.mark.parametrize(
    "bad",
    [
        np.full((2, DIMENSION), np.nan, dtype=np.float32),
        np.full((2, DIMENSION), 2.0, dtype=np.float32),
    ],
    ids=["not finite", "not unit length"],
)
def test_bad_vectors_are_refused(bad: np.ndarray) -> None:
    good: np.ndarray = stack(vector(1, 0), vector(1, 1))
    with pytest.raises(CalibrationError):
        calibrate(bad, [1, 2], stack(), CONFIG)
    with pytest.raises(CalibrationError):
        calibrate(good, [1, 2], bad, CONFIG)


def test_negatives_of_another_size_are_refused() -> None:
    with pytest.raises(CalibrationError):
        calibrate(
            stack(vector(1, 0), vector(1, 1)),
            [1, 2],
            np.ones((1, 3), dtype=np.float32) / np.sqrt(3),
            CONFIG,
        )


def test_the_parameters_are_checked() -> None:
    views: np.ndarray = stack(vector(1, 0), vector(1, 1))
    with pytest.raises(AssertionError):
        calibrate(views, [1], stack(), CONFIG)  # One photo id for two views
    with pytest.raises(AssertionError):
        calibrate(views.tolist(), [1, 2], stack(), CONFIG)  # type: ignore[arg-type]


def test_tau_is_never_below_the_floor() -> None:
    # Three photos that look different from each other: the formula alone gives a very low value.
    views: np.ndarray = stack(vector(1, 0), vector(0.3, 1), vector(-0.4, 1))
    no_floor = dataclasses.replace(CONFIG, tau_min=0.0)
    low = calibrate(views, [1, 2, 3], stack(), no_floor)
    floored = calibrate(views, [1, 2, 3], stack(), CONFIG)
    assert low.tau < CONFIG.tau_min  # The formula alone is below the floor
    assert floored.tau == CONFIG.tau_min
    assert floored.tau_strong == pytest.approx(
        CONFIG.tau_min + CONFIG.tau_strong_offset
    )


def test_the_floor_does_not_lower_a_higher_value() -> None:
    views: np.ndarray = stack(vector(1, 0.0), vector(1, 0.05), vector(1, 0.1))
    no_floor = dataclasses.replace(CONFIG, tau_min=0.0)
    assert calibrate(views, [1, 2, 3], stack(), CONFIG).tau == pytest.approx(
        calibrate(views, [1, 2, 3], stack(), no_floor).tau
    )


def test_photos_that_look_very_different_give_a_note() -> None:
    views: np.ndarray = stack(vector(1, 0), vector(0.3, 1), vector(-0.4, 1))
    result = calibrate(views, [1, 2, 3], stack(), CONFIG)
    assert result.note == NOTE_PHOTOS_DIFFER and result.own_mean < CONFIG.tau_min


def test_the_floor_alone_does_not_give_the_look_alike_note() -> None:
    # The old rule gave that note whenever tau was above the mean of the photos. The floor can do that too.
    views: np.ndarray = stack(vector(1, 0), vector(0.3, 1), vector(-0.4, 1))
    far: np.ndarray = stack(vector(0, 0, 1))
    result = calibrate(views, [1, 2, 3], far, CONFIG)
    assert result.tau > result.own_mean and result.note != NOTE_LOOK_ALIKE


def test_a_floor_of_zero_gives_the_old_result() -> None:
    views: np.ndarray = stack(vector(1, 0.1), vector(1, 0.0), vector(1, 0.2))
    plain = calibrate(
        views, [1, 2, 3], stack(), dataclasses.replace(CONFIG, tau_min=0.0)
    )
    pairs: np.ndarray = expected_pairs(views, [1, 2, 3])
    assert plain.tau == pytest.approx(
        pairs.mean() - CONFIG.tau_std_factor * pairs.std()
    )
