"""Calibration of an enrolled object (stage E4). Pure numpy, no model, no database.

The inputs are unit-length vectors. The similarity of two vectors is their
inner product (the cosine).

- `own`: the similarities between views of **different photos** of the object.
  Views of one photo are almost the same picture, so they are left out.
- `tau = mean(own) - tau_std_factor * std(own)`. It is at least `tau_min`: with few photos, the
  formula alone can give a very low value (see `benchmarks/calibration`).
- If other objects are enrolled, their views are the negatives. The highest
  similarity between an own view and a negative is `negative_max`.
  `tau` is at least `negative_max + margin_min`.
- `tau_strong = tau + tau_strong_offset`.
"""

import numpy as np

from ai.errors import CalibrationError
from ai.settings import EnrollmentConfig
from ai.types import Calibration

# How far the length of a vector may be from 1.0.
NORM_TOLERANCE: float = 1e-3

NOTE_PHOTOS_DIFFER: str = (
    "The photos of this object look very different from each other. Matches may be missed. "
    "Add clearer photos of the object."
)

NOTE_ONE_PLACE: str = (
    "All photos of this object show the same place. The computer can mix up the object and the place. "
    "Add photos of the object in the other places where it is used."
)

NOTE_LOOK_ALIKE: str = (
    "Another enrolled object looks very similar. Matches for this object may be missed or confused."
)


def _check_unit_vectors(vectors: np.ndarray, name: str) -> None:
    assert (
        isinstance(vectors, np.ndarray) and vectors.ndim == 2
    ), f"{name} must be a 2-D array"
    if vectors.shape[0] and not np.isfinite(vectors).all():
        raise CalibrationError(f"{name} contain a value that is not finite")
    if (
        vectors.shape[0]
        and np.abs(np.linalg.norm(vectors, axis=1) - 1.0).max() > NORM_TOLERANCE
    ):
        raise CalibrationError(f"{name} must have unit length")


def calibrate(
    views: np.ndarray,
    photo_ids: list[int],
    others: np.ndarray,
    config: EnrollmentConfig,
) -> Calibration:
    """Calculate the thresholds of one object.

    `views` has one row for each view. `photo_ids` has the photo of each row.
    `others` has the views of all other objects, or no rows.
    Raise `CalibrationError` if the views come from fewer than 2 photos.
    """
    assert isinstance(photo_ids, list) and len(photo_ids) == len(
        views
    ), "one photo id per view"
    assert isinstance(config, EnrollmentConfig), "config must be an EnrollmentConfig"
    _check_unit_vectors(views, "views")
    _check_unit_vectors(others, "others")
    if others.shape[0] and others.shape[1] != views.shape[1]:
        raise CalibrationError("The views of the other objects have another size")

    photos: np.ndarray = np.asarray(photo_ids)
    photo_count: int = len(set(photo_ids))
    if photo_count < 2:
        raise CalibrationError("The calibration needs views of at least 2 photos")

    similarities: np.ndarray = views @ views.T
    upper: tuple[np.ndarray, np.ndarray] = np.triu_indices(len(views), k=1)
    different_photo: np.ndarray = photos[upper[0]] != photos[upper[1]]
    own: np.ndarray = similarities[upper][different_photo]
    own_mean: float = float(own.mean())
    own_std: float = float(own.std())

    tau: float = max(own_mean - config.tau_std_factor * own_std, config.tau_min)
    negative_max: float | None = None
    if others.shape[0]:
        negative_max = float((views @ others.T).max())
        tau = max(tau, negative_max + config.margin_min)
    note: str | None = None
    if negative_max is not None and negative_max + config.margin_min > own_mean:
        note = NOTE_LOOK_ALIKE
    elif own_mean < config.tau_min:
        note = NOTE_PHOTOS_DIFFER

    tau = min(tau, 1.0)
    return Calibration(
        tau=tau,
        tau_strong=min(tau + config.tau_strong_offset, 1.0),
        margin_min=config.margin_min,
        own_mean=own_mean,
        own_std=own_std,
        negative_max=negative_max,
        n_views=len(views),
        n_photos=photo_count,
        note=note,
    )


def one_place(place_vectors: np.ndarray, same_min: float) -> bool:
    """True if every two photos have a background with a cosine of `same_min` or more (plan part D1).

    `place_vectors` has one unit-length row for each confirmed photo. Fewer than 2 photos are never one place.
    """
    assert (
        isinstance(place_vectors, np.ndarray) and place_vectors.ndim == 2
    ), "2-D array"
    if place_vectors.shape[0] < 2:
        return False
    similarity: np.ndarray = place_vectors @ place_vectors.T
    others: np.ndarray = similarity[~np.eye(len(place_vectors), dtype=bool)]
    return bool(others.min() >= same_min)
