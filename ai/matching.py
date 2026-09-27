"""Matching of region vectors with the enrolled objects (stage I4). Pure numpy, no model.

For each region that is good enough, the module finds the objects that the
region may show and scores them:

- `similarity` is the highest cosine between the region and a view of the object.
- `margin` is the similarity minus the highest similarity to any *other* object.
  The other objects are the negatives, as in the calibration (stage E4). A
  look-alike (`tracked = False`) takes part as a negative, but it is never reported.
- A sighting is accepted when `similarity >= tau` and `margin >= margin_min`.
  It is strong when `similarity >= tau_strong` and `margin >= margin_strong`.

Each object keeps its best region only (at most one sighting per object).
"""

import logging
from collections.abc import Sequence

import numpy as np

from ai.errors import MatchingError
from ai.types import MatchObject, MatchRegion, Sighting

logger: logging.Logger = logging.getLogger(__name__)

# How far the length of a vector may be from 1.0.
NORM_TOLERANCE: float = 1e-3

# The margin of an object that has no other object to compare with.
NO_RIVAL_SIMILARITY: float = 0.0


def _check_vectors(vectors: np.ndarray, name: str) -> None:
    assert (
        isinstance(vectors, np.ndarray) and vectors.ndim == 2
    ), f"{name} must be a 2-D array"
    if vectors.shape[0] == 0:
        return
    if not np.isfinite(vectors).all():
        raise MatchingError(f"{name} contain a value that is not finite")
    if np.abs(np.linalg.norm(vectors, axis=1) - 1.0).max() > NORM_TOLERANCE:
        raise MatchingError(f"{name} must have unit length")


def _may_show(region: MatchRegion, obj: MatchObject, generic_phrase: str) -> bool:
    """Return True if the phrases of the region allow the object.

    A region without a phrase, and a region from the generic pass, may show any object.
    """
    if not region.phrases or generic_phrase in region.phrases:
        return True
    return any(phrase in obj.phrases for phrase in region.phrases)


def match_regions(
    vectors: np.ndarray,
    regions: Sequence[MatchRegion],
    objects: Sequence[MatchObject],
    generic_phrase: str,
    margin_strong: float,
) -> list[Sighting]:
    """Return the sightings of the enrolled objects in the regions of one frame.

    `vectors` has one row for each region. The result has at most one sighting
    for each object, the one with the highest similarity, sorted by object id.
    Raise `MatchingError` if a vector or a view is not usable.
    """
    assert isinstance(regions, Sequence), "regions must be a sequence"
    assert len(regions) == len(vectors), "one vector per region"
    assert isinstance(generic_phrase, str) and generic_phrase, "generic phrase needed"
    assert margin_strong >= 0.0, "margin_strong must not be negative"
    _check_vectors(vectors, "vectors")
    if not len(regions) or not objects:
        return []
    ids: list[int] = [obj.object_id for obj in objects]
    assert len(set(ids)) == len(ids), "object ids must be unique"

    dimension: int = vectors.shape[1]
    for obj in objects:
        _check_vectors(obj.views, f"views of object {obj.object_id}")
        if obj.views.shape[0] == 0:
            raise MatchingError(f"Object {obj.object_id} has no views")
        if obj.views.shape[1] != dimension:
            raise MatchingError(f"Object {obj.object_id} has views of another size")

    # similarity[i, j]: the best view of object j for region i.
    similarity: np.ndarray = np.stack(
        [(vectors @ obj.views.T).max(axis=1) for obj in objects], axis=1
    )

    best: dict[int, Sighting] = {}
    for i, region in enumerate(regions):
        if not region.quality_ok:
            continue
        for j, obj in enumerate(objects):
            if not obj.tracked or not _may_show(region, obj, generic_phrase):
                continue
            rivals: np.ndarray = np.delete(similarity[i], j)
            rival: float = float(rivals.max()) if rivals.size else NO_RIVAL_SIMILARITY
            sim: float = float(similarity[i, j])
            margin: float = sim - rival
            if sim < obj.tau or margin < obj.margin_min:
                continue
            strong: bool = sim >= obj.tau_strong and margin >= margin_strong
            current: Sighting | None = best.get(obj.object_id)
            if current is None or sim > current.similarity:
                best[obj.object_id] = Sighting(obj.object_id, i, sim, margin, strong)

    logger.info(
        "Matching: %d region(s), %d object(s), %d sighting(s), %d strong",
        len(regions),
        len(objects),
        len(best),
        sum(s.strong for s in best.values()),
    )
    return [best[object_id] for object_id in sorted(best)]
