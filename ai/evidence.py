"""Evidence-based matching of regions with the enrolled objects (stage I4, mode `evidence`). Pure numpy, no model.

The mode `tau` accepts a region with one test: the DINOv2 similarity must reach the threshold of the object.
One test is a single point of failure. A threshold that is too high hides the object, and one that is too low
accepts look-alikes. This module combines independent signals into one score:

- `dino_agree`: the mean of the best DINOv2 cosine of the `top_photos` best reference photos.
  A real match looks like several photos of the object. A lucky match looks like one view only.
- `margin`: the best DINOv2 cosine to this object minus the best cosine to any other enrolled object.
- `visual`: the same agreement as `dino_agree`, with SigLIP vectors of the reference crops. SigLIP is a
  second model with other errors. DINOv2 alone confused objects that were photographed on the same table.
- `meaning`: the share of the words of the object (its category phrase and synonyms) in a SigLIP comparison
  with the words of the other objects and of common things ("a hand", "a table"). It tells if the region shows
  the right kind of thing. It cannot tell one book from another, so its positive part is capped: it can stop a
  match, but it cannot make one.

score = bias + w_meaning * min(logit(meaning), meaning_cap) + w_visual * (visual - visual_centre) / visual_scale
        + w_dino * (dino_agree - dino_centre) / dino_scale + w_margin * min(margin, margin_cap) / margin_scale

A region is a sighting if `score >= score_maybe`. It is strong if `score >= score_sure`.
Two rules can only refuse, never accept: a region with `quality_ok = False` is not scored, and a region whose
`margin` is below the `margin_min` of the object is refused. The second rule keeps the protection against
look-alikes: a twin that is as close as the object blocks the match, as in the mode `tau`.
The weights and the bars are in `[matching]` of `config.toml`. The benchmark is `benchmarks/matching/evidence.py`.
"""

import logging
from collections.abc import Sequence

import numpy as np

from ai.errors import MatchingError
from ai.settings import MatchingConfig
from ai.types import MatchObject, MatchRegion, Sighting

logger: logging.Logger = logging.getLogger(__name__)

# A logit of the meaning is cut to this range, so one extreme value cannot decide alone.
LOGIT_LIMIT: float = 5.0
NORM_TOLERANCE: float = 1e-3


def _check_unit(vectors: np.ndarray, name: str) -> None:
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise MatchingError(f"{name} must be a 2-D array")
    if vectors.shape[0] == 0:
        return
    if not np.isfinite(vectors).all():
        raise MatchingError(f"{name} contain a value that is not finite")
    if np.abs(np.linalg.norm(vectors, axis=1) - 1.0).max() > NORM_TOLERANCE:
        raise MatchingError(f"{name} must have unit length")


def _check_object(obj: MatchObject, dimension: int) -> None:
    _check_unit(obj.views, f"views of object {obj.object_id}")
    if obj.views.shape[0] == 0 or obj.views.shape[1] != dimension:
        raise MatchingError(f"Object {obj.object_id} has no usable views")
    if obj.view_photos is None or len(obj.view_photos) != obj.views.shape[0]:
        raise MatchingError(f"Object {obj.object_id} needs a photo id for each view")
    if obj.references is None or obj.reference_photos is None:
        raise MatchingError(f"Object {obj.object_id} has no reference vectors")
    _check_unit(obj.references, f"references of object {obj.object_id}")
    if obj.references.shape[0] == 0:
        raise MatchingError(f"Object {obj.object_id} has no reference vectors")
    if len(obj.reference_photos) != obj.references.shape[0]:
        raise MatchingError(
            f"Object {obj.object_id} needs a photo id for each reference"
        )


def photo_agreement(scores: np.ndarray, photo_ids: np.ndarray, top: int) -> float:
    """Return the mean of the best score of the `top` best photos.

    `scores` has one value for each view and `photo_ids` names the photo of each view.
    """
    assert (
        scores.ndim == 1 and scores.shape == photo_ids.shape
    ), "one photo id per score"
    assert top > 0, "top must be positive"
    best: list[float] = sorted(
        (float(scores[photo_ids == photo].max()) for photo in np.unique(photo_ids)),
        reverse=True,
    )
    return float(np.mean(best[:top]))


def _logit(share: float) -> float:
    clipped: float = min(max(share, 1e-6), 1.0 - 1e-6)
    return float(np.clip(np.log(clipped / (1.0 - clipped)), -LOGIT_LIMIT, LOGIT_LIMIT))


def evidence_score(
    dino_agree: float,
    margin: float,
    visual: float,
    meaning: float,
    config: MatchingConfig,
) -> float:
    """Return the score of one region for one object. See the module text for the formula."""
    return (
        config.bias
        + config.weight_meaning * min(_logit(meaning), config.meaning_cap)
        + config.weight_visual * (visual - config.visual_centre) / config.visual_scale
        + config.weight_dino * (dino_agree - config.dino_centre) / config.dino_scale
        + config.weight_margin * min(margin, config.margin_cap) / config.margin_scale
    )


def class_shares(class_logits: np.ndarray) -> np.ndarray:
    """Turn the logits of each region over the classes into shares that add up to 1 (a softmax by row)."""
    assert class_logits.ndim == 2, "class_logits must be a 2-D array"
    shifted: np.ndarray = class_logits - class_logits.max(axis=1, keepdims=True)
    exp: np.ndarray = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def best_similarity(vectors: np.ndarray, objects: Sequence[MatchObject]) -> np.ndarray:
    """Return the best DINOv2 view cosine of each region for each object, shape (regions, objects)."""
    return np.stack([(vectors @ obj.views.T).max(axis=1) for obj in objects], axis=1)


def select_regions(
    vectors: np.ndarray,
    regions: Sequence[MatchRegion],
    objects: Sequence[MatchObject],
    config: MatchingConfig,
) -> list[int]:
    """Return the indices of the regions to score, the most similar first.

    A region is a candidate if its quality is good and its DINOv2 similarity to a tracked object is at least
    `candidate_min`. At most `max_regions` regions are kept, because each one costs a SigLIP run.
    """
    assert len(regions) == len(vectors), "one vector per region"
    _check_unit(vectors, "vectors")
    tracked: list[MatchObject] = [o for o in objects if o.tracked]
    if not len(regions) or not tracked:
        return []
    similarity: np.ndarray = best_similarity(vectors, tracked).max(axis=1)
    chosen: list[int] = [
        i
        for i in np.argsort(-similarity).tolist()
        if regions[i].quality_ok and similarity[i] >= config.candidate_min
    ]
    return chosen[: config.max_regions]


def score_regions(
    region_indices: Sequence[int],
    vectors: np.ndarray,
    crop_vectors: np.ndarray,
    class_logits: np.ndarray,
    objects: Sequence[MatchObject],
    config: MatchingConfig,
) -> list[Sighting]:
    """Score the chosen regions of one frame. Return at most one sighting for each tracked object.

    `crop_vectors` has one SigLIP vector for each index in `region_indices`. `class_logits` has one row for
    each of them and one column for each class: first the objects, in the order of `objects`, then the contrast
    phrases. A class has the logit of its best phrase. Raise `MatchingError` if the data is not usable.
    """
    assert len(crop_vectors) == len(region_indices), "one crop vector per region"
    assert len(class_logits) == len(region_indices), "one logit row per region"
    if not len(region_indices) or not objects:
        return []
    _check_unit(vectors, "vectors")
    _check_unit(crop_vectors, "crop vectors")
    if class_logits.shape[1] < len(objects) or not np.isfinite(class_logits).all():
        raise MatchingError("class_logits need a finite column for each object")
    ids: list[int] = [o.object_id for o in objects]
    assert len(set(ids)) == len(ids), "object ids must be unique"
    for obj in objects:
        _check_object(obj, vectors.shape[1])

    chosen: np.ndarray = vectors[list(region_indices)]
    similarity: np.ndarray = best_similarity(chosen, objects)
    shares: np.ndarray = class_shares(class_logits)
    best: dict[int, Sighting] = {}
    for row, region_index in enumerate(region_indices):
        for j, obj in enumerate(objects):
            if not obj.tracked:
                continue
            rivals: np.ndarray = np.delete(similarity[row], j)
            margin: float = float(
                similarity[row, j] - (rivals.max() if rivals.size else 0.0)
            )
            if margin < obj.margin_min:
                continue  # Another enrolled object is as close: a look-alike blocks the match
            dino_agree: float = photo_agreement(
                obj.views @ chosen[row], obj.view_photos, config.top_photos
            )
            visual: float = photo_agreement(
                obj.references @ crop_vectors[row],
                obj.reference_photos,
                config.top_photos,
            )
            score: float = evidence_score(
                dino_agree, margin, visual, float(shares[row, j]), config
            )
            if score < config.score_maybe:
                continue
            current: Sighting | None = best.get(obj.object_id)
            if current is None or (current.score is not None and score > current.score):
                best[obj.object_id] = Sighting(
                    obj.object_id,
                    int(region_index),
                    float(similarity[row, j]),
                    margin,
                    score >= config.score_sure,
                    score,
                )
    logger.info(
        "Evidence matching: %d region(s) scored, %d object(s), %d sighting(s), %d strong",
        len(region_indices),
        len(objects),
        len(best),
        sum(s.strong for s in best.values()),
    )
    return [best[object_id] for object_id in sorted(best)]
