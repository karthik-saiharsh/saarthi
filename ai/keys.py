"""Matching with two picture keys and a word vote (stage I4, mode `keys`). Pure numpy, no model.

The mode `tau` has one test, so one bad value hides an object. The mode `evidence` adds signals, but it still
starts from the pooled DINOv2 region vector, which is close to noise for a small object. This mode uses two
vectors of the crop of each box, from two models with different errors:

- `s`, the **SigLIP key**: the agreement of the SigLIP crop vector with the reference crops of the object.
- `d`, the **DINOv2 key**: the agreement of the DINOv2 CLS token of the tight crop with the reference crops.

The agreement is the mean of the best cosine of the `top_photos` best reference photos. A real match looks like
several photos of the object, a lucky one like one photo only. The rule:

| Result | Rule |
|--------|------|
| Strong | `s >= sig_strong` and `d >= dino_strong` |
| Weak | `s >= sig_weak`, and one more vote: `d >= dino_weak`, or a detector score of `word_vote` or more for a word of the object |
| None | All other cases |

Rules that can only lower a result:

- A box inside a screen, frame or mirror (`depicted`) is never a sighting.
- A box with a bad crop (`quality_ok = False`: small, blurred or cut by the edge) is weak at most.
  A blurred crop of the true object is still evidence. The memory needs a second weak sighting for it.
- One box is one object. If several objects pass on one box, the object with the highest `s` keeps it.
  A look-alike (`tracked = False`) can keep a box. Then the box gives no sighting.
- One sighting for each object in a frame: the box with the best result, then the highest `s`.

Words never make a sighting alone and never refuse a strong one. The bars are in `[matching]` of `config.toml`.
They come from the replay of `benchmarks/recall` with fixed bars (no cohort). They are start values.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ai.errors import MatchingError
from ai.evidence import photo_agreement
from ai.types import MatchObject, MatchRegion, Sighting

logger: logging.Logger = logging.getLogger(__name__)

NORM_TOLERANCE: float = 1e-3

NONE: int = 0
WEAK: int = 1
STRONG: int = 2


@dataclass(frozen=True)
class KeyBars:
    """The bars of the rule. See the module text."""

    top_photos: int
    sig_strong: float
    dino_strong: float
    sig_weak: float
    dino_weak: float
    word_vote: float

    def __post_init__(self) -> None:
        assert self.top_photos > 0, "top_photos must be positive"
        assert self.sig_weak <= self.sig_strong, "sig_weak must not be above sig_strong"


def _check_unit(vectors: np.ndarray, name: str) -> None:
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise MatchingError(f"{name} must be a 2-D array")
    if vectors.shape[0] == 0:
        return
    if not np.isfinite(vectors).all():
        raise MatchingError(f"{name} contain a value that is not finite")
    if np.abs(np.linalg.norm(vectors, axis=1) - 1.0).max() > NORM_TOLERANCE:
        raise MatchingError(f"{name} must have unit length")


def _check_object(obj: MatchObject, sig_size: int, dino_size: int) -> None:
    if (
        obj.references is None
        or obj.key_references is None
        or obj.reference_photos is None
    ):
        raise MatchingError(f"Object {obj.object_id} has no key vectors")
    _check_unit(obj.references, f"SigLIP references of object {obj.object_id}")
    _check_unit(obj.key_references, f"DINOv2 references of object {obj.object_id}")
    rows: int = obj.references.shape[0]
    if rows == 0 or obj.key_references.shape[0] != rows:
        raise MatchingError(
            f"Object {obj.object_id} needs one SigLIP and one DINOv2 vector for each photo"
        )
    if len(obj.reference_photos) != rows:
        raise MatchingError(f"Object {obj.object_id} needs a photo id for each row")
    if obj.references.shape[1] != sig_size or obj.key_references.shape[1] != dino_size:
        raise MatchingError(f"Object {obj.object_id} has vectors of another size")


def word_score(region: MatchRegion, obj: MatchObject) -> float:
    """The detector score of the region if one of its phrases is a phrase of the object, else 0."""
    return region.det_score if set(region.phrases) & set(obj.phrases) else 0.0


def key_level(
    s: float, d: float, word: float, region: MatchRegion, bars: KeyBars
) -> int:
    """Return NONE, WEAK or STRONG for one box and one object. See the rule in the module text."""
    if region.depicted:
        return NONE
    level: int = NONE
    if s >= bars.sig_strong and d >= bars.dino_strong:
        level = STRONG
    elif s >= bars.sig_weak and (d >= bars.dino_weak or word >= bars.word_vote):
        level = WEAK
    if level == STRONG and not region.quality_ok:
        level = WEAK  # A bad crop is evidence, but never sure evidence
    return level


def match_keys(
    sig_vectors: np.ndarray,
    dino_vectors: np.ndarray,
    regions: Sequence[MatchRegion],
    objects: Sequence[MatchObject],
    bars: KeyBars,
) -> list[Sighting]:
    """Return the sightings of the tracked objects in the boxes of one frame, at most one for each object.

    `sig_vectors` and `dino_vectors` have one row for each region. Each object needs `references`,
    `key_references` and `reference_photos`, with one row for each reference photo.
    `Sighting.similarity` is `s`, `Sighting.margin` is `s` minus the best `s` of another object, and
    `Sighting.score` is `d`. Raise `MatchingError` if the data is not usable.
    """
    assert len(sig_vectors) == len(regions), "one SigLIP vector per region"
    assert len(dino_vectors) == len(regions), "one DINOv2 vector per region"
    if not len(regions) or not objects:
        return []
    _check_unit(sig_vectors, "SigLIP vectors")
    _check_unit(dino_vectors, "DINOv2 vectors")
    ids: list[int] = [o.object_id for o in objects]
    assert len(set(ids)) == len(ids), "object ids must be unique"
    for obj in objects:
        _check_object(obj, sig_vectors.shape[1], dino_vectors.shape[1])

    n_objects: int = len(objects)
    s: np.ndarray = np.zeros((len(regions), n_objects))
    d: np.ndarray = np.zeros((len(regions), n_objects))
    level: np.ndarray = np.zeros((len(regions), n_objects), dtype=int)
    for j, obj in enumerate(objects):
        assert obj.references is not None and obj.key_references is not None
        assert obj.reference_photos is not None
        sig_cos: np.ndarray = sig_vectors @ obj.references.T
        dino_cos: np.ndarray = dino_vectors @ obj.key_references.T
        for i, region in enumerate(regions):
            s[i, j] = photo_agreement(sig_cos[i], obj.reference_photos, bars.top_photos)
            d[i, j] = photo_agreement(
                dino_cos[i], obj.reference_photos, bars.top_photos
            )
            level[i, j] = key_level(
                s[i, j], d[i, j], word_score(region, obj), region, bars
            )

    best: dict[int, tuple[int, float, Sighting]] = {}
    for i in range(len(regions)):
        passing: np.ndarray = np.where(level[i] > NONE)[0]
        if not passing.size:
            continue
        # One box is one object: the passing object with the highest SigLIP key keeps it.
        owner: int = int(passing[np.argmax(s[i, passing])])
        obj: MatchObject = objects[owner]
        if not obj.tracked:
            continue  # A look-alike keeps the box. It is never reported
        rivals: np.ndarray = np.delete(s[i], owner)
        margin: float = float(s[i, owner] - (rivals.max() if rivals.size else 0.0))
        rank: tuple[int, float] = (int(level[i, owner]), float(s[i, owner]))
        current = best.get(obj.object_id)
        if current is None or rank > current[:2]:
            sighting: Sighting = Sighting(
                obj.object_id,
                i,
                float(s[i, owner]),
                margin,
                level[i, owner] == STRONG,
                float(d[i, owner]),
            )
            best[obj.object_id] = (*rank, sighting)
    found: list[Sighting] = [best[k][2] for k in sorted(best)]
    logger.info(
        "Key matching: %d region(s), %d object(s), %d sighting(s), %d strong",
        len(regions),
        n_objects,
        len(found),
        sum(f.strong for f in found),
    )
    return found
