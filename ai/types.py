"""Plain data objects that the ai component returns."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Source = Literal["full", "tile", "generic"]


@dataclass(frozen=True)
class Box:
    """A box in pixels. `x1` and `y1` are exclusive. The origin is the top-left corner."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        assert all(isinstance(v, int) for v in (self.x0, self.y0, self.x1, self.y1))
        assert self.x0 < self.x1 and self.y0 < self.y1, "box must not be empty"

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class QualityReport:
    """Result of the image quality check."""

    blur_score: float  # Variance of the Laplacian. A low value means a blurry image
    phash: str  # 16 hex characters


@dataclass(frozen=True)
class Candidate:
    """A box that might show an object of the vocabulary."""

    box: Box
    det_score: float  # Detector score. For analysis only, not for identity
    source: Source
    phrases: tuple[str, ...]  # Labels of all merged boxes
    in_hand: bool
    quality_ok: bool  # False if the crop is small, blurry or cut by the image edge


@dataclass(frozen=True)
class RawDetection:
    """One box from one detector pass, before post-processing."""

    box: Box
    score: float
    label: str  # Text label from the detector. It can be a part of a phrase
    source: Source


@dataclass(frozen=True)
class LocatedObject:
    """Where an enrolled object is in a reference photo (stage E2)."""

    box: Box
    det_score: float
    phrase: str  # The phrase that the detector linked to the box


@dataclass(frozen=True)
class Calibration:
    """The result of the calibration of one object (stage E4)."""

    tau: float  # A match needs at least this similarity
    tau_strong: float  # A match above this value is strong
    margin_min: float  # A match must beat the nearest look-alike by this value
    own_mean: float  # Mean similarity between views of different photos
    own_std: float
    negative_max: (
        float | None
    )  # The highest similarity to a view of another object, if any
    n_views: int
    n_photos: int
    note: str | None  # A short text for the carer, for example a warning


@dataclass(frozen=True)
class MatchRegion:
    """What the matching step needs to know about one region besides its vector."""

    phrases: tuple[str, ...]  # The detector phrases of the region
    quality_ok: bool  # False: the region is never matched


@dataclass(frozen=True)
class MatchObject:
    """An enrolled object as the matching step sees it. The object must be `ready`."""

    object_id: int
    tracked: bool  # False: a look-alike. It is a negative and is never reported
    phrases: tuple[str, ...]  # The category phrase and the synonyms
    views: np.ndarray  # Unit-length vectors, shape (number of views, dimension)
    tau: float
    tau_strong: float
    margin_min: float


@dataclass(frozen=True)
class Sighting:
    """A region that matches an enrolled object (stage I4)."""

    object_id: int
    region_index: int  # The position of the region in the call
    similarity: float  # The highest cosine to a view of the object
    margin: float  # The similarity minus the highest similarity to any other object
    strong: bool  # True: above `tau_strong` with the strong margin


@dataclass(frozen=True)
class VerifyResult:
    """The result of the check of a sighting against the reference pictures (stage I5)."""

    verified: (
        bool  # True: a reference picture has at least `min_matches` matching keypoints
    )
    matches: int  # The highest number of matching keypoints that was found
    compared: int  # The number of reference pictures that were compared
