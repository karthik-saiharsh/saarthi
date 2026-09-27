"""Public interface of the ai component.

`AiEngine` loads both models once. The `backend` module calls its methods in
the order that it chooses. The engine never reads the database or the disk.
"""

import logging
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ai.calibration import calibrate
from ai.detector import CandidateDetector
from ai.embedder import RegionEmbedder
from ai.enrollment import brighten, scale_box
from ai.errors import (
    AiConfigError,
    AiError,
    CalibrationError,
    ImageError,
    InferenceError,
    MatchingError,
    ModelLoadError,
)
from ai.matching import match_regions
from ai.postprocess import apply_crop_quality, select_candidates
from ai.quality import blur_score, hamming_distance, perceptual_hash
from ai.runtime import configure_threads
from ai.settings import AiConfig, load_config
from ai.types import (
    Box,
    Calibration,
    Candidate,
    LocatedObject,
    MatchObject,
    MatchRegion,
    QualityReport,
    Sighting,
)

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "AiConfigError",
    "AiEngine",
    "AiError",
    "CalibrationError",
    "ImageError",
    "InferenceError",
    "MatchingError",
    "ModelLoadError",
    "open_image",
]


def open_image(data: bytes, max_pixels: int) -> Image.Image:
    """Decode image bytes into an RGB image.

    The EXIF orientation is applied, so the pixel grid is the one that a
    person sees. Raise `ImageError` for a file that is not an image, is
    broken, or has more than `max_pixels` pixels.
    """
    assert isinstance(data, bytes), "data must be bytes"
    assert max_pixels > 0, "max_pixels must be positive"
    try:
        image: Image.Image = Image.open(BytesIO(data))
        width, height = image.size
        if width * height > max_pixels:
            raise ImageError(f"Image has more than {max_pixels} pixels")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except ImageError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        logger.warning("Cannot decode an image: %s", type(error).__name__)
        raise ImageError("The file is not a usable image") from error


class AiEngine:
    """The stage functions of the ingestion pipeline."""

    def __init__(
        self,
        config: AiConfig | None = None,
        detector: CandidateDetector | None = None,
        embedder: RegionEmbedder | None = None,
    ) -> None:
        """Load the models. Pass `detector` or `embedder` only in tests."""
        self._config: AiConfig = config if config is not None else load_config()
        configure_threads(self._config.models.cpu_threads)
        self._detector: CandidateDetector = detector or CandidateDetector(self._config)
        self._embedder: RegionEmbedder = embedder or RegionEmbedder(self._config)
        logger.info(
            "AI engine ready: detector %s, embedder %s",
            self._detector.name,
            self._embedder.name,
        )

    @property
    def embedding_model(self) -> str:
        """Name for `Region.embedding_model`, for example `dinov2-small@ed25f3a3`."""
        return self._embedder.name

    @property
    def embedding_dimension(self) -> int:
        """Length of one vector. It must equal `vectors.dimension` of the db."""
        return self._embedder.dimension

    def open_image(self, data: bytes) -> Image.Image:
        """Decode image bytes. See the function `open_image`."""
        return open_image(data, self._config.image.max_pixels)

    def assess_quality(self, image: Image.Image) -> QualityReport:
        """Return the blur score and the perceptual hash of an image."""
        assert isinstance(image, Image.Image), "image must be a PIL image"
        return QualityReport(
            blur_score=blur_score(image, self._config.quality.analysis_max_side),
            phash=perceptual_hash(image),
        )

    def is_blurry(self, report: QualityReport) -> bool:
        """Return True if the image is too blurry to use."""
        assert isinstance(report, QualityReport), "report must be a QualityReport"
        return report.blur_score < self._config.quality.blur_min

    def is_near_duplicate(self, phash: str, recent_phashes: list[str]) -> bool:
        """Return True if a recent hash is close to `phash`."""
        limit: int = self._config.quality.phash_max_distance
        return any(hamming_distance(phash, other) <= limit for other in recent_phashes)

    def detect_candidates(
        self,
        image: Image.Image,
        phrases: list[str] | None = None,
        small_phrases: list[str] | None = None,
        include_generic: bool = False,
        tiling: bool | None = None,
    ) -> list[Candidate]:
        """Find boxes that might show an object. The best score comes first.

        Without `phrases`, the phrases of `config.toml` are used. Set
        `include_generic` for the extra pass with the phrase "an object". `tiling`
        switches the tiled pass for this call. None uses the config.
        """
        assert isinstance(image, Image.Image), "image must be a PIL image"
        cfg = self._config.detection
        chosen: list[str] = (
            list(phrases) if phrases is not None else list(cfg.default_phrases)
        )
        assert chosen, "phrases must not be empty"
        small: list[str] = (
            list(small_phrases)
            if small_phrases is not None
            else [p for p in cfg.small_phrases if p in chosen]
        )
        raw = self._detector.detect(image, chosen, small, include_generic, tiling)
        candidates: list[Candidate] = select_candidates(
            raw, image.width, image.height, chosen, cfg
        )
        quality = self._config.quality
        result: list[Candidate] = apply_crop_quality(
            candidates, image, quality.min_crop_side, quality.crop_blur_min
        )
        logger.info(
            "Found %d candidate(s) from %d raw box(es), %d with good quality",
            len(result),
            len(raw),
            sum(c.quality_ok for c in result),
        )
        return result

    def embed_regions(self, image: Image.Image, boxes: list[Box]) -> np.ndarray:
        """Return one unit-length float32 vector for each box."""
        return self._embedder.embed(image, boxes)

    # ---- Enrollment (stages E2 to E4) ----

    @property
    def min_photos(self) -> int:
        """The number of confirmed photos that an object needs before it is calibrated."""
        return self._config.enrollment.min_photos

    def locate_object(
        self,
        image: Image.Image,
        category_phrase: str,
        synonyms: list[str] | None = None,
    ) -> LocatedObject | None:
        """Find the enrolled object in a reference photo (stage E2).

        Run the detector with the category phrase and the synonyms. Return the
        box with the best score, or None if the detector finds nothing.
        """
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert isinstance(category_phrase, str) and category_phrase, "phrase needed"
        phrases: list[str] = [category_phrase, *(synonyms or [])]
        # Only the phrases of the object. A hand or a book cover (a "picture frame") must not
        # compete for the labels, and the depiction rule must not drop an object that lies on a book.
        raw = self._detector.detect(image, phrases, (), False, False, False)
        candidates: list[Candidate] = select_candidates(
            raw, image.width, image.height, phrases, self._config.detection
        )
        matches: list[tuple[Candidate, str]] = [
            (c, p) for c in candidates for p in c.phrases if p in phrases
        ]
        if not matches:
            logger.info("The object was not found in the reference photo")
            return None
        candidate, phrase = max(matches, key=lambda m: m[0].det_score)
        logger.info("Object found with score %.2f", candidate.det_score)
        return LocatedObject(candidate.box, candidate.det_score, phrase)

    def embed_views(self, image: Image.Image, box: Box) -> list[tuple[str, np.ndarray]]:
        """Make the views of an object in a reference photo (stage E3).

        One vector for each name in `enrollment.augmentations`. Return a list
        of (name, vector) in the order of the config.
        """
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert isinstance(box, Box), "box must be a Box"
        cfg = self._config.enrollment
        width, height = image.size
        boxes: dict[str, Box] = {
            "original": box,
            "box_smaller": scale_box(box, -cfg.box_scale, width, height),
            "box_larger": scale_box(box, cfg.box_scale, width, height),
        }
        box_names: list[str] = [n for n in cfg.augmentations if n in boxes]
        vectors: dict[str, np.ndarray] = {}
        if box_names:
            made: np.ndarray = self._embedder.embed(
                image, [boxes[n] for n in box_names]
            )
            vectors.update({name: made[i] for i, name in enumerate(box_names)})
        if "brighter" in cfg.augmentations:
            vectors["brighter"] = self._embedder.embed(
                brighten(image, cfg.brightness), [box]
            )[0]
        return [(name, vectors[name]) for name in cfg.augmentations]

    def calibrate_object(
        self, views: np.ndarray, photo_ids: list[int], other_views: np.ndarray
    ) -> Calibration:
        """Calculate the thresholds of an object from its views and the negatives (stage E4)."""
        return calibrate(views, photo_ids, other_views, self._config.enrollment)

    # ---- Matching (stage I4) ----

    def match_regions(
        self,
        vectors: np.ndarray,
        regions: list[MatchRegion],
        objects: list[MatchObject],
    ) -> list[Sighting]:
        """Match the region vectors of one frame with the enrolled objects (stage I4).

        The objects must be `ready`. A region with `quality_ok = False` is
        skipped. The result has at most one sighting for each object.
        """
        return match_regions(
            vectors,
            regions,
            objects,
            self._config.detection.generic_phrase,
            self._config.enrollment.margin_strong,
        )
