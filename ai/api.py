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
from ai.query import QueryEncoder
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
    VerifyResult,
)
from ai.verifier import KeypointVerifier

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
        query_encoder: QueryEncoder | None = None,
        verifier: KeypointVerifier | None = None,
    ) -> None:
        """Load the models. Pass `detector`, `embedder`, `query_encoder` or `verifier` only in tests.

        The verifier loads only if `verification.enabled` is true.
        """
        self._config: AiConfig = config if config is not None else load_config()
        configure_threads(self._config.models.cpu_threads)
        self._detector: CandidateDetector = detector or CandidateDetector(self._config)
        self._embedder: RegionEmbedder = embedder or RegionEmbedder(self._config)
        self._query_encoder: QueryEncoder = query_encoder or QueryEncoder(self._config)
        self._verifier: KeypointVerifier | None = verifier or (
            KeypointVerifier(self._config)
            if self._config.verification.enabled
            else None
        )
        logger.info(
            "AI engine ready: detector %s, embedder %s, query encoder %s, verifier %s",
            self._detector.name,
            self._embedder.name,
            self._query_encoder.name,
            self._verifier.name if self._verifier is not None else "off",
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

    # ---- Questions of the user (stage Q1) ----

    @property
    def query_model(self) -> str:
        """Name of the query encoder, for example `all-MiniLM-L6-v2@1110a243`."""
        return self._query_encoder.name

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return one unit-length float32 vector for each text (a question or a description).

        The vectors of the query encoder are not comparable with the vectors of
        the regions. Compare them with each other only.
        """
        return self._query_encoder.encode(texts)

    # ---- Verification (stage I5) ----

    @property
    def verification_enabled(self) -> bool:
        """True if the verifier is loaded. If False, `verify_match` must not be called."""
        return self._verifier is not None

    def verification_crop(self, image: Image.Image, box: Box) -> Image.Image:
        """Cut the picture that the verifier compares: the box with `verification.crop_padding` around it.

        Use it for the reference photos and for the candidate, so both are cut in the same way.
        """
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert isinstance(box, Box), "box must be a Box"
        pad: float = self._config.verification.crop_padding
        pad_x: int = round(box.width * pad)
        pad_y: int = round(box.height * pad)
        return image.crop(
            (
                max(0, box.x0 - pad_x),
                max(0, box.y0 - pad_y),
                min(image.width, box.x1 + pad_x),
                min(image.height, box.y1 + pad_y),
            )
        )

    def verify_match(
        self, references: list[Image.Image], candidate: Image.Image
    ) -> VerifyResult:
        """Check a candidate against the reference pictures of an object (stage I5).

        Compare in the given order. Stop at the first reference with at least
        `verification.min_matches` matching keypoints, so put the most likely
        reference first. The pictures come from `verification_crop`.
        """
        assert self._verifier is not None, "the verifier is switched off"
        assert isinstance(references, list), "references must be a list"
        assert isinstance(candidate, Image.Image), "candidate must be a PIL image"
        needed: int = self._config.verification.min_matches
        best: int = 0
        compared: int = 0
        for reference in references:
            best = max(best, self._verifier.count_matches(reference, candidate))
            compared += 1
            if best >= needed:
                break
        result: VerifyResult = VerifyResult(best >= needed, best, compared)
        logger.info(
            "Verification: %s (best %d of %d needed, %d reference(s) compared)",
            "passed" if result.verified else "not passed",
            best,
            needed,
            compared,
        )
        return result
