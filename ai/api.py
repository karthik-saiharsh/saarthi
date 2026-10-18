"""Public interface of the ai component.

`AiEngine` loads both models once. The `backend` module calls its methods in
the order that it chooses. The engine never reads the database or the disk.
"""

import hashlib
import logging
import threading
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ai.calibration import NOTE_ONE_PLACE, calibrate, one_place
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
from ai.evidence import score_regions, select_regions
from ai.keys import KeyBars, match_keys, review_scores
from ai.language import LanguageModel
from ai.matching import match_regions
from ai.open_detector import OpenDetector
from ai.postprocess import apply_crop_quality, select_candidates
from ai.quality import blur_score, hamming_distance, perceptual_hash
from ai.query import QueryEncoder
from ai.runtime import configure_threads
from ai.scene import SceneClassifier, context_box
from ai.settings import AiConfig, load_config
from ai.speech import SAMPLE_RATE, SpeechRecognizer
from ai.types import (
    Box,
    Calibration,
    Candidate,
    LocatedObject,
    MatchObject,
    MatchRegion,
    OpenDetection,
    QualityReport,
    RawDetection,
    Sighting,
    VerifyResult,
)
from ai.verifier import KeypointVerifier

logger: logging.Logger = logging.getLogger(__name__)

# The colour that hides the object in a photo, so that only its place is left.
PLACE_GREY: tuple[int, int, int] = (128, 128, 128)

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


def _bare_phrase(phrase: str) -> str:
    """Return a phrase in lower case without its article, so "A bottle" and "bottle" are the same."""
    words: list[str] = phrase.lower().split()
    if words and words[0] in ("a", "an", "the"):
        words = words[1:]
    return " ".join(words)


class AiEngine:
    """The stage functions of the ingestion pipeline."""

    def __init__(
        self,
        config: AiConfig | None = None,
        detector: CandidateDetector | None = None,
        embedder: RegionEmbedder | None = None,
        query_encoder: QueryEncoder | None = None,
        verifier: KeypointVerifier | None = None,
        scene_classifier: SceneClassifier | None = None,
        speech_recognizer: SpeechRecognizer | None = None,
        open_detector: OpenDetector | None = None,
        language_model: LanguageModel | None = None,
    ) -> None:
        """Load the models. Pass `detector`, `embedder`, `query_encoder`, `verifier` or `scene_classifier` only in tests.

        The verifier loads only if `verification.enabled` is true.
        The scene classifier loads only if `scene.enabled` is true.
        The speech recognizer loads only if `speech.enabled` is true.
        The broad detector loads only if `open_detection.enabled` is true.
        The language model loads only if `language.enabled` is true.
        """
        self._config: AiConfig = config if config is not None else load_config()
        configure_threads(self._config.models.cpu_threads)
        self._detector: CandidateDetector = detector or CandidateDetector(self._config)
        self._embedder: RegionEmbedder = embedder or RegionEmbedder(self._config)
        self._query_encoder: QueryEncoder = query_encoder or QueryEncoder(self._config)
        self._scene: SceneClassifier | None = scene_classifier or (
            SceneClassifier(self._config) if self._config.scene.enabled else None
        )
        self._speech: SpeechRecognizer | None = speech_recognizer or (
            SpeechRecognizer(self._config) if self._config.speech.enabled else None
        )
        self._open_detector: OpenDetector | None = open_detector or (
            OpenDetector(self._config) if self._config.open_detection.enabled else None
        )
        self._language: LanguageModel | None = language_model or (
            LanguageModel(self._config) if self._config.language.enabled else None
        )
        self._verifier: KeypointVerifier | None = verifier or (
            KeypointVerifier(self._config)
            if self._config.verification.enabled
            else None
        )
        self._phrase_lock: threading.Lock = threading.Lock()
        self._phrase_cache: tuple[tuple[str, ...], np.ndarray] | None = None
        logger.info(
            "AI engine ready: detector %s, embedder %s, query encoder %s, verifier %s, scene %s, speech %s, broad detector %s, language %s",
            self._detector.name,
            self._embedder.name,
            self._query_encoder.name,
            self._verifier.name if self._verifier is not None else "off",
            self._scene.name if self._scene is not None else "off",
            self._speech.name if self._speech is not None else "off",
            self._open_detector.name if self._open_detector is not None else "off",
            self._language.name if self._language is not None else "off",
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

    def detect_phrases(
        self, image: Image.Image, phrases: list[str]
    ) -> list[RawDetection]:
        """Run the detector on one frame with the words of a user. Return every box with its score.

        Each phrase is made lowercase, loses its periods and is cut to `detection.phrase_max_chars`.
        There are no hand or depiction phrases, and no tiles. The boxes are raw: the caller decides.
        The result is empty if no phrase is left. It is used by the verified open search.
        """
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert isinstance(phrases, list), "phrases must be a list"
        limit: int = self._config.detection.phrase_max_chars
        cleaned: list[str] = []
        for phrase in phrases:
            text: str = " ".join(phrase.lower().replace(".", " ").split())[
                :limit
            ].strip()
            if text and text not in cleaned:
                cleaned.append(text)
        if not cleaned:
            return []
        return self._detector.detect(image, cleaned, (), False, False, False)

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

    def embed_place(self, image: Image.Image, box: Box) -> np.ndarray:
        """Return the background of a reference photo: the DINOv2 key of the whole photo, with the object painted grey."""
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert isinstance(box, Box), "box must be a Box"
        background: Image.Image = image.convert(
            "RGB"
        )  # A copy: the caller keeps its picture
        background.paste(PLACE_GREY, (box.x0, box.y0, box.x1, box.y1))
        whole: Box = Box(0, 0, image.width, image.height)
        return self.embed_key_crops(background, [whole])[0]

    def place_note(self, place_vectors: np.ndarray) -> str | None:
        """Return a warning for the carer if all photos of an object show one place (plan part D1), or None."""
        if one_place(place_vectors, self._config.enrollment.place_same_min):
            return NOTE_ONE_PLACE
        return None

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

    # ---- Evidence-based matching (stage I4, mode `evidence`) ----

    @property
    def matching_mode(self) -> str:
        """`keys`, `evidence` or `tau`. Both `keys` and `evidence` need the scene model. Without it the mode is `tau`."""
        if (
            self._config.matching.mode in ("evidence", "keys")
            and self._scene is not None
        ):
            return self._config.matching.mode
        return "tau"

    def evidence_regions(
        self,
        vectors: np.ndarray,
        regions: list[MatchRegion],
        objects: list[MatchObject],
    ) -> list[int]:
        """Return the indices of the regions of one frame that `score_evidence` must score, the most similar first."""
        return select_regions(vectors, regions, objects, self._config.matching)

    def embed_evidence_crops(self, image: Image.Image, boxes: list[Box]) -> np.ndarray:
        """Return a unit-length SigLIP vector for each box, cut with `matching.crop_padding`.

        Use it for all boxes of a frame and for the boxes of the reference photos, so all are cut in the same way.
        The vectors are in the space of the phrase vectors (`embed_open_phrases`), so the open search uses them too.
        """
        assert self._scene is not None, "the scene classifier is switched off"
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert isinstance(boxes, list), "boxes must be a list"
        pad: float = self._config.matching.crop_padding
        crops: list[Image.Image] = []
        for box in boxes:
            pad_x: int = round(box.width * pad)
            pad_y: int = round(box.height * pad)
            crops.append(
                image.crop(
                    (
                        max(0, box.x0 - pad_x),
                        max(0, box.y0 - pad_y),
                        min(image.width, box.x1 + pad_x),
                        min(image.height, box.y1 + pad_y),
                    )
                )
            )
        return self._scene.embed_crops(crops)

    def score_evidence(
        self,
        region_indices: list[int],
        vectors: np.ndarray,
        crop_vectors: np.ndarray,
        objects: list[MatchObject],
    ) -> list[Sighting]:
        """Score the chosen regions of one frame (see `ai/evidence.py`). Return at most one sighting for each object.

        `crop_vectors` come from `embed_evidence_crops`, one for each index. Each object needs `view_photos`,
        `references` and `reference_photos`.
        """
        assert self._scene is not None, "the scene classifier is switched off"
        if not region_indices or not objects:
            return []
        phrases, owners = self._evidence_classes(objects)
        phrase_vectors: np.ndarray = self._phrase_vectors(phrases)
        logits: np.ndarray = self._scene.pair_logits(crop_vectors, phrase_vectors)
        n_classes: int = int(owners.max()) + 1
        class_logits: np.ndarray = np.stack(
            [logits[:, owners == c].max(axis=1) for c in range(n_classes)], axis=1
        )
        return score_regions(
            region_indices,
            vectors,
            crop_vectors,
            class_logits,
            objects,
            self._config.matching,
        )

    # ---- Matching with two keys (stage I4, mode `keys`) ----

    @property
    def key_model(self) -> str:
        """Name of the DINOv2 key for `Region.key_model`, with the crop size: keys of another size cannot be compared."""
        return f"{self._embedder.name}/cls{self._config.matching.key_crop_size}"

    def embed_key_crops(self, image: Image.Image, boxes: list[Box]) -> np.ndarray:
        """Return the DINOv2 key of each box: the CLS token of the tight crop. Use it for frames and reference photos."""
        return self._embedder.embed_crops_cls(
            image, boxes, self._config.matching.key_crop_size
        )

    def broad_candidates(
        self, image: Image.Image, detections: list[OpenDetection]
    ) -> list[Candidate]:
        """Turn the boxes of the broad detector into candidates for the mode `keys`, with the crop gate applied.

        They have no phrase, because the names of the broad detector are not the words of the gallery.
        They are never in a hand and never inside a depiction: only the candidate detector finds hands and screens.
        """
        assert isinstance(image, Image.Image), "image must be a PIL image"
        quality = self._config.quality
        raw: list[Candidate] = [
            Candidate(d.box, d.confidence, "broad", (), False, True) for d in detections
        ]
        return apply_crop_quality(
            raw, image, quality.min_crop_side, quality.crop_blur_min
        )

    def match_keys(
        self,
        sig_vectors: np.ndarray,
        dino_vectors: np.ndarray,
        regions: list[MatchRegion],
        objects: list[MatchObject],
    ) -> list[Sighting]:
        """Match the boxes of one frame with two keys and a word vote (see `ai/keys.py`).

        `sig_vectors` come from `embed_evidence_crops`, `dino_vectors` from `embed_key_crops`, one row for each region.
        Each object needs `references`, `key_references` and `reference_photos`. A look-alike also needs them.
        """
        cfg = self._config.matching
        bars: KeyBars = KeyBars(
            cfg.top_photos,
            cfg.sig_strong,
            cfg.dino_strong,
            cfg.sig_weak,
            cfg.dino_weak,
            cfg.word_vote,
        )
        return match_keys(sig_vectors, dino_vectors, regions, objects, bars)

    def review_scores(
        self, sig_vectors: np.ndarray, dino_vectors: np.ndarray, obj: MatchObject
    ) -> np.ndarray:
        """Return the SigLIP key of each stored box for one object, for "Is this your object?". A refused box gets -inf."""
        return review_scores(
            sig_vectors, dino_vectors, obj, self._config.matching.top_photos
        )

    def _evidence_classes(
        self, objects: list[MatchObject]
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Return the phrases and the class of each phrase: one class for each object, then one for each contrast phrase.

        A contrast phrase that is also a phrase of an object is left out. Else "a bottle" of a registered bottle and the
        contrast "a bottle" would share the same score, and the meaning of the object would look unsure.
        """
        phrases: list[str] = []
        owners: list[int] = []
        for index, obj in enumerate(objects):
            phrases.extend(obj.phrases)
            owners.extend([index] * len(obj.phrases))
        own: set[str] = {_bare_phrase(p) for p in phrases}
        contrast: list[str] = [
            p
            for p in self._config.matching.contrast_phrases
            if _bare_phrase(p) not in own
        ]
        for offset, phrase in enumerate(contrast):
            phrases.append(phrase)
            owners.append(len(objects) + offset)
        return tuple(phrases), np.array(owners)

    def _phrase_vectors(self, phrases: tuple[str, ...]) -> np.ndarray:
        """Return the SigLIP vectors of the phrases. The last list is kept, because the gallery seldom changes."""
        assert self._scene is not None, "the scene classifier is switched off"
        with self._phrase_lock:
            if self._phrase_cache is not None and self._phrase_cache[0] == phrases:
                return self._phrase_cache[1]
        vectors: np.ndarray = self._scene.embed_phrases(list(phrases))
        with self._phrase_lock:
            self._phrase_cache = (phrases, vectors)
        return vectors

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

    # ---- The place of an object (stage Q2) ----

    @property
    def language_enabled(self) -> bool:
        """True if the language model is loaded. If False, `rewrite_query` must not be called."""
        return self._language is not None

    def rewrite_query(self, words: str) -> list[str]:
        """Turn a description of an object into up to `language.max_names` common names, for example "laptop, computer".

        Use it when the words name no object that the system knows. The names only feed the search.
        The list can be empty. The words stay in memory and are never logged.
        """
        assert isinstance(words, str), "words must be a string"
        assert self._language is not None, "the language model is switched off"
        return self._language.rewrite_query(words)

    @property
    def description_model(self) -> str:
        """Name for the stored descriptions: the language model and a fingerprint of the instruction. It changes when either changes."""
        assert self._language is not None, "the language model is switched off"
        text: str = self._config.language.describe_system
        return f"{self._language.name}/{hashlib.sha1(text.encode()).hexdigest()[:8]}"

    def describe_scene(self, picture: Image.Image, facts: list[str]) -> str:
        """Write one sentence about the place of an object from the picture and the checked facts (a description).

        The sentence can contain details that nobody checked: the caller must check its words against the facts.
        It takes 10 to 17 s on a CPU, so call it in the background.
        """
        assert isinstance(picture, Image.Image), "picture must be a PIL image"
        assert isinstance(facts, list) and facts, "facts must not be empty"
        assert self._language is not None, "the language model is switched off"
        return self._language.describe_scene(picture, facts)

    @property
    def open_detection_enabled(self) -> bool:
        """True if the broad detector is loaded. If False, `detect_open` must not be called."""
        return self._open_detector is not None

    def detect_open(self, image: Image.Image) -> list[OpenDetection]:
        """Name the objects of a frame without a list of classes (broad detection).

        Use it for objects that nobody registered. The names can be wrong, so treat them as hints.
        The most confident box comes first. The list can be empty.
        """
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert self._open_detector is not None, "the broad detector is switched off"
        return self._open_detector.detect(image)

    @property
    def open_vector_dimension(self) -> int:
        """The size of the crop vectors of `embed_evidence_crops` and of `embed_open_phrases`. They need the scene model."""
        assert self._scene is not None, "the scene classifier is switched off"
        return self._scene.dimension

    @property
    def open_vector_model(self) -> str:
        """Name of the model that makes the crop vectors and the phrase vectors. Store it with the crop vectors."""
        assert self._scene is not None, "the scene classifier is switched off"
        return self._scene.name

    def embed_open_phrases(self, phrases: list[str]) -> np.ndarray:
        """Return a unit-length SigLIP vector for each phrase, for example `laptop`. An article is added if it is missing."""
        assert self._scene is not None, "the scene classifier is switched off"
        return self._scene.embed_phrases(phrases)

    def open_pair_scores(self, crops: np.ndarray, phrases: np.ndarray) -> np.ndarray:
        """Return the probability that each crop vector shows each phrase vector, shape (crops, phrases).

        The values are the SigLIP scores. About 0.5 or more is a good fit. They are not a share of 1.
        """
        assert self._scene is not None, "the scene classifier is switched off"
        return self._scene.pair_probabilities(crops, phrases)

    @property
    def speech_enabled(self) -> bool:
        """True if the speech recognizer is loaded. If False, `transcribe` must not be called."""
        return self._speech is not None

    @property
    def speech_max_seconds(self) -> int:
        """The longest recording that `transcribe` accepts."""
        return self._config.speech.max_seconds

    @property
    def speech_sample_rate(self) -> int:
        """The sample rate of the audio that `transcribe` accepts."""
        return SAMPLE_RATE

    def transcribe(self, samples: np.ndarray) -> str:
        """Turn speech into English text. `samples` are float32, mono, 16 kHz, and not longer than `speech_max_seconds`.

        The audio stays in memory. The result can be empty if the model hears no speech.
        """
        assert self._speech is not None, "the speech recognizer is switched off"
        return self._speech.transcribe(samples)

    @property
    def scene_enabled(self) -> bool:
        """True if the scene classifier is loaded. If False, the two methods below must not be called."""
        return self._scene is not None

    @property
    def scene_model(self) -> str:
        """Name for the cache of the room scores: the model, and a fingerprint of the prompt and the word lists.

        It changes when the model, the prompt or a list changes, so old scores are not used with new words.
        """
        assert self._scene is not None, "the scene classifier is switched off"
        scene = self._config.scene
        text: str = "\n".join([scene.prompt, *scene.rooms, "--", *scene.surfaces])
        return f"{self._scene.name}/{hashlib.sha1(text.encode()).hexdigest()[:8]}"

    def score_rooms(self, image: Image.Image) -> dict[str, float]:
        """Return the probability of each room of `scene.rooms` for a whole picture. The values add up to 1."""
        assert self._scene is not None, "the scene classifier is switched off"
        assert isinstance(image, Image.Image), "image must be a PIL image"
        return self._scene.room_scores(image)

    def score_surfaces(self, image: Image.Image, box: Box) -> list[tuple[str, float]]:
        """Return the surfaces of `scene.surfaces` that fit the place of the object, the best first.

        The model looks at the box with space around it (`scene.surface_context`).
        """
        assert self._scene is not None, "the scene classifier is switched off"
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert isinstance(box, Box), "box must be a Box"
        around: Box = context_box(
            box, image.width, image.height, self._config.scene.surface_context
        )
        return self._scene.surface_scores(
            image.crop((around.x0, around.y0, around.x1, around.y1))
        )
