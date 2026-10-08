"""Settings of the ai component. They come from `config.toml`."""

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ai.errors import AiConfigError

logger: logging.Logger = logging.getLogger(__name__)

AI_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = AI_DIR / "config.toml"

# DINOv2 patch size in pixels.
PATCH_SIZE: int = 14
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class ModelsConfig:
    detector_id: str
    detector_revision: str
    embedder_id: str
    embedder_revision: str
    query_encoder_id: str
    query_encoder_revision: str
    verifier_id: str
    verifier_revision: str
    scene_id: str
    scene_revision: str
    speech_id: str
    speech_revision: str
    language_id: str
    language_revision: str
    open_detector_file: str
    open_detector_url: str
    open_detector_sha256: str
    device: str
    cpu_threads: int
    local_files_only: bool


@dataclass(frozen=True)
class ImageConfig:
    max_pixels: int


@dataclass(frozen=True)
class QualityConfig:
    blur_min: float
    analysis_max_side: int
    phash_max_distance: int
    min_crop_side: int
    crop_blur_min: float


@dataclass(frozen=True)
class DetectionConfig:
    box_threshold: float
    text_threshold: float
    input_short_edge: int
    max_per_phrase: int
    chunk_size: int
    nms_iou: float
    max_frame_fraction: float
    min_box_side: int
    hand_overlap: float
    depiction_inside: float
    default_phrases: tuple[str, ...]
    small_phrases: tuple[str, ...]
    hand_phrase: str
    depiction_phrases: tuple[str, ...]
    generic_phrase: str
    tiling_enabled: bool
    tile_overlap: float


@dataclass(frozen=True)
class EmbeddingConfig:
    frame_short_edge: int
    frame_long_edge_max: int
    crop_size: int
    small_box_min_side: int
    crop_padding: float


# The names of the views that a reference photo can give (see `ai/enrollment.py`).
AUGMENTATIONS: tuple[str, ...] = ("original", "box_smaller", "box_larger", "brighter")


@dataclass(frozen=True)
class EnrollmentConfig:
    augmentations: tuple[str, ...]
    box_scale: float
    brightness: float
    tau_std_factor: float
    tau_strong_offset: float
    tau_min: float
    margin_min: float
    margin_strong: float
    min_photos: int


@dataclass(frozen=True)
class QueryConfig:
    max_tokens: int


@dataclass(frozen=True)
class VerificationConfig:
    enabled: bool
    min_matches: int
    match_threshold: float
    crop_padding: float


@dataclass(frozen=True)
class SceneConfig:
    enabled: bool
    prompt: str
    rooms: tuple[str, ...]
    surfaces: tuple[str, ...]
    surface_context: float


@dataclass(frozen=True)
class SpeechConfig:
    enabled: bool
    max_seconds: int
    max_new_tokens: int


@dataclass(frozen=True)
class LanguageConfig:
    enabled: bool
    max_input_chars: int
    max_new_tokens: int
    max_names: int
    rewrite_system: str
    rewrite_examples: tuple[tuple[str, str], ...]
    describe_system: str
    describe_max_new_tokens: int
    describe_image_side: int


@dataclass(frozen=True)
class OpenDetectionConfig:
    enabled: bool
    confidence: float
    max_boxes: int
    input_size: int
    min_box_side: int
    crop_padding: float


@dataclass(frozen=True)
class AiConfig:
    models: ModelsConfig
    image: ImageConfig
    quality: QualityConfig
    detection: DetectionConfig
    embedding: EmbeddingConfig
    enrollment: EnrollmentConfig
    query: QueryConfig
    verification: VerificationConfig
    scene: SceneConfig
    speech: SpeechConfig
    open_detection: OpenDetectionConfig
    language: LanguageConfig


class _Table:
    """Reads typed values from one table of the config file."""

    def __init__(self, raw: dict, name: str) -> None:
        table: object = raw.get(name)
        if not isinstance(table, dict):
            raise AiConfigError(f"Missing [{name}] table in config")
        self._table: dict = table
        self._name: str = name

    def _value(self, key: str) -> object:
        if key not in self._table:
            raise AiConfigError(f"Missing key {self._name}.{key} in config")
        return self._table[key]

    def _fail(self, key: str, what: str) -> AiConfigError:
        return AiConfigError(f"{self._name}.{key} must be {what}")

    def text(self, key: str) -> str:
        value: object = self._value(key)
        if not isinstance(value, str) or not value.strip():
            raise self._fail(key, "a non-empty string")
        return value

    def choice(self, key: str, choices: tuple[str, ...]) -> str:
        value: str = self.text(key)
        if value not in choices:
            raise self._fail(key, f"one of {', '.join(choices)}")
        return value

    def boolean(self, key: str) -> bool:
        value: object = self._value(key)
        if not isinstance(value, bool):
            raise self._fail(key, "true or false")
        return value

    def integer(self, key: str, minimum: int = 1, maximum: int | None = None) -> int:
        value: object = self._value(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise self._fail(key, f"an integer of {minimum} or more")
        if maximum is not None and value > maximum:
            raise self._fail(key, f"an integer of {maximum} or less")
        return value

    def number(self, key: str, low: float, high: float) -> float:
        value: object = self._value(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._fail(key, "a number")
        if not low <= value <= high:
            raise self._fail(key, f"between {low} and {high}")
        return float(value)

    def texts(self, key: str, allow_empty: bool = False) -> tuple[str, ...]:
        value: object = self._value(key)
        if not isinstance(value, list) or not all(
            isinstance(v, str) and v.strip() for v in value
        ):
            raise self._fail(key, "a list of non-empty strings")
        if not value and not allow_empty:
            raise self._fail(key, "a list with at least one string")
        return tuple(value)


def _multiple_of_patch(table: _Table, key: str) -> int:
    value: int = table.integer(key)
    if value % PATCH_SIZE:
        raise AiConfigError(f"{table._name}.{key} must be a multiple of {PATCH_SIZE}")
    return value


def _multiple_of(value: int, base: int, name: str) -> int:
    if value % base:
        raise AiConfigError(f"{name} must be a multiple of {base}")
    return value


def _weights_file(value: str) -> str:
    """A plain file name. A path or a name with `..` could write outside `ai/models`."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.pt", value) or ".." in value:
        raise AiConfigError(
            "models.open_detector_file must be a plain file name that ends with .pt"
        )
    return value


def _https_url(value: str) -> str:
    if not value.startswith("https://"):
        raise AiConfigError("models.open_detector_url must start with https://")
    return value


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise AiConfigError("models.open_detector_sha256 must be 64 hex characters")
    return value


def _examples(table: _Table) -> tuple[tuple[str, str], ...]:
    """Read `rewrite_examples`: a list of `[question, answer]` pairs. The list can be empty."""
    value: object = table._value("rewrite_examples")
    if not isinstance(value, list) or not all(
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(text, str) and text.strip() for text in pair)
        for pair in value
    ):
        raise table._fail("rewrite_examples", "a list of [question, answer] pairs")
    return tuple((pair[0], pair[1]) for pair in value)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AiConfig:
    """Read and check the config file. Raise `AiConfigError` if it is not correct."""
    assert isinstance(config_path, Path), "config_path must be a Path"
    try:
        with config_path.open("rb") as file:
            raw: dict = tomllib.load(file)
    except FileNotFoundError as error:
        raise AiConfigError(f"Config file not found: {config_path}") from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise AiConfigError(f"Cannot read config {config_path}: {error}") from error

    models: _Table = _Table(raw, "models")
    image: _Table = _Table(raw, "image")
    quality: _Table = _Table(raw, "quality")
    detection: _Table = _Table(raw, "detection")
    embedding: _Table = _Table(raw, "embedding")
    enrollment: _Table = _Table(raw, "enrollment")
    query: _Table = _Table(raw, "query")
    verification: _Table = _Table(raw, "verification")
    scene: _Table = _Table(raw, "scene")
    speech: _Table = _Table(raw, "speech")
    open_detection: _Table = _Table(raw, "open_detection")
    language: _Table = _Table(raw, "language")

    default_phrases: tuple[str, ...] = detection.texts("default_phrases")
    small_phrases: tuple[str, ...] = detection.texts("small_phrases", allow_empty=True)
    unknown: list[str] = [p for p in small_phrases if p not in default_phrases]
    if unknown:
        raise AiConfigError(
            f"detection.small_phrases not in default_phrases: {unknown}"
        )

    frame_short: int = _multiple_of_patch(embedding, "frame_short_edge")
    frame_long: int = _multiple_of_patch(embedding, "frame_long_edge_max")
    if frame_long < frame_short:
        raise AiConfigError(
            "embedding.frame_long_edge_max must not be below frame_short_edge"
        )

    augmentations: tuple[str, ...] = enrollment.texts("augmentations")
    unknown_names: list[str] = [a for a in augmentations if a not in AUGMENTATIONS]
    if unknown_names:
        raise AiConfigError(
            f"enrollment.augmentations has unknown names {unknown_names}. "
            f"Allowed: {', '.join(AUGMENTATIONS)}"
        )
    if len(set(augmentations)) != len(augmentations):
        raise AiConfigError("enrollment.augmentations must not repeat a name")
    margin_min: float = enrollment.number("margin_min", 0.0, 1.0)
    margin_strong: float = enrollment.number("margin_strong", 0.0, 1.0)
    if margin_strong < margin_min:
        raise AiConfigError("enrollment.margin_strong must not be below margin_min")

    rooms: tuple[str, ...] = scene.texts("rooms")
    surfaces: tuple[str, ...] = scene.texts("surfaces")
    for name, words in (("rooms", rooms), ("surfaces", surfaces)):
        if len(set(words)) != len(words):
            raise AiConfigError(f"scene.{name} must not repeat a word")
    prompt: str = scene.text("prompt")
    if "{label}" not in prompt:
        raise AiConfigError("scene.prompt must contain {label}")
    scene_config: SceneConfig = SceneConfig(
        enabled=scene.boolean("enabled"),
        prompt=prompt,
        rooms=rooms,
        surfaces=surfaces,
        surface_context=scene.number("surface_context", 0.0, 5.0),
    )

    config: AiConfig = AiConfig(
        models=ModelsConfig(
            detector_id=models.text("detector_id"),
            detector_revision=models.text("detector_revision"),
            embedder_id=models.text("embedder_id"),
            embedder_revision=models.text("embedder_revision"),
            query_encoder_id=models.text("query_encoder_id"),
            query_encoder_revision=models.text("query_encoder_revision"),
            verifier_id=models.text("verifier_id"),
            verifier_revision=models.text("verifier_revision"),
            scene_id=models.text("scene_id"),
            scene_revision=models.text("scene_revision"),
            speech_id=models.text("speech_id"),
            speech_revision=models.text("speech_revision"),
            language_id=models.text("language_id"),
            language_revision=models.text("language_revision"),
            open_detector_file=_weights_file(models.text("open_detector_file")),
            open_detector_url=_https_url(models.text("open_detector_url")),
            open_detector_sha256=_sha256(models.text("open_detector_sha256")),
            device=models.choice("device", DEVICES),
            cpu_threads=models.integer("cpu_threads", minimum=0),
            local_files_only=models.boolean("local_files_only"),
        ),
        image=ImageConfig(max_pixels=image.integer("max_pixels")),
        quality=QualityConfig(
            blur_min=quality.number("blur_min", 0.0, 1e9),
            analysis_max_side=quality.integer("analysis_max_side"),
            phash_max_distance=quality.integer("phash_max_distance", minimum=0),
            min_crop_side=quality.integer("min_crop_side"),
            crop_blur_min=quality.number("crop_blur_min", 0.0, 1e9),
        ),
        detection=DetectionConfig(
            box_threshold=detection.number("box_threshold", 0.0, 1.0),
            text_threshold=detection.number("text_threshold", 0.0, 1.0),
            input_short_edge=detection.integer("input_short_edge"),
            max_per_phrase=detection.integer("max_per_phrase"),
            chunk_size=detection.integer("chunk_size"),
            nms_iou=detection.number("nms_iou", 0.01, 1.0),
            max_frame_fraction=detection.number("max_frame_fraction", 0.01, 1.0),
            min_box_side=detection.integer("min_box_side"),
            hand_overlap=detection.number("hand_overlap", 0.0, 1.0),
            depiction_inside=detection.number("depiction_inside", 0.0, 1.0),
            default_phrases=default_phrases,
            small_phrases=small_phrases,
            hand_phrase=detection.text("hand_phrase"),
            depiction_phrases=detection.texts("depiction_phrases", allow_empty=True),
            generic_phrase=detection.text("generic_phrase"),
            tiling_enabled=detection.boolean("tiling_enabled"),
            tile_overlap=detection.number("tile_overlap", 0.0, 0.9),
        ),
        embedding=EmbeddingConfig(
            frame_short_edge=frame_short,
            frame_long_edge_max=frame_long,
            crop_size=_multiple_of_patch(embedding, "crop_size"),
            small_box_min_side=embedding.integer("small_box_min_side"),
            crop_padding=embedding.number("crop_padding", 0.0, 1.0),
        ),
        enrollment=EnrollmentConfig(
            augmentations=augmentations,
            box_scale=enrollment.number("box_scale", 0.01, 0.5),
            brightness=enrollment.number("brightness", 0.01, 1.0),
            tau_std_factor=enrollment.number("tau_std_factor", 0.0, 5.0),
            tau_strong_offset=enrollment.number("tau_strong_offset", 0.0, 1.0),
            tau_min=enrollment.number("tau_min", 0.0, 1.0),
            margin_min=margin_min,
            margin_strong=margin_strong,
            min_photos=enrollment.integer("min_photos", minimum=2),
        ),
        query=QueryConfig(max_tokens=query.integer("max_tokens")),
        verification=VerificationConfig(
            enabled=verification.boolean("enabled"),
            min_matches=verification.integer("min_matches"),
            match_threshold=verification.number("match_threshold", 0.0, 1.0),
            crop_padding=verification.number("crop_padding", 0.0, 1.0),
        ),
        scene=scene_config,
        speech=SpeechConfig(
            enabled=speech.boolean("enabled"),
            max_seconds=speech.integer("max_seconds", minimum=1, maximum=30),
            max_new_tokens=speech.integer("max_new_tokens", minimum=1, maximum=440),
        ),
        open_detection=OpenDetectionConfig(
            enabled=open_detection.boolean("enabled"),
            confidence=open_detection.number("confidence", 0.01, 1.0),
            max_boxes=open_detection.integer("max_boxes", minimum=1, maximum=200),
            input_size=_multiple_of(
                open_detection.integer("input_size"), 32, "open_detection.input_size"
            ),
            min_box_side=open_detection.integer("min_box_side"),
            crop_padding=open_detection.number("crop_padding", 0.0, 1.0),
        ),
        language=LanguageConfig(
            enabled=language.boolean("enabled"),
            max_input_chars=language.integer(
                "max_input_chars", minimum=10, maximum=2000
            ),
            max_new_tokens=language.integer("max_new_tokens", minimum=1, maximum=200),
            max_names=language.integer("max_names", minimum=1, maximum=10),
            rewrite_system=language.text("rewrite_system"),
            rewrite_examples=_examples(language),
            describe_system=language.text("describe_system"),
            describe_max_new_tokens=language.integer(
                "describe_max_new_tokens", minimum=1, maximum=300
            ),
            describe_image_side=language.integer(
                "describe_image_side", minimum=64, maximum=2048
            ),
        ),
    )
    logger.info("Loaded ai config from %s", config_path)
    return config
