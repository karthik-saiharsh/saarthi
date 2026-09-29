"""Settings of the ai component. They come from `config.toml`."""

import logging
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
    margin_min: float
    margin_strong: float
    min_photos: int


@dataclass(frozen=True)
class QueryConfig:
    max_tokens: int


@dataclass(frozen=True)
class AiConfig:
    models: ModelsConfig
    image: ImageConfig
    quality: QualityConfig
    detection: DetectionConfig
    embedding: EmbeddingConfig
    enrollment: EnrollmentConfig
    query: QueryConfig


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

    def integer(self, key: str, minimum: int = 1) -> int:
        value: object = self._value(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise self._fail(key, f"an integer of {minimum} or more")
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

    config: AiConfig = AiConfig(
        models=ModelsConfig(
            detector_id=models.text("detector_id"),
            detector_revision=models.text("detector_revision"),
            embedder_id=models.text("embedder_id"),
            embedder_revision=models.text("embedder_revision"),
            query_encoder_id=models.text("query_encoder_id"),
            query_encoder_revision=models.text("query_encoder_revision"),
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
            margin_min=margin_min,
            margin_strong=margin_strong,
            min_photos=enrollment.integer("min_photos", minimum=2),
        ),
        query=QueryConfig(max_tokens=query.integer("max_tokens")),
    )
    logger.info("Loaded ai config from %s", config_path)
    return config
