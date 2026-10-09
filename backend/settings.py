"""Settings of the backend component. They come from `config.toml`."""

import logging
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.errors import BackendConfigError

logger: logging.Logger = logging.getLogger(__name__)

BACKEND_DIR: Path = Path(__file__).resolve().parent
PROJECT_DIR: Path = BACKEND_DIR.parent
# These environment variables replace `server.tls_certfile` and `server.tls_keyfile`.
TLS_CERT_VARIABLE: str = "SAARTHI_TLS_CERT"
TLS_KEY_VARIABLE: str = "SAARTHI_TLS_KEY"
DEFAULT_CONFIG_PATH: Path = BACKEND_DIR / "config.toml"

_ENV_NAME: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class WorkerConfig:
    idle_wait_seconds: int
    generic_pass_every: int
    duplicate_window_minutes: int
    duplicate_max_compared: int
    error_pause_seconds: int
    open_link_iou: float


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    worker_stop_seconds: int
    # Paths of the TLS certificate and key. Empty: the server uses HTTP.
    # A relative path starts at the project folder.
    tls_certfile: str = ""
    tls_keyfile: str = ""

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_certfile)

    def tls_paths(self) -> tuple[Path, Path]:
        """Return the existing certificate file and key file. Only call it if TLS is on."""
        assert self.tls_enabled, "TLS is off"
        paths: list[Path] = []
        for name, value in (
            ("tls_certfile", self.tls_certfile),
            ("tls_keyfile", self.tls_keyfile),
        ):
            path: Path = Path(value)
            if not path.is_absolute():
                path = PROJECT_DIR / path
            if not path.is_file():
                raise BackendConfigError(f"server.{name} is not a file: {path}")
            paths.append(path)
        return paths[0], paths[1]


@dataclass(frozen=True)
class MaintenanceConfig:
    interval_minutes: int
    orphan_delete_limit: int
    retention_batch: int


@dataclass(frozen=True)
class MemoryConfig:
    hysteresis_window_seconds: int
    hysteresis_hits: int
    verify_weak: bool = True
    verify_max_references: int = 3


@dataclass(frozen=True)
class SearchConfig:
    stale_after_hours: int
    window_minutes: int
    max_scan_hours: int
    max_regions_per_window: int
    max_frames: int
    crop_padding: float
    verify_weak: bool = True
    verify_max_references: int = 3


@dataclass(frozen=True)
class QueryConfig:
    matched_min_score: float
    matched_min_gap: float
    suggest_min_score: float
    max_candidates: int
    max_chars: int


@dataclass(frozen=True)
class DescribeConfig:
    neighbour_window_seconds: int
    neighbour_max_frames: int
    room_min_frames: int
    room_agree_fraction: float
    room_min_score: float
    surface_min_score: float
    # IANA name of the time zone for the part of the day. Empty: the time zone of the server.
    time_zone: str
    # A sighting at least this old gets a part of the day ("yesterday evening") in the sentence.
    day_part_min_hours: int
    write: bool
    name_min_confidence: float
    max_words: int
    safe_words: tuple[str, ...]


@dataclass(frozen=True)
class OpenSearchConfig:
    label_min_score: float
    label_tier: float
    visual_min_probability: float
    visual_margin: float
    neutral_phrases: tuple[str, ...]
    rewrite_min_words: int
    rewrite_min_score: float
    max_scan_hours: int
    max_boxes: int
    mode: str  # "classic" or "verified"
    verify_top_boxes: int
    verify_max_frames: int
    verify_min_score: float
    verify_min_iou: float


OPEN_SEARCH_MODES: tuple[str, ...] = ("classic", "verified")


@dataclass(frozen=True)
class VoiceConfig:
    min_seconds: float
    silence_rms: float


@dataclass(frozen=True)
class EnrollmentConfig:
    max_photos_per_object: int
    max_objects: int


@dataclass(frozen=True)
class BackendConfig:
    max_pending: int
    token_env_var: str
    worker: WorkerConfig
    server: ServerConfig
    maintenance: MaintenanceConfig
    enrollment: EnrollmentConfig
    memory: MemoryConfig
    search: SearchConfig
    query: QueryConfig
    describe: DescribeConfig
    voice: VoiceConfig
    open_search: OpenSearchConfig


def _table(raw: dict, name: str) -> dict:
    table: object = raw.get(name)
    if not isinstance(table, dict):
        raise BackendConfigError(f"Missing [{name}] table in config")
    return table


def _integer(
    table: dict, name: str, key: str, minimum: int, maximum: int | None = None
) -> int:
    """Read an integer from a table. Raise `BackendConfigError` if it is missing or out of range."""
    if key not in table:
        raise BackendConfigError(f"Missing key {name}.{key} in config")
    value: object = table[key]
    too_big: bool = maximum is not None and isinstance(value, int) and value > maximum
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or too_big
    ):
        limit: str = (
            f"between {minimum} and {maximum}" if maximum else f"{minimum} or more"
        )
        raise BackendConfigError(f"{name}.{key} must be an integer {limit}")
    return value


def _boolean(table: dict, name: str, key: str) -> bool:
    """Read true or false from a table. Raise `BackendConfigError` if it is missing or not a boolean."""
    if key not in table:
        raise BackendConfigError(f"Missing key {name}.{key} in config")
    value: object = table[key]
    if not isinstance(value, bool):
        raise BackendConfigError(f"{name}.{key} must be true or false")
    return value


def _number(table: dict, name: str, key: str, low: float, high: float) -> float:
    """Read a number from a table. Raise `BackendConfigError` if it is missing or out of range."""
    if key not in table:
        raise BackendConfigError(f"Missing key {name}.{key} in config")
    value: object = table[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BackendConfigError(f"{name}.{key} must be a number")
    if not low <= value <= high:
        raise BackendConfigError(f"{name}.{key} must be between {low} and {high}")
    return float(value)


def _boolean(table: dict, name: str, key: str) -> bool:
    """Read a boolean from a table. Raise `BackendConfigError` if it is missing or not a boolean."""
    if key not in table:
        raise BackendConfigError(f"Missing key {name}.{key} in config")
    value: object = table[key]
    if not isinstance(value, bool):
        raise BackendConfigError(f"{name}.{key} must be true or false")
    return value


def _words(table: dict, name: str, key: str) -> tuple[str, ...]:
    """Read a list of single lower-case words. The list can be empty."""
    if key not in table:
        raise BackendConfigError(f"Missing key {name}.{key} in config")
    value: object = table[key]
    if not isinstance(value, list) or not all(
        isinstance(w, str) and re.fullmatch(r"[a-z']+", w) for w in value
    ):
        raise BackendConfigError(
            f"{name}.{key} must be a list of single lower-case words"
        )
    return tuple(value)


def _time_zone(table: dict) -> str:
    """Read `describe.time_zone`. It is an IANA name such as `Asia/Kolkata`, or empty for the zone of the server."""
    value: object = table.get("time_zone")
    if not isinstance(value, str):
        raise BackendConfigError("describe.time_zone must be a string")
    name: str = value.strip()
    if name:
        try:
            ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise BackendConfigError(
                f"describe.time_zone is not a known time zone: {name}"
            ) from error
    return name


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> BackendConfig:
    """Read and check the config file. Raise `BackendConfigError` if it is not correct."""
    assert isinstance(config_path, Path), "config_path must be a Path"
    try:
        with config_path.open("rb") as file:
            raw: dict = tomllib.load(file)
    except FileNotFoundError as error:
        raise BackendConfigError(f"Config file not found: {config_path}") from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BackendConfigError(
            f"Cannot read config {config_path}: {error}"
        ) from error

    ingest: dict = _table(raw, "ingest")
    api: dict = _table(raw, "api")
    for table, name, key in (
        (ingest, "ingest", "max_pending"),
        (api, "api", "token_env_var"),
    ):
        if key not in table:
            raise BackendConfigError(f"Missing key {name}.{key} in config")

    max_pending: object = ingest["max_pending"]
    if (
        isinstance(max_pending, bool)
        or not isinstance(max_pending, int)
        or max_pending < 1
    ):
        raise BackendConfigError("ingest.max_pending must be an integer of 1 or more")
    token_env_var: object = api["token_env_var"]
    if not isinstance(token_env_var, str) or not _ENV_NAME.fullmatch(token_env_var):
        raise BackendConfigError(
            "api.token_env_var must be a valid environment variable name"
        )

    worker: dict = _table(raw, "worker")
    worker_config: WorkerConfig = WorkerConfig(
        idle_wait_seconds=_integer(worker, "worker", "idle_wait_seconds", 1),
        generic_pass_every=_integer(worker, "worker", "generic_pass_every", 0),
        duplicate_window_minutes=_integer(
            worker, "worker", "duplicate_window_minutes", 1
        ),
        duplicate_max_compared=_integer(worker, "worker", "duplicate_max_compared", 1),
        error_pause_seconds=_integer(worker, "worker", "error_pause_seconds", 0),
        open_link_iou=_number(worker, "worker", "open_link_iou", 0.01, 1.0),
    )
    server: dict = _table(raw, "server")
    host: object = server.get("host")
    if not isinstance(host, str) or not host.strip():
        raise BackendConfigError("server.host must be a non-empty string")
    tls: dict[str, str] = {}
    for key in ("tls_certfile", "tls_keyfile"):
        value: object = server.get(key)
        if not isinstance(value, str):
            raise BackendConfigError(f"server.{key} must be a string")
        tls[key] = value.strip()
    # The script `./start` sets these variables, so the config file stays the same on every machine.
    for key, variable in (
        ("tls_certfile", TLS_CERT_VARIABLE),
        ("tls_keyfile", TLS_KEY_VARIABLE),
    ):
        override: str = os.environ.get(variable, "").strip()
        if override:
            tls[key] = override
    if bool(tls["tls_certfile"]) != bool(tls["tls_keyfile"]):
        raise BackendConfigError(
            "server.tls_certfile and server.tls_keyfile must both be set, or both be empty"
        )
    server_config: ServerConfig = ServerConfig(
        host=host,
        port=_integer(server, "server", "port", 1, 65535),
        worker_stop_seconds=_integer(server, "server", "worker_stop_seconds", 1),
        tls_certfile=tls["tls_certfile"],
        tls_keyfile=tls["tls_keyfile"],
    )
    maintenance: dict = _table(raw, "maintenance")
    maintenance_config: MaintenanceConfig = MaintenanceConfig(
        interval_minutes=_integer(maintenance, "maintenance", "interval_minutes", 1),
        orphan_delete_limit=_integer(
            maintenance, "maintenance", "orphan_delete_limit", 0
        ),
        retention_batch=_integer(maintenance, "maintenance", "retention_batch", 1),
    )
    enrollment: dict = _table(raw, "enrollment")
    enrollment_config: EnrollmentConfig = EnrollmentConfig(
        max_photos_per_object=_integer(
            enrollment, "enrollment", "max_photos_per_object", 1
        ),
        max_objects=_integer(enrollment, "enrollment", "max_objects", 1),
    )
    memory: dict = _table(raw, "memory")
    memory_config: MemoryConfig = MemoryConfig(
        hysteresis_window_seconds=_integer(
            memory, "memory", "hysteresis_window_seconds", 1
        ),
        hysteresis_hits=_integer(memory, "memory", "hysteresis_hits", 1),
        verify_weak=_boolean(memory, "memory", "verify_weak"),
        verify_max_references=_integer(memory, "memory", "verify_max_references", 1),
    )
    search: dict = _table(raw, "search")
    search_config: SearchConfig = SearchConfig(
        stale_after_hours=_integer(search, "search", "stale_after_hours", 1),
        window_minutes=_integer(search, "search", "window_minutes", 1),
        max_scan_hours=_integer(search, "search", "max_scan_hours", 1),
        max_regions_per_window=_integer(search, "search", "max_regions_per_window", 1),
        max_frames=_integer(search, "search", "max_frames", 1),
        crop_padding=_number(search, "search", "crop_padding", 0.0, 5.0),
        verify_weak=_boolean(search, "search", "verify_weak"),
        verify_max_references=_integer(search, "search", "verify_max_references", 1),
    )
    query: dict = _table(raw, "query")
    query_config: QueryConfig = QueryConfig(
        matched_min_score=_number(query, "query", "matched_min_score", -1.0, 1.0),
        matched_min_gap=_number(query, "query", "matched_min_gap", 0.0, 2.0),
        suggest_min_score=_number(query, "query", "suggest_min_score", -1.0, 1.0),
        max_candidates=_integer(query, "query", "max_candidates", 1),
        max_chars=_integer(query, "query", "max_chars", 1),
    )
    if query_config.suggest_min_score > query_config.matched_min_score:
        raise BackendConfigError(
            "query.suggest_min_score must not be above query.matched_min_score"
        )
    describe: dict = _table(raw, "describe")
    describe_config: DescribeConfig = DescribeConfig(
        neighbour_window_seconds=_integer(
            describe, "describe", "neighbour_window_seconds", 1
        ),
        neighbour_max_frames=_integer(describe, "describe", "neighbour_max_frames", 1),
        room_min_frames=_integer(describe, "describe", "room_min_frames", 1),
        room_agree_fraction=_number(
            describe, "describe", "room_agree_fraction", 0.0, 1.0
        ),
        room_min_score=_number(describe, "describe", "room_min_score", 0.0, 1.0),
        surface_min_score=_number(describe, "describe", "surface_min_score", 0.0, 1.0),
        time_zone=_time_zone(describe),
        day_part_min_hours=_integer(describe, "describe", "day_part_min_hours", 1, 24),
        write=_boolean(describe, "describe", "write"),
        name_min_confidence=_number(
            describe, "describe", "name_min_confidence", 0.0, 1.0
        ),
        max_words=_integer(describe, "describe", "max_words", 3, 100),
        safe_words=_words(describe, "describe", "safe_words"),
    )
    voice: dict = _table(raw, "voice")
    voice_config: VoiceConfig = VoiceConfig(
        min_seconds=_number(voice, "voice", "min_seconds", 0.1, 10.0),
        silence_rms=_number(voice, "voice", "silence_rms", 0.0, 1.0),
    )
    open_search: dict = _table(raw, "open_search")
    neutral: object = open_search.get("neutral_phrases")
    if (
        not isinstance(neutral, list)
        or not neutral
        or not all(isinstance(p, str) and p.strip() for p in neutral)
    ):
        raise BackendConfigError(
            "open_search.neutral_phrases must be a list of words that is not empty"
        )
    mode: object = open_search.get("mode")
    if mode not in OPEN_SEARCH_MODES:
        raise BackendConfigError(
            f"open_search.mode must be one of {', '.join(OPEN_SEARCH_MODES)}"
        )
    open_search_config: OpenSearchConfig = OpenSearchConfig(
        label_min_score=_number(
            open_search, "open_search", "label_min_score", 0.0, 1.0
        ),
        label_tier=_number(open_search, "open_search", "label_tier", 0.0, 1.0),
        visual_min_probability=_number(
            open_search, "open_search", "visual_min_probability", 0.0, 1.0
        ),
        visual_margin=_number(open_search, "open_search", "visual_margin", 0.0, 1.0),
        neutral_phrases=tuple(p.strip() for p in neutral),
        rewrite_min_words=_integer(
            open_search, "open_search", "rewrite_min_words", 1, 20
        ),
        rewrite_min_score=_number(
            open_search, "open_search", "rewrite_min_score", 0.0, 1.0
        ),
        max_scan_hours=_integer(open_search, "open_search", "max_scan_hours", 1),
        max_boxes=_integer(open_search, "open_search", "max_boxes", 1, 20000),
        mode=str(mode),
        verify_top_boxes=_integer(
            open_search, "open_search", "verify_top_boxes", 1, 200
        ),
        verify_max_frames=_integer(
            open_search, "open_search", "verify_max_frames", 1, 50
        ),
        verify_min_score=_number(
            open_search, "open_search", "verify_min_score", 0.0, 1.0
        ),
        verify_min_iou=_number(open_search, "open_search", "verify_min_iou", 0.0, 1.0),
    )
    logger.info("Loaded backend config from %s", config_path)
    return BackendConfig(
        max_pending,
        token_env_var,
        worker_config,
        server_config,
        maintenance_config,
        enrollment_config,
        memory_config,
        search_config,
        query_config,
        describe_config,
        voice_config,
        open_search_config,
    )
