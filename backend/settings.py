"""Settings of the backend component. They come from `config.toml`."""

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from backend.errors import BackendConfigError

logger: logging.Logger = logging.getLogger(__name__)

BACKEND_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = BACKEND_DIR / "config.toml"

_ENV_NAME: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class WorkerConfig:
    idle_wait_seconds: int
    generic_pass_every: int
    duplicate_window_minutes: int
    duplicate_max_compared: int
    error_pause_seconds: int


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    worker_stop_seconds: int


@dataclass(frozen=True)
class MaintenanceConfig:
    interval_minutes: int
    orphan_delete_limit: int
    retention_batch: int


@dataclass(frozen=True)
class MemoryConfig:
    hysteresis_window_seconds: int
    hysteresis_hits: int


@dataclass(frozen=True)
class SearchConfig:
    stale_after_hours: int
    window_minutes: int
    max_scan_hours: int
    max_regions_per_window: int
    max_frames: int


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
    )
    server: dict = _table(raw, "server")
    host: object = server.get("host")
    if not isinstance(host, str) or not host.strip():
        raise BackendConfigError("server.host must be a non-empty string")
    server_config: ServerConfig = ServerConfig(
        host=host,
        port=_integer(server, "server", "port", 1, 65535),
        worker_stop_seconds=_integer(server, "server", "worker_stop_seconds", 1),
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
    )
    search: dict = _table(raw, "search")
    search_config: SearchConfig = SearchConfig(
        stale_after_hours=_integer(search, "search", "stale_after_hours", 1),
        window_minutes=_integer(search, "search", "window_minutes", 1),
        max_scan_hours=_integer(search, "search", "max_scan_hours", 1),
        max_regions_per_window=_integer(search, "search", "max_regions_per_window", 1),
        max_frames=_integer(search, "search", "max_frames", 1),
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
    )
