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
class BackendConfig:
    max_pending: int
    token_env_var: str
    worker: WorkerConfig


def _table(raw: dict, name: str) -> dict:
    table: object = raw.get(name)
    if not isinstance(table, dict):
        raise BackendConfigError(f"Missing [{name}] table in config")
    return table


def _integer(table: dict, key: str, minimum: int) -> int:
    """Read an integer of `minimum` or more from the [worker] table."""
    if key not in table:
        raise BackendConfigError(f"Missing key worker.{key} in config")
    value: object = table[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BackendConfigError(
            f"worker.{key} must be an integer of {minimum} or more"
        )
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
        idle_wait_seconds=_integer(worker, "idle_wait_seconds", 1),
        generic_pass_every=_integer(worker, "generic_pass_every", 0),
        duplicate_window_minutes=_integer(worker, "duplicate_window_minutes", 1),
        duplicate_max_compared=_integer(worker, "duplicate_max_compared", 1),
        error_pause_seconds=_integer(worker, "error_pause_seconds", 0),
    )
    logger.info("Loaded backend config from %s", config_path)
    return BackendConfig(max_pending, token_env_var, worker_config)
