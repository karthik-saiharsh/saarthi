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
class BackendConfig:
    max_pending: int
    token_env_var: str


def _table(raw: dict, name: str) -> dict:
    table: object = raw.get(name)
    if not isinstance(table, dict):
        raise BackendConfigError(f"Missing [{name}] table in config")
    return table


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

    logger.info("Loaded backend config from %s", config_path)
    return BackendConfig(max_pending=max_pending, token_env_var=token_env_var)
