"""Public interface of the storage component.

Call `init_storage` once at start-up. The `backend` module calls the helpers
in the order that it chooses.
"""

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

from storage import files
from storage.errors import (
    ImageTooLargeError,
    StorageConfigError,
    StorageError,
    StorageFullError,
    StorageIOError,
    StorageNotFoundError,
    StoragePathError,
    UnsupportedImageError,
)
from storage.files import DeleteReport
from storage.imaging import PreparedImage
from storage.imaging import prepare_image as _prepare_image
from storage.settings import DEFAULT_CONFIG_PATH, StorageConfig, load_config

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "DeleteReport",
    "ImageTooLargeError",
    "PreparedImage",
    "StorageConfig",
    "StorageConfigError",
    "StorageError",
    "StorageFullError",
    "StorageIOError",
    "StorageNotFoundError",
    "StoragePathError",
    "UnsupportedImageError",
    "current_config",
    "delete_images",
    "find_missing_files",
    "find_orphan_files",
    "free_space_bytes",
    "init_storage",
    "list_stored_paths",
    "prepare_image",
    "read_image",
    "save_image",
]

# The open configuration. `init_storage` sets it.
_config: StorageConfig | None = None


def _require_config() -> StorageConfig:
    if _config is None:
        raise StorageConfigError("Storage is not ready. Call init_storage first")
    return _config


def init_storage(config_path: Path = DEFAULT_CONFIG_PATH) -> StorageConfig:
    """Read the config, prepare the root folder and remove stale temporary files.

    Return the config. You can call this function more than once.
    """
    global _config
    assert isinstance(config_path, Path), "config_path must be a Path"
    try:
        config: StorageConfig = load_config(config_path)
    except StorageConfigError as error:
        logger.error("Invalid config: %s", error)
        raise
    files.ensure_root(config)
    files.remove_stale_temp_files(config)
    _config = config
    logger.info(
        "Storage ready at %s, %d bytes free", config.root, files.free_space(config)
    )
    return config


def current_config() -> StorageConfig:
    """Return the config that `init_storage` loaded (limits, retention days, ...)."""
    return _require_config()


def prepare_image(data: bytes) -> PreparedImage:
    """Check an upload and make the JPEG without metadata. Nothing is written to disk."""
    return _prepare_image(data, _require_config())


def save_image(prepared: PreparedImage) -> str:
    """Write a prepared image to disk. Return its path relative to the storage root."""
    assert isinstance(prepared, PreparedImage), "prepared must be a PreparedImage"
    assert (
        hashlib.sha256(prepared.data).hexdigest() == prepared.sha256
    ), "hash does not match"
    return files.write_file(_require_config(), prepared.data)


def read_image(relative_path: str) -> bytes:
    """Return the bytes of a stored image."""
    return files.read_file(_require_config(), relative_path)


def delete_images(relative_paths: list[str]) -> DeleteReport:
    """Delete stored images. A path that is not valid stops the call before any delete."""
    return files.delete_files(_require_config(), relative_paths)


def list_stored_paths() -> list[str]:
    """Return the relative paths of all stored images, sorted."""
    return files.list_paths(_require_config())


def find_orphan_files(
    known_paths: Iterable[str], min_age_seconds: int | None = None
) -> list[str]:
    """Return stored files that the database does not know.

    `known_paths` are the paths in the database. Files younger than
    `min_age_seconds` are left out. The default is `orphan_grace_seconds`.
    """
    return files.find_orphans(_require_config(), known_paths, min_age_seconds)


def find_missing_files(known_paths: Iterable[str]) -> list[str]:
    """Return the paths of the database that have no file on disk."""
    return files.find_missing(_require_config(), known_paths)


def free_space_bytes() -> int:
    """Return the free space on the disk of the storage root, in bytes."""
    return files.free_space(_require_config())
