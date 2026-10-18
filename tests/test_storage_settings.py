from pathlib import Path

import pytest

from storage.errors import StorageConfigError
from storage.settings import (
    DEFAULT_CONFIG_PATH,
    STORAGE_DIR,
    StorageConfig,
    load_config,
)

DEFAULT_TEXT: str = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def write(tmp_path: Path, text: str) -> Path:
    path: Path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def replace(old: str, new: str) -> str:
    assert old in DEFAULT_TEXT, f"test setup: {old!r} not in the default config"
    return DEFAULT_TEXT.replace(old, new, 1)


def test_default_config_loads() -> None:
    config: StorageConfig = load_config()
    assert config.root == (STORAGE_DIR / "uploads").resolve()
    assert config.jpeg_quality == 95
    assert (
        config.retention_days == 1
    )  # The rolling buffer of 24 hours (owner decision 2026-11-12)


def test_absolute_root_is_accepted(tmp_path: Path) -> None:
    config: StorageConfig = load_config(
        write(tmp_path, replace('root = "uploads"', f'root = "{tmp_path}"'))
    )
    assert config.root == tmp_path.resolve()


def test_relative_root_inside_storage_is_accepted(tmp_path: Path) -> None:
    config: StorageConfig = load_config(
        write(tmp_path, replace('root = "uploads"', 'root = "a/b"'))
    )
    assert config.root == (STORAGE_DIR / "a" / "b").resolve()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('root = "uploads"', 'root = "../outside"'),
        ('root = "uploads"', 'root = "a/../../outside"'),
        ('root = "uploads"', 'root = ""'),
        ('root = "uploads"', "root = 5"),
        ("max_bytes = 15728640", "max_bytes = 0"),
        ("max_bytes = 15728640", 'max_bytes = "big"'),
        ("max_pixels = 40000000", "max_pixels = true"),
        ("min_free_bytes = 1073741824", "min_free_bytes = -1"),
        ("jpeg_quality = 95", "jpeg_quality = 96"),
        ("jpeg_quality = 95", "jpeg_quality = 0"),
        ("days = 1", "days = -1"),
        ("orphan_grace_seconds = 3600", "orphan_grace_seconds = 1.5"),
    ],
)
def test_bad_values_are_refused(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(StorageConfigError):
        load_config(write(tmp_path, replace(old, new)))


@pytest.mark.parametrize(
    "table", ["location", "limits", "image", "retention", "cleanup"]
)
def test_missing_table_is_refused(tmp_path: Path, table: str) -> None:
    with pytest.raises(StorageConfigError, match=table):
        load_config(write(tmp_path, DEFAULT_TEXT.replace(f"[{table}]", f"[{table}_x]")))


def test_missing_key_names_the_key(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigError, match="limits.max_bytes"):
        load_config(write(tmp_path, replace("max_bytes = 15728640\n", "")))


def test_missing_and_broken_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigError):
        load_config(tmp_path / "none.toml")
    with pytest.raises(StorageConfigError):
        load_config(write(tmp_path, "not toml ["))
