"""Tests of the public storage helpers. They use the actual configured root.

Each test removes the files that it made.
"""

from collections.abc import Iterator
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import storage.api as storage
from storage.api import PreparedImage, StorageConfigError


def photo(seed: int = 0) -> bytes:
    rng: np.random.Generator = np.random.default_rng(seed)
    buffer: BytesIO = BytesIO()
    Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
        buffer, "PNG"
    )
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def ready() -> Iterator[list[str]]:
    """Open the storage. Give the test a list for the paths that it makes."""
    storage.init_storage()
    made: list[str] = []
    yield made
    if made:
        storage.delete_images(made)


def test_init_returns_the_config_and_can_run_twice() -> None:
    first = storage.init_storage()
    assert first.root.is_dir() and storage.init_storage() == first


def test_prepare_save_read_delete_round_trip(ready: list[str]) -> None:
    prepared: PreparedImage = storage.prepare_image(photo())
    path: str = storage.save_image(prepared)
    ready.append(path)
    assert path in storage.list_stored_paths()
    assert storage.read_image(path) == prepared.data
    assert path not in storage.find_orphan_files([path], min_age_seconds=0)
    assert path in storage.find_orphan_files([], min_age_seconds=0)
    assert storage.find_missing_files([path]) == []
    report = storage.delete_images([path])
    ready.clear()
    assert report.deleted == [path]
    assert storage.find_missing_files([path]) == [path]


def test_save_refuses_data_that_does_not_match_its_hash() -> None:
    prepared: PreparedImage = storage.prepare_image(photo())
    forged: PreparedImage = PreparedImage(prepared.data + b"x", prepared.sha256, 80, 60)
    with pytest.raises(AssertionError):
        storage.save_image(forged)


def test_free_space_is_a_positive_number() -> None:
    assert storage.free_space_bytes() > 0


def test_helpers_need_init_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "_config", None)
    with pytest.raises(StorageConfigError):
        storage.list_stored_paths()
    with pytest.raises(StorageConfigError):
        storage.prepare_image(photo())


def test_current_config_is_the_loaded_config() -> None:
    assert storage.current_config() == storage.init_storage()


def test_current_config_needs_init_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "_config", None)
    with pytest.raises(StorageConfigError):
        storage.current_config()
