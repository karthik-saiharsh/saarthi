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


# ---- read_crop ----


def test_a_crop_is_a_clean_jpeg_of_the_part_of_the_image() -> None:

    prepared: PreparedImage = storage.prepare_image(photo(3))  # 80 x 60
    path: str = storage.save_image(prepared)
    try:
        crop: bytes = storage.read_crop(path, (10, 5, 50, 45))
        image: Image.Image = Image.open(BytesIO(crop))
        assert image.format == "JPEG" and image.size == (40, 40)
        assert len(image.getexif()) == 0 and "icc_profile" not in image.info
        # A crop is not made larger, and a large crop is made smaller.
        assert Image.open(
            BytesIO(storage.read_crop(path, (0, 0, 80, 60), max_side=40))
        ).size == (40, 30)
        assert Image.open(
            BytesIO(storage.read_crop(path, (0, 0, 80, 60), max_side=500))
        ).size == (80, 60)
    finally:
        storage.delete_images([path])


@pytest.mark.parametrize(
    "box",
    [
        (-1, 0, 10, 10),
        (0, 0, 81, 10),
        (0, 0, 10, 61),
        (10, 10, 10, 20),
        (20, 10, 10, 20),
    ],
)
def test_a_crop_outside_the_image_is_refused(box: tuple[int, int, int, int]) -> None:
    path: str = storage.save_image(storage.prepare_image(photo(4)))
    try:
        with pytest.raises(storage.InvalidCropError):
            storage.read_crop(path, box)
    finally:
        storage.delete_images([path])


def test_a_crop_of_a_missing_file_or_a_bad_path_is_refused() -> None:
    with pytest.raises(storage.StorageNotFoundError):
        storage.read_crop("2026/01/01/" + "a" * 32 + ".jpg", (0, 0, 10, 10))
    with pytest.raises(storage.StoragePathError):
        storage.read_crop("../etc/passwd", (0, 0, 10, 10))
