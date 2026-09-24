"""Helpers that tests of the vector features share. They use the actual db."""

import uuid

import numpy as np
import pytest

from db.api import (
    NewRegion,
    add_image,
    add_regions,
    list_embedded_region_ids,
    list_vector_ids,
)
from db.schema import (
    Image,
    ImageStatus,
    PhotoStatus,
    ReferencePhoto,
    Region,
    RegionSource,
)

TEST_PREFIX: str = "__test__/"
MODEL: str = "test-model"
DIMENSION: int = 384  # Same as vectors.dimension in config.toml


def make_image() -> Image:
    """Make an image in the status `processing`."""
    image: Image = add_image(
        f"{TEST_PREFIX}{uuid.uuid4().hex}.jpg", uuid.uuid4().hex * 2, 640, 480
    )
    Image.update(status=ImageStatus.PROCESSING.value).where(
        Image.id == image.id
    ).execute()
    return image


def make_regions(image: Image, count: int) -> list[Region]:
    new: list[NewRegion] = [
        NewRegion(i, i, i + 10, i + 10, 0.3, RegionSource.FULL, ["a bottle"])
        for i in range(count)
    ]
    return add_regions(image.id, new)


def set_status(image: Image, status: ImageStatus) -> None:
    Image.update(status=status.value).where(Image.id == image.id).execute()


def unit_vectors(count: int, seed: int = 0) -> np.ndarray:
    rng: np.random.Generator = np.random.default_rng(seed)
    raw: np.ndarray = rng.normal(size=(count, DIMENSION)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def skip_if_actual_data_has_problems() -> None:
    """Skip the test if the actual db or index needs a repair.

    A test that calls `startup_recovery` would repair real data too.
    """
    problems: list[str] = []
    if Image.select().where(Image.status == ImageStatus.PROCESSING.value).exists():
        problems.append("processing images")
    if (
        ReferencePhoto.select()
        .where(ReferencePhoto.status == PhotoStatus.PROCESSING.value)
        .exists()
    ):
        problems.append("processing reference photos")
    gap = Region.select(Region.image).where(Region.embedding_model.is_null())
    if (
        Image.select()
        .where(Image.status == ImageStatus.DONE.value, Image.id.in_(gap))
        .exists()
    ):
        problems.append("done images with a region without a vector")
    if set(list_vector_ids()) != set(list_embedded_region_ids()):
        problems.append("a difference between the index and the tables")
    if problems:
        pytest.skip("Actual database has: " + ", ".join(problems))
