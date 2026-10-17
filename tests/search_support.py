"""Helpers that the tests of the search share. They use the actual database."""

import uuid
from datetime import timedelta
from io import BytesIO

import numpy as np
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from db.api import gallery
from db.schema import (
    EnrolledObject,
    Image,
    ImageStatus,
    PhotoStatus,
    ReferencePhoto,
    utc_now,
)
from tests.ai_support import make_engine
from tests.db_support import DIMENSION, TEST_PREFIX

PREFIX: str = "__test__ "
TAU: float = 0.8
TAU_STRONG: float = 0.99
FRAME_SIZE: tuple[int, int] = (80, 60)
BOX: tuple[int, int, int, int] = (20, 15, 60, 45)  # 40 x 30 inside the frame


def direction(*numbers: float) -> np.ndarray:
    """A unit vector. The first numbers are given, the rest is zero."""
    padded: np.ndarray = np.zeros(DIMENSION, dtype=np.float32)
    padded[: len(numbers)] = numbers
    return padded / np.linalg.norm(padded)


BOTTLE: np.ndarray = direction(1, 0)
OTHER: np.ndarray = direction(0, 1)
WEAK: np.ndarray = direction(
    1, 0.45
)  # Cosine 0.91 to the bottle: above tau, below tau_strong


def model() -> str:
    return make_engine().embedding_model


def make_object(
    ready: bool = True,
    tracked: bool = True,
    views: np.ndarray = BOTTLE,
    name: str | None = None,
    phrase: str = "a bottle",
) -> EnrolledObject:
    """An object with one confirmed view. `ready` objects are calibrated."""
    obj: EnrolledObject = gallery.create_object(
        f"{PREFIX}{name or uuid.uuid4().hex[:8]}", phrase, tracked=tracked
    )
    photo = gallery.add_reference_photo(
        obj.id, f"{TEST_PREFIX}{uuid.uuid4().hex}.jpg", uuid.uuid4().hex * 2, 80, 60
    )
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo.id
    ).execute()  # The worker would do this. A claim could take a real photo.
    gallery.store_photo_views(
        photo.id, (1, 1, 5, 5), 0.5, [("original", views)], model()
    )
    gallery.confirm_photo(photo.id)
    if ready:
        gallery.set_calibration(obj.id, TAU, TAU_STRONG, 0.03, model())
    return obj


def store_frame(
    minutes_ago: int = 10, vector: np.ndarray = BOTTLE, in_hand: bool = False
) -> Image:
    """A `done` frame with a real file and one region (BOX) with a vector."""
    pixels: np.ndarray = np.random.default_rng(minutes_ago).integers(
        0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8
    )
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(pixels).save(buffer, "JPEG")
    prepared = storage.prepare_image(buffer.getvalue())
    path: str = storage.save_image(prepared)
    image: Image = db.add_image(path, prepared.sha256, prepared.width, prepared.height)
    Image.update(status=ImageStatus.PROCESSING.value).where(
        Image.id == image.id
    ).execute()
    region = db.add_regions(
        image.id,
        [db.NewRegion(*BOX, 0.5, db.RegionSource.FULL, ["a bottle"], in_hand=in_hand)],
    )[0]
    db.store_region_vectors([region.id], vector.reshape(1, -1), model())
    db.mark_image_done(image.id)
    Image.update(captured_at=utc_now() - timedelta(minutes=minutes_ago)).where(
        Image.id == image.id
    ).execute()
    return image


def stored_photo() -> tuple[str, str, int, int]:
    """Store a real file. Return its path, hash, width and height."""
    noise: np.ndarray = np.random.default_rng(uuid.uuid4().int % 2**32).integers(
        0, 255, (400, 400, 3), dtype=np.uint8
    )
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(noise).save(buffer, "JPEG")
    prepared = storage.prepare_image(buffer.getvalue())
    return (
        storage.save_image(prepared),
        prepared.sha256,
        prepared.width,
        prepared.height,
    )


def make_object_with_references(
    files: list[str], references: list[tuple[int, np.ndarray]]
) -> EnrolledObject:
    """A ready object. Each reference is (the side of its square box, its view vector)."""
    obj: EnrolledObject = gallery.create_object(
        f"{PREFIX}{uuid.uuid4().hex[:8]}", "a bottle"
    )
    model: str = make_engine().embedding_model
    for side, vector in references:
        path, sha, width, height = stored_photo()
        files.append(path)
        photo = gallery.add_reference_photo(obj.id, path, sha, width, height)
        ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
            ReferencePhoto.id == photo.id
        ).execute()  # The worker would do this
        gallery.store_photo_views(
            photo.id, (40, 40, 40 + side, 40 + side), 0.5, [("original", vector)], model
        )
        gallery.confirm_photo(photo.id)
    gallery.set_calibration(obj.id, TAU, TAU_STRONG, 0.03, model)
    return obj


def add_broad_boxes(
    image: Image,
    boxes: list[tuple[tuple[int, int, int, int], str, float]],
    crops: np.ndarray,
    crop_model: str,
) -> list:
    """Add boxes of the broad detector to a `done` frame, as the worker stores them: a region with `source = broad`,
    its name, a DINOv2 region vector and a SigLIP crop vector. Each box is (box, name, confidence).
    """
    status: str = db.get_image(image.id).status  # type: ignore[union-attr]
    Image.update(status=ImageStatus.PROCESSING.value).where(
        Image.id == image.id
    ).execute()
    regions = db.add_regions(
        image.id,
        [
            db.NewRegion(*box, confidence, db.RegionSource.BROAD, [], label=label)
            for box, label, confidence in boxes
        ],
    )
    ids: list[int] = [r.id for r in regions]
    db.store_region_vectors(ids, np.stack([BOTTLE] * len(ids)), model())
    db.crop_vectors.store(ids, crops, crop_model, None, None)
    Image.update(status=status).where(Image.id == image.id).execute()
    return regions
