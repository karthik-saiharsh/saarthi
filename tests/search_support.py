"""Helpers that the tests of the search share. They use the actual database."""

import uuid

import numpy as np

from db.api import gallery
from db.schema import EnrolledObject, PhotoStatus, ReferencePhoto
from tests.ai_support import make_engine
from tests.db_support import DIMENSION, TEST_PREFIX

PREFIX: str = "__test__ "
TAU: float = 0.8
TAU_STRONG: float = 0.99


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
