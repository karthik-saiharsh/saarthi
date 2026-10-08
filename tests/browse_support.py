"""Helpers that the tests of the memory browser share. They use the actual database and storage."""

import numpy as np

import db.api as db
from db.open_regions import NewOpenRegion
from db.schema import EnrolledObject, Image
from tests.search_support import BOX, make_object, store_frame


def open_vectors(count: int) -> np.ndarray:
    raw = np.random.default_rng(count).normal(
        size=(count, db.current_config().open_vector_dimension)
    )
    raw = raw.astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def full_frame(minutes_ago: int, name: str = "bottle") -> tuple[Image, EnrolledObject]:
    """A `done` frame with a real file, a registered region, a sighting, a last-seen record, an open box and room scores."""
    obj: EnrolledObject = make_object(name=name)
    frame: Image = store_frame(minutes_ago)
    region = db.get_regions(frame.id)[0]
    seen_at = db.get_image(frame.id).captured_at  # type: ignore[union-attr]
    db.sightings.add_sighting(
        obj.id, frame.id, region.id, seen_at, 0.7, 0.2, True, False
    )
    db.sightings.set_last_seen(obj.id, frame.id, BOX, seen_at, 0.7, 0.2, True, False)
    db.open_regions.add_open_regions(
        frame.id,
        [NewOpenRegion(5, 5, 30, 25, "laptop", 0.9, region.id)],
        open_vectors(1),
        "test-open-model",
    )
    db.scenes.set_room_scores(
        frame.id, "test-scene", {"a bedroom": 0.7, "a corridor": 0.2, "a kitchen": 0.1}
    )
    return frame, obj
