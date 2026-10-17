"""The answers of the carer to "Is this your object?" (plan part D2).

The backend shows the carer boxes of recent frames that look like an object. The carer answers:

- "yes": the backend makes the frame a reference photo of the object (`db.gallery.add_confirmed_photo`).
  This module keeps the answer, so the box is not shown again.
- "no": the box is a negative of the object. The matching mode `keys` refuses a box that is closer to a
  negative than to the object.

The crop vectors of the box are copied into the row (as the views of the gallery), so a negative stays
when retention deletes the frame. These helpers change rows only. They run no model and read no file.
"""

import logging

import numpy as np

from db.errors import InvalidRecordError
from db.schema import EnrolledObject, ReviewAnswer
from db.support import translate_errors

logger: logging.Logger = logging.getLogger(__name__)

ANSWERS: tuple[str, ...] = ("yes", "no")
NORM_TOLERANCE: float = 1e-3


def _check_id(value: int, name: str) -> None:
    assert isinstance(value, int) and value > 0, f"{name} must be a positive integer"


def _to_bytes(vector: np.ndarray) -> bytes:
    """Check a vector and return its bytes. Raise `InvalidRecordError` for a bad value."""
    assert isinstance(vector, np.ndarray) and vector.ndim == 1, "a vector must be 1-D"
    numbers: np.ndarray = vector.astype(np.float32)
    if not np.isfinite(numbers).all():
        raise InvalidRecordError("A vector contains a value that is not finite")
    if abs(float(np.linalg.norm(numbers)) - 1.0) > NORM_TOLERANCE:
        raise InvalidRecordError("A vector must have unit length")
    return numbers.tobytes()


def add_answer(
    object_id: int,
    region_id: int,
    answer: str,
    crop_vector: np.ndarray,
    crop_model: str,
    key_vector: np.ndarray,
    key_model: str,
) -> ReviewAnswer:
    """Keep the answer of the carer for one box. Raise `InvalidRecordError` if the object or the region does not exist."""
    _check_id(object_id, "object_id")
    _check_id(region_id, "region_id")
    assert answer in ANSWERS, f"answer must be one of {ANSWERS}"
    assert isinstance(crop_model, str) and crop_model, "crop model needed"
    assert isinstance(key_model, str) and key_model, "key model needed"
    with translate_errors("add_answer"):
        row: ReviewAnswer = ReviewAnswer.create(
            enrolled_object=object_id,
            region=region_id,
            answer=answer,
            crop_vector=_to_bytes(crop_vector),
            key_vector=_to_bytes(key_vector),
            crop_model=crop_model,
            key_model=key_model,
        )
    logger.info("Answer %s for object %d is kept", answer, object_id)
    return row


def answered_region_ids(object_id: int) -> set[int]:
    """Return the regions that the carer answered for this object. A region that retention deleted is not in the set."""
    _check_id(object_id, "object_id")
    with translate_errors("answered_region_ids"):
        return {
            row.region_id
            for row in ReviewAnswer.select(ReviewAnswer.region).where(
                ReviewAnswer.enrolled_object == object_id,
                ReviewAnswer.region.is_null(False),
            )
        }


def load_negatives(
    object_id: int, crop_model: str, key_model: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the crop vectors and the keys of the "no" answers of an object, for these models. Each can have 0 rows."""
    _check_id(object_id, "object_id")
    with translate_errors("load_negatives"):
        rows: list[ReviewAnswer] = list(
            ReviewAnswer.select(ReviewAnswer.crop_vector, ReviewAnswer.key_vector)
            .join(EnrolledObject)
            .where(
                ReviewAnswer.enrolled_object == object_id,
                ReviewAnswer.answer == "no",
                ReviewAnswer.crop_model == crop_model,
                ReviewAnswer.key_model == key_model,
            )
            .order_by(ReviewAnswer.id)
        )
    crops: list[np.ndarray] = [
        np.frombuffer(r.crop_vector, dtype=np.float32) for r in rows
    ]
    keys: list[np.ndarray] = [
        np.frombuffer(r.key_vector, dtype=np.float32) for r in rows
    ]
    if not rows:
        return np.empty((0, 0), np.float32), np.empty((0, 0), np.float32)
    return np.stack(crops), np.stack(keys)


__all__ = ["ANSWERS", "add_answer", "answered_region_ids", "load_negatives"]
