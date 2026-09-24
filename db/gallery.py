"""The gallery: enrolled objects, their reference photos and their views.

A carer gives an object a name, a category phrase and some photos. The worker
finds the object in each photo and makes vectors (the views). The carer
confirms each crop. Then the object is calibrated and becomes `ready`.

The gallery is small, so the vectors are stored in SQLite as `float32` bytes.
These helpers change rows only. They do not read files, run models, or open
the vector index of the archive.
"""

import logging

import numpy as np

from db.errors import (
    DuplicateObjectError,
    InvalidRecordError,
    InvalidStateError,
)
from db.schema import (
    EnrolledObject,
    ObjectStatus,
    ObjectView,
    PhotoStatus,
    ReferencePhoto,
    database_proxy,
    utc_now,
)
from db.support import translate_errors

logger: logging.Logger = logging.getLogger(__name__)

# Longest `status_detail` text that the database stores.
MAX_STATUS_DETAIL_LENGTH: int = 500
# How far the length of a vector may be from 1.0.
NORM_TOLERANCE: float = 1e-3

# The number of numbers in one vector. `db.api.init_db` sets it. None: no check.
_expected_dimension: int | None = None


def configure(dimension: int) -> None:
    """Set the size of the vectors that the gallery accepts."""
    global _expected_dimension
    assert isinstance(dimension, int) and dimension > 0, "dimension must be positive"
    _expected_dimension = dimension


def _check_id(value: int, name: str) -> None:
    assert isinstance(value, int) and value > 0, f"{name} must be a positive integer"


def _vector_to_bytes(vector: np.ndarray) -> bytes:
    """Check a vector and return its bytes. Raise `InvalidRecordError` for a bad value."""
    assert (
        isinstance(vector, np.ndarray) and vector.ndim == 1
    ), "a view must be a 1-D array"
    if _expected_dimension is not None and vector.shape[0] != _expected_dimension:
        raise InvalidRecordError(
            f"A view needs {_expected_dimension} numbers, not {vector.shape[0]}"
        )
    numbers: np.ndarray = vector.astype(np.float32)
    if not np.isfinite(numbers).all():
        raise InvalidRecordError("A view contains a value that is not finite")
    if abs(float(np.linalg.norm(numbers)) - 1.0) > NORM_TOLERANCE:
        raise InvalidRecordError("A view must have unit length")
    return numbers.tobytes()


# ---- Objects ----


def create_object(
    name: str,
    category_phrase: str,
    synonyms: list[str] | None = None,
    tracked: bool = True,
) -> EnrolledObject:
    """Add an object with the status `draft`. Raise `DuplicateObjectError` if the name exists.

    The caller checks the words (the component that owns the detector prompt).
    """
    assert isinstance(name, str) and name.strip(), "name must be a non-empty string"
    assert isinstance(category_phrase, str) and category_phrase.strip(), "phrase needed"
    assert synonyms is None or all(isinstance(s, str) and s for s in synonyms)
    assert isinstance(tracked, bool), "tracked must be a bool"
    with translate_errors("create_object", duplicate=DuplicateObjectError):
        obj: EnrolledObject = EnrolledObject.create(
            name=name,
            category_phrase=category_phrase,
            synonyms=list(synonyms or []),
            tracked=tracked,
        )
    logger.info("Created object %d", obj.id)
    return obj


def get_object(object_id: int) -> EnrolledObject | None:
    """Return the object with this id, or None."""
    _check_id(object_id, "object_id")
    with translate_errors("get_object"):
        return EnrolledObject.get_or_none(EnrolledObject.id == object_id)


def list_objects() -> list[EnrolledObject]:
    """Return all objects, oldest first."""
    with translate_errors("list_objects"):
        return list(EnrolledObject.select().order_by(EnrolledObject.id))


def count_objects() -> int:
    """Return the number of objects."""
    with translate_errors("count_objects"):
        return EnrolledObject.select().count()


def delete_object(object_id: int) -> list[str]:
    """Delete an object with its photos and views. Return the file paths of its photos.

    The caller deletes the files afterwards. Rows go first, files last.
    An unknown id returns an empty list. A photo that the worker holds
    (`processing`) stops the delete: raise `InvalidStateError`.
    """
    _check_id(object_id, "object_id")
    with translate_errors("delete_object"), database_proxy.atomic("IMMEDIATE"):
        photos: list[ReferencePhoto] = list(
            ReferencePhoto.select().where(ReferencePhoto.enrolled_object == object_id)
        )
        if any(p.status == PhotoStatus.PROCESSING.value for p in photos):
            raise InvalidStateError("A photo of this object is being analysed")
        EnrolledObject.delete().where(EnrolledObject.id == object_id).execute()
    paths: list[str] = [p.file_path for p in photos]
    logger.info("Deleted object %d with %d photo(s)", object_id, len(paths))
    return paths


# ---- Reference photos ----


def add_reference_photo(
    object_id: int, file_path: str, sha256: str, width: int, height: int
) -> ReferencePhoto:
    """Add a photo with the status `pending`.

    Raise `InvalidStateError` if the object does not exist. Raise
    `DuplicateImageError` if the object has this picture already (same hash),
    or the path is used.
    """
    _check_id(object_id, "object_id")
    assert isinstance(file_path, str) and file_path, "file_path must be a string"
    assert isinstance(sha256, str) and len(sha256) == 64, "sha256 must have 64 chars"
    assert isinstance(width, int) and width > 0, "width must be positive"
    assert isinstance(height, int) and height > 0, "height must be positive"
    with translate_errors("add_reference_photo"):
        if not EnrolledObject.select().where(EnrolledObject.id == object_id).exists():
            raise InvalidStateError(f"Object {object_id} does not exist")
        photo: ReferencePhoto = ReferencePhoto.create(
            enrolled_object=object_id,
            file_path=file_path,
            sha256=sha256,
            width=width,
            height=height,
        )
    logger.debug("Added photo %d to object %d", photo.id, object_id)
    return photo


def get_photo(photo_id: int) -> ReferencePhoto | None:
    """Return the photo with this id, or None."""
    _check_id(photo_id, "photo_id")
    with translate_errors("get_photo"):
        return ReferencePhoto.get_or_none(ReferencePhoto.id == photo_id)


def list_photos(object_id: int) -> list[ReferencePhoto]:
    """Return the photos of an object, oldest first."""
    _check_id(object_id, "object_id")
    with translate_errors("list_photos"):
        return list(
            ReferencePhoto.select()
            .where(ReferencePhoto.enrolled_object == object_id)
            .order_by(ReferencePhoto.id)
        )


def count_photos_by_status(object_id: int) -> dict[PhotoStatus, int]:
    """Return how many photos of an object have each status."""
    counts: dict[PhotoStatus, int] = {status: 0 for status in PhotoStatus}
    for photo in list_photos(object_id):
        counts[PhotoStatus(photo.status)] += 1
    return counts


def claim_next_reference_photo() -> ReferencePhoto | None:
    """Set the oldest `pending` photo to `processing` and return it, or None.

    The read and the update run in one write transaction, as for the frames.
    """
    with (
        translate_errors("claim_next_reference_photo"),
        database_proxy.atomic("IMMEDIATE"),
    ):
        photo: ReferencePhoto | None = (
            ReferencePhoto.select()
            .where(ReferencePhoto.status == PhotoStatus.PENDING.value)
            .order_by(ReferencePhoto.id)
            .first()
        )
        if photo is None:
            return None
        changed: int = (
            ReferencePhoto.update(status=PhotoStatus.PROCESSING.value)
            .where(
                ReferencePhoto.id == photo.id,
                ReferencePhoto.status == PhotoStatus.PENDING.value,
            )
            .execute()
        )
        assert changed == 1, "claim must change exactly one row"
        photo.status = PhotoStatus.PROCESSING.value
    logger.info("Claimed reference photo %d", photo.id)
    return photo


def _change_status(
    photo_id: int, expected: PhotoStatus, **changes: object
) -> ReferencePhoto:
    """Change a photo with one conditional UPDATE. Raise `InvalidStateError` if the status differs."""
    changed: int = (
        ReferencePhoto.update(**changes)
        .where(ReferencePhoto.id == photo_id, ReferencePhoto.status == expected.value)
        .execute()
    )
    if changed != 1:
        raise InvalidStateError(
            f"Photo {photo_id} does not exist or is not in status {expected.value}"
        )
    return ReferencePhoto.get_by_id(photo_id)


def store_photo_views(
    photo_id: int,
    box: tuple[int, int, int, int],
    det_score: float,
    views: list[tuple[str, np.ndarray]],
    embedding_model: str,
) -> ReferencePhoto:
    """Save where the object is in a `processing` photo, and its views. The photo is `located`.

    `views` is a list of (augmentation name, vector). Do all steps or none.
    """
    _check_id(photo_id, "photo_id")
    assert len(box) == 4 and all(
        isinstance(v, int) for v in box
    ), "box needs 4 integers"
    assert isinstance(det_score, float), "det_score must be a float"
    assert views, "views must not be empty"
    assert isinstance(embedding_model, str) and embedding_model, "model name needed"
    rows: list[tuple[str, bytes]] = [
        (name, _vector_to_bytes(vector)) for name, vector in views
    ]

    with translate_errors("store_photo_views"), database_proxy.atomic():
        photo: ReferencePhoto = _change_status(
            photo_id,
            PhotoStatus.PROCESSING,
            status=PhotoStatus.LOCATED.value,
            x0=box[0],
            y0=box[1],
            x1=box[2],
            y1=box[3],
            det_score=det_score,
        )
        for name, data in rows:
            ObjectView.create(
                enrolled_object=photo.enrolled_object_id,
                photo=photo_id,
                vector=data,
                embedding_model=embedding_model,
                augmentation=name,
            )
    logger.info("Photo %d is located with %d view(s)", photo_id, len(rows))
    return photo


def mark_photo_failed(photo_id: int, reason: str) -> ReferencePhoto:
    """Set a `processing` photo to `failed`. `reason` must not contain personal data."""
    _check_id(photo_id, "photo_id")
    assert isinstance(reason, str) and reason, "reason must be a non-empty string"
    with translate_errors("mark_photo_failed"):
        photo: ReferencePhoto = _change_status(
            photo_id,
            PhotoStatus.PROCESSING,
            status=PhotoStatus.FAILED.value,
            status_detail=reason[:MAX_STATUS_DETAIL_LENGTH],
        )
    logger.info("Photo %d failed", photo_id)
    return photo


def confirm_photo(photo_id: int) -> ReferencePhoto:
    """Set a `located` photo to `confirmed`. The carer accepted the crop."""
    _check_id(photo_id, "photo_id")
    with translate_errors("confirm_photo"):
        return _change_status(
            photo_id, PhotoStatus.LOCATED, status=PhotoStatus.CONFIRMED.value
        )


def remove_photo(photo_id: int) -> str | None:
    """Delete a photo and its views. Return its file path, or None if the id is unknown.

    The caller deletes the file afterwards. A photo that the worker holds
    (`processing`) cannot be removed: raise `InvalidStateError`.
    """
    _check_id(photo_id, "photo_id")
    with translate_errors("remove_photo"), database_proxy.atomic("IMMEDIATE"):
        photo: ReferencePhoto | None = ReferencePhoto.get_or_none(
            ReferencePhoto.id == photo_id
        )
        if photo is None:
            return None
        if photo.status == PhotoStatus.PROCESSING.value:
            raise InvalidStateError(f"Photo {photo_id} is being analysed")
        ReferencePhoto.delete().where(ReferencePhoto.id == photo_id).execute()
    return photo.file_path


def retry_photo(photo_id: int) -> ReferencePhoto:
    """Set a `failed` photo back to `pending`, so the worker tries it again."""
    _check_id(photo_id, "photo_id")
    with translate_errors("retry_photo"):
        return _change_status(
            photo_id,
            PhotoStatus.FAILED,
            status=PhotoStatus.PENDING.value,
            status_detail=None,
        )


# ---- Views ----


def _stack(rows: list[ObjectView]) -> np.ndarray:
    if not rows:
        return np.empty((0, _expected_dimension or 0), dtype=np.float32)
    try:
        return np.stack([np.frombuffer(row.vector, dtype=np.float32) for row in rows])
    except ValueError as error:
        raise InvalidRecordError("The stored views have different sizes") from error


def load_views(object_id: int, embedding_model: str) -> tuple[np.ndarray, list[int]]:
    """Return the views of the confirmed photos of an object, made with one model.

    The result is an array of shape (number of views, size of a vector) and the
    photo id of each row. Rows of one photo are next to each other.
    """
    _check_id(object_id, "object_id")
    assert isinstance(embedding_model, str) and embedding_model, "model name needed"
    with translate_errors("load_views"):
        rows: list[ObjectView] = list(
            ObjectView.select()
            .join(ReferencePhoto)
            .where(
                ObjectView.enrolled_object == object_id,
                ObjectView.embedding_model == embedding_model,
                ReferencePhoto.status == PhotoStatus.CONFIRMED.value,
            )
            .order_by(ObjectView.photo, ObjectView.id)
        )
        return _stack(rows), [row.photo_id for row in rows]


def load_other_views(object_id: int, embedding_model: str) -> np.ndarray:
    """Return the views of the confirmed photos of all other objects (the negatives)."""
    _check_id(object_id, "object_id")
    assert isinstance(embedding_model, str) and embedding_model, "model name needed"
    with translate_errors("load_other_views"):
        rows: list[ObjectView] = list(
            ObjectView.select()
            .join(ReferencePhoto)
            .where(
                ObjectView.enrolled_object != object_id,
                ObjectView.embedding_model == embedding_model,
                ReferencePhoto.status == PhotoStatus.CONFIRMED.value,
            )
            .order_by(ObjectView.id)
        )
        return _stack(rows)


# ---- Calibration ----


def set_calibration(
    object_id: int,
    tau: float,
    tau_strong: float,
    margin_min: float,
    embedding_model: str,
    note: str | None = None,
) -> EnrolledObject:
    """Save the result of the calibration. The object is `ready`."""
    _check_id(object_id, "object_id")
    assert all(
        isinstance(v, float) for v in (tau, tau_strong, margin_min)
    ), "floats needed"
    assert isinstance(embedding_model, str) and embedding_model, "model name needed"
    assert note is None or isinstance(note, str), "note must be a string or None"
    return _update_object(
        object_id,
        status=ObjectStatus.READY.value,
        tau=tau,
        tau_strong=tau_strong,
        margin_min=margin_min,
        calibration_model=embedding_model,
        calibration_note=note,
    )


def clear_calibration(object_id: int, note: str | None = None) -> EnrolledObject:
    """Remove the calibration. The object is `draft`. `note` tells the carer what is missing."""
    _check_id(object_id, "object_id")
    assert note is None or isinstance(note, str), "note must be a string or None"
    return _update_object(
        object_id,
        status=ObjectStatus.DRAFT.value,
        tau=None,
        tau_strong=None,
        margin_min=None,
        calibration_model=None,
        calibration_note=note,
    )


def _update_object(object_id: int, **changes: object) -> EnrolledObject:
    with translate_errors("update object"):
        changed: int = (
            EnrolledObject.update(updated_at=utc_now(), **changes)
            .where(EnrolledObject.id == object_id)
            .execute()
        )
        if changed != 1:
            raise InvalidStateError(f"Object {object_id} does not exist")
        return EnrolledObject.get_by_id(object_id)


def gallery_phrases() -> list[str]:
    """Return the detector phrases of all `ready` objects: the category and the synonyms.

    Each phrase is listed once. The order is stable. Objects with
    `tracked = False` are included: the detector must also find the look-alikes.
    """
    with translate_errors("gallery_phrases"):
        phrases: list[str] = []
        for obj in (
            EnrolledObject.select()
            .where(EnrolledObject.status == ObjectStatus.READY.value)
            .order_by(EnrolledObject.id)
        ):
            for phrase in [obj.category_phrase, *obj.synonyms]:
                if phrase not in phrases:
                    phrases.append(phrase)
        return phrases


def list_photo_paths() -> list[str]:
    """Return the `file_path` of every reference photo, sorted."""
    with translate_errors("list_photo_paths"):
        return [
            row.file_path
            for row in ReferencePhoto.select(ReferencePhoto.file_path).order_by(
                ReferencePhoto.file_path
            )
        ]


def reset_processing_photos() -> list[int]:
    """Set every `processing` photo back to `pending`. Return their ids.

    Call it inside the start-up recovery: a stop left these photos behind.
    """
    ids: list[int] = [
        row.id
        for row in ReferencePhoto.select(ReferencePhoto.id).where(
            ReferencePhoto.status == PhotoStatus.PROCESSING.value
        )
    ]
    if ids:
        ReferencePhoto.update(status=PhotoStatus.PENDING.value).where(
            ReferencePhoto.id.in_(ids)
        ).execute()
    return ids
