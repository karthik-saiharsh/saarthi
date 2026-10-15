"""The vectors of the reference crops of each object, for the modes `evidence` (`ai/evidence.py`) and `keys` (`ai/keys.py`).

Each confirmed photo gives two vectors of the crop at the box: the SigLIP vector and the DINOv2 key (the CLS token).

The vectors are made from the stored reference photos: the confirmed photos, cut at the box that the carer
accepted. The carer does nothing new. The vectors stay in memory. They are made again only when the set of
confirmed photos of an object changes, so a frame costs nothing here in the normal case.
A photo whose file cannot be read is left out. An object without a readable photo has no vectors.
The log shows ids and counts only.
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import AiError
from ai.types import Box
from db.schema import PhotoStatus, ReferencePhoto

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class References:
    """The vectors of the reference crops of one object. The rows of both arrays are the same photos."""

    vectors: np.ndarray  # SigLIP, unit length, shape (photos, dimension)
    photo_ids: np.ndarray  # The photo of each row
    key_vectors: (
        np.ndarray
    )  # The DINOv2 key (CLS token of the crop), unit length, shape (photos, dimension)


class ReferenceVectors:
    """Keeps the reference vectors of each object. It is safe to call from several threads."""

    def __init__(self, engine: AiEngine) -> None:
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._engine: AiEngine = engine
        self._lock: threading.Lock = threading.Lock()
        # object id -> (the confirmed photo ids that the vectors come from, the vectors)
        self._kept: dict[int, tuple[tuple[int, ...], References | None]] = {}

    def get(self, object_id: int) -> References | None:
        """Return the reference vectors of an object, or None if no confirmed photo can be read.

        Raise `DbError` if the photos cannot be listed. An error of the model raises `AiError`.
        """
        assert isinstance(object_id, int) and object_id > 0, "object_id must be > 0"
        photos: list[ReferencePhoto] = [
            p
            for p in db.gallery.list_photos(object_id)
            if p.status == PhotoStatus.CONFIRMED.value and p.x0 is not None
        ]
        key: tuple[int, ...] = tuple(p.id for p in photos)
        with self._lock:
            kept = self._kept.get(object_id)
            if kept is not None and kept[0] == key:
                return kept[1]
        references: References | None = self._make(object_id, photos)
        with self._lock:
            self._kept[object_id] = (key, references)
        return references

    def _make(self, object_id: int, photos: list[ReferencePhoto]) -> References | None:
        vectors: list[np.ndarray] = []
        key_vectors: list[np.ndarray] = []
        photo_ids: list[int] = []
        for photo in photos:
            try:
                picture = self._engine.open_image(storage.read_image(photo.file_path))
            except (AiError, storage.StorageError):
                logger.warning("Reference photo %d cannot be read", photo.id)
                continue
            box: Box = Box(photo.x0, photo.y0, photo.x1, photo.y1)
            vectors.append(self._engine.embed_evidence_crops(picture, [box])[0])
            key_vectors.append(self._engine.embed_key_crops(picture, [box])[0])
            photo_ids.append(photo.id)
        logger.info(
            "Reference vectors of object %d: %d of %d photo(s)",
            object_id,
            len(vectors),
            len(photos),
        )
        if not vectors:
            return None
        return References(np.stack(vectors), np.array(photo_ids), np.stack(key_vectors))
