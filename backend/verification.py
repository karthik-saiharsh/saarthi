"""The check of a sighting against the reference photos of an object (stage I5).

The memory (a weak sighting in a new frame) and the search (a weak hit in the archive) use it.
It takes the confirmed reference photos of the object, the most similar to the candidate first.
It cuts the same part from each photo (the box of the object, with some space around it), and
the verifier counts the matching keypoints (see `docs/backend/Verification.md`).

The check only helps: an error (the verifier, or a reference file that cannot be read) gives no
result, and the caller keeps the sighting as it is. The log shows ids and counts only.
"""

import logging

import numpy as np
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import AiError
from ai.types import Box, VerifyResult

logger: logging.Logger = logging.getLogger(__name__)


class ReferenceVerifier:
    """Checks a candidate against the reference photos of an object. It is safe to call from several threads."""

    def __init__(self, engine: AiEngine, max_references: int) -> None:
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        assert max_references > 0, "max_references must be positive"
        self._engine: AiEngine = engine
        self._max_references: int = max_references

    @property
    def enabled(self) -> bool:
        """True if the engine has a verifier. If False, `verify` must not be called."""
        return self._engine.verification_enabled

    def verify(
        self,
        object_id: int,
        vector: np.ndarray,
        picture: PilImage.Image,
        box: Box,
    ) -> VerifyResult | None:
        """Check the part `box` of `picture` against the reference photos of the object.

        `vector` is the vector of the candidate. It orders the reference photos.
        Return None if the check cannot be made (an error of the verifier or of a file).
        """
        assert self.enabled, "the verifier is switched off"
        assert isinstance(picture, PilImage.Image), "picture must be a PIL image"
        assert isinstance(box, Box), "box must be a Box"
        try:
            references: list[PilImage.Image] = self._reference_crops(object_id, vector)
            return self._engine.verify_match(
                references, self._engine.verification_crop(picture, box)
            )
        except (AiError, storage.StorageError):
            logger.warning("The check of a sighting of object %d failed", object_id)
            return None

    def _reference_crops(
        self, object_id: int, vector: np.ndarray
    ) -> list[PilImage.Image]:
        """Return the pictures of the confirmed reference photos, the most similar first.

        A photo is as similar as its most similar view. At most `max_references`
        photos are used. A photo that cannot be read is left out.
        """
        views, photo_ids = db.gallery.load_views(
            object_id, self._engine.embedding_model
        )
        if not len(views):
            return []
        similarity: dict[int, float] = {}
        for photo_id, value in zip(photo_ids, views @ vector):
            similarity[photo_id] = max(similarity.get(photo_id, -1.0), float(value))
        order: list[int] = sorted(similarity, key=lambda i: -similarity[i])
        crops: list[PilImage.Image] = []
        for photo_id in order[: self._max_references]:
            photo = db.gallery.get_photo(photo_id)
            if photo is None or photo.x0 is None:
                continue
            try:
                picture: PilImage.Image = self._engine.open_image(
                    storage.read_image(photo.file_path)
                )
            except (AiError, storage.StorageError):
                logger.warning("Reference photo %d cannot be read", photo_id)
                continue
            crops.append(
                self._engine.verification_crop(
                    picture, Box(photo.x0, photo.y0, photo.x1, photo.y1)
                )
            )
        return crops
