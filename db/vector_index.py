"""FAISS index of the region vectors.

The index maps a region id to a vector. It uses the inner product on
unit-length vectors, so a score is the cosine similarity. This module knows
nothing about the database. `db.api` keeps the index and the tables in step.
"""

import logging
import os
import threading
from pathlib import Path

import faiss
import numpy as np

from db.errors import VectorIndexError

logger: logging.Logger = logging.getLogger(__name__)

# How far the length of a vector may be from 1.0.
NORM_TOLERANCE: float = 1e-3


class VectorIndex:
    """A thread-safe FAISS index that is saved to one file."""

    def __init__(self, path: Path, dimension: int, checkpoint_every: int) -> None:
        assert isinstance(path, Path), "path must be a Path"
        assert isinstance(dimension, int) and dimension > 0, "dimension must be > 0"
        assert isinstance(checkpoint_every, int) and checkpoint_every > 0

        self._path: Path = path
        self._dimension: int = dimension
        self._checkpoint_every: int = checkpoint_every
        self._lock: threading.RLock = threading.RLock()
        self._unsaved: int = 0
        self._index: faiss.IndexIDMap2 = self._open()
        self._ids: set[int] = set(faiss.vector_to_array(self._index.id_map).tolist())
        logger.info(
            "Vector index ready: %d vector(s), dimension %d", len(self._ids), dimension
        )

    def _open(self) -> faiss.IndexIDMap2:
        """Load the index file, or make an empty index if there is no file."""
        if not self._path.exists():
            logger.info("No index file at %s. Starting with an empty index", self._path)
            return faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
        try:
            index = faiss.read_index(str(self._path))
        except RuntimeError as error:
            logger.error("Cannot read index file %s: %s", self._path, error)
            raise VectorIndexError(f"Cannot read index file: {self._path}") from error
        if not isinstance(index, faiss.IndexIDMap2) or index.d != self._dimension:
            raise VectorIndexError(
                f"Index file {self._path} does not match the configured dimension "
                f"{self._dimension}"
            )
        return index

    @property
    def count(self) -> int:
        """Return the number of vectors."""
        with self._lock:
            return len(self._ids)

    @property
    def unsaved(self) -> int:
        """Return how many changes the file does not have yet."""
        with self._lock:
            return self._unsaved

    def ids(self) -> set[int]:
        """Return a copy of the set of region ids in the index."""
        with self._lock:
            return set(self._ids)

    def get_vectors(self, ids: list[int]) -> np.ndarray:
        """Return the stored vectors of these ids, one row for each id, in the same order.

        Raise `VectorIndexError` if an id is not in the index. The result is a copy.
        """
        assert isinstance(ids, list), "ids must be a list"
        assert all(isinstance(i, int) and i > 0 for i in ids), "ids must be positive"
        with self._lock:
            missing: list[int] = [i for i in ids if i not in self._ids]
            if missing:
                raise VectorIndexError(f"ids not in the index: {missing[:5]}")
            if not ids:
                return np.empty((0, self._dimension), dtype=np.float32)
            return np.stack([self._index.reconstruct(i) for i in ids]).astype(
                np.float32
            )

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        """Add vectors. Refuse the whole call if any id is already in the index."""
        assert isinstance(ids, list) and ids, "ids must be a non-empty list"
        assert all(isinstance(i, int) and i > 0 for i in ids), "ids must be positive"
        assert isinstance(vectors, np.ndarray), "vectors must be a numpy array"
        assert vectors.dtype == np.float32, "vectors must be float32"
        assert vectors.shape == (len(ids), self._dimension), (
            f"vectors must have shape ({len(ids)}, {self._dimension}), "
            f"got {vectors.shape}"
        )
        if len(set(ids)) != len(ids):
            raise VectorIndexError("ids contain a duplicate")
        if not np.isfinite(vectors).all():
            raise VectorIndexError("vectors contain a value that is not finite")
        lengths: np.ndarray = np.linalg.norm(vectors, axis=1)
        if np.abs(lengths - 1.0).max() > NORM_TOLERANCE:
            raise VectorIndexError("vectors must have unit length")

        with self._lock:
            known: list[int] = [i for i in ids if i in self._ids]
            if known:
                raise VectorIndexError(f"ids already in the index: {known}")
            self._index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))
            self._ids.update(ids)
            self._unsaved += len(ids)

    def remove(self, ids: list[int]) -> int:
        """Remove vectors. Ids that are not in the index are ignored. Return the count."""
        assert isinstance(ids, list), "ids must be a list"
        assert all(isinstance(i, int) and i > 0 for i in ids), "ids must be positive"
        with self._lock:
            present: list[int] = [i for i in set(ids) if i in self._ids]
            if not present:
                return 0
            self._index.remove_ids(np.asarray(present, dtype=np.int64))
            self._ids.difference_update(present)
            self._unsaved += len(present)
            return len(present)

    def checkpoint_if_due(self) -> bool:
        """Save the index if enough changes are unsaved. A failure is logged, not raised."""
        with self._lock:
            if self._unsaved < self._checkpoint_every:
                return False
            try:
                self.save()
            except VectorIndexError:
                return False  # Already logged. The next change tries again.
            return True

    def save(self) -> None:
        """Write the index to disk. The old file stays whole if the write fails."""
        with self._lock:
            temp_path: Path = self._path.with_name(self._path.name + ".tmp")
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self._index, str(temp_path))
                with temp_path.open("rb") as written:
                    os.fsync(written.fileno())
                os.replace(temp_path, self._path)
            except (OSError, RuntimeError) as error:
                logger.error("Cannot save index to %s: %s", self._path, error)
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Cannot remove temp file %s", temp_path)
                raise VectorIndexError(f"Cannot save index: {self._path}") from error
            logger.info("Saved index: %d vector(s)", len(self._ids))
            self._unsaved = 0
