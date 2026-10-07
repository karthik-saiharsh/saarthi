"""The boxes of the broad detector and their vectors (the open search).

The broad detector finds objects that nobody registered (see `docs/backend/Open Search.md`).
A box has a row in the table `open_region` and a SigLIP vector in a second FAISS index. The id of the
row is the id of the vector, as for the regions of the registered objects. This module keeps the rows
and the vectors in step. It runs no model and reads no file.

Rule: the rows and the vectors of one frame are added in one call. The recovery at start-up repairs
what a stop leaves: a row without a vector is deleted, and a vector without a row is removed.
The boxes of a frame that is processed again are deleted first, so a retry makes no duplicates.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from db.errors import DbError, InvalidRecordError, VectorIndexError
from db.schema import Image, ImageStatus, OpenRegion, database_proxy
from db.support import translate_errors
from db.vector_index import VectorIndex

logger: logging.Logger = logging.getLogger(__name__)

# The longest label that the table stores.
MAX_LABEL_LENGTH: int = 80
# The most ids in one call, and the most rows that a search reads.
MAX_IDS: int = 20000

_index: VectorIndex | None = None


@dataclass(frozen=True)
class NewOpenRegion:
    """Data for one box to insert. Coordinates are pixels in the frame."""

    x0: int
    y0: int
    x1: int
    y1: int
    label: str
    confidence: float
    # The candidate box of a registered object that this box overlaps, or None.
    region_id: int | None = None

    def __post_init__(self) -> None:
        assert all(isinstance(v, int) for v in (self.x0, self.y0, self.x1, self.y1))
        assert 0 <= self.x0 < self.x1, "x1 must be more than x0, and x0 at least 0"
        assert 0 <= self.y0 < self.y1, "y1 must be more than y0, and y0 at least 0"
        assert isinstance(self.label, str) and self.label.strip(), "label needed"
        assert len(self.label) <= MAX_LABEL_LENGTH, "label is too long"
        assert isinstance(self.confidence, float), "confidence must be a float"
        assert 0.0 <= self.confidence <= 1.0, "confidence must be from 0 to 1"
        assert self.region_id is None or (
            isinstance(self.region_id, int) and self.region_id > 0
        ), "region_id must be positive or None"


def configure(path: Path, dimension: int, checkpoint_every: int) -> None:
    """Open the vector index of this module. `db.api.init_db` calls it. Save the old index first."""
    global _index
    if _index is not None and _index.unsaved:
        _index.save()
    _index = VectorIndex(path, dimension, checkpoint_every)


def _require_index() -> VectorIndex:
    if _index is None:
        raise VectorIndexError("The open index is not open. Call init_db first")
    return _index


def _check_ids(ids: list[int]) -> None:
    assert isinstance(ids, list), "ids must be a list"
    assert len(ids) <= MAX_IDS, f"at most {MAX_IDS} ids"
    assert all(isinstance(i, int) and i > 0 for i in ids), "ids must be positive"


def add_open_regions(
    image_id: int,
    items: list[NewOpenRegion],
    vectors: np.ndarray,
    vector_model: str,
) -> list[OpenRegion]:
    """Insert the boxes of a frame and add their vectors. Do both steps or none.

    `vectors` has one float32 row of unit length for each item, in the same order.
    Raise `InvalidRecordError` if the frame does not exist.
    """
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    assert isinstance(items, list) and items, "items must not be empty"
    assert isinstance(vector_model, str) and vector_model, "model name needed"
    index: VectorIndex = _require_index()

    with translate_errors("add_open_regions"), database_proxy.atomic():
        rows: list[OpenRegion] = [
            OpenRegion.create(
                image=image_id,
                x0=item.x0,
                y0=item.y0,
                x1=item.x1,
                y1=item.y1,
                label=item.label.strip(),
                confidence=item.confidence,
                region=item.region_id,
                vector_model=vector_model,
            )
            for item in items
        ]
        ids: list[int] = [row.id for row in rows]
        # If the vectors are refused, the exception leaves the block and the transaction rolls the rows back.
        index.add(ids, vectors)
    index.checkpoint_if_due()
    logger.debug("Stored %d open box(es) of image %d", len(rows), image_id)
    return rows


def get_open_regions(image_id: int) -> list[OpenRegion]:
    """Return the boxes of a frame, the most confident first."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    with translate_errors("get_open_regions"):
        return list(
            OpenRegion.select()
            .where(OpenRegion.image == image_id)
            .order_by(OpenRegion.confidence.desc(), OpenRegion.id)
        )


def list_open_regions(
    since: datetime, until: datetime, vector_model: str, limit: int = 5000
) -> list[OpenRegion]:
    """Return the boxes of `done` frames that a search can compare, newest frame first.

    A frame is in the window if `since` < `captured_at` <= `until` (naive UTC times). Only boxes with a
    vector of `vector_model` are listed. Each box has its frame in `region.image`.
    """
    assert isinstance(since, datetime) and isinstance(until, datetime), "datetimes"
    assert since <= until, "since must not be after until"
    assert isinstance(vector_model, str) and vector_model, "model name needed"
    assert isinstance(limit, int) and 0 < limit <= MAX_IDS, "limit out of range"
    with translate_errors("list_open_regions"):
        return list(
            OpenRegion.select(OpenRegion, Image)
            .join(Image)
            .where(
                Image.status == ImageStatus.DONE.value,
                Image.captured_at > since,
                Image.captured_at <= until,
                OpenRegion.vector_model == vector_model,
            )
            .order_by(Image.captured_at.desc(), Image.id.desc(), OpenRegion.id)
            .limit(limit)
        )


def get_vectors(ids: list[int]) -> np.ndarray:
    """Return the stored vectors of these boxes, one row for each id, in the same order.

    Raise `VectorIndexError` if a box has no vector.
    """
    _check_ids(ids)
    return _require_index().get_vectors(ids)


def ids_for_images(image_ids: list[int]) -> list[int]:
    """Return the ids of the boxes of these frames. `db.api.delete_images` uses it."""
    _check_ids(image_ids)
    if not image_ids:
        return []
    with translate_errors("ids_for_images"):
        return [
            row.id
            for row in OpenRegion.select(OpenRegion.id)
            .where(OpenRegion.image.in_(set(image_ids)))
            .order_by(OpenRegion.id)
        ]


def remove_vectors(ids: list[int]) -> int:
    """Remove vectors from the index. Ids that are not in it are ignored. Return the count."""
    _check_ids(ids)
    index: VectorIndex = _require_index()
    removed: int = index.remove(ids)
    index.checkpoint_if_due()
    return removed


def drop_for_images(image_ids: list[int]) -> int:
    """Delete the boxes and the vectors of these frames. Return the number of boxes deleted.

    Use it before a frame is processed again, so the retry makes no duplicates.
    """
    _check_ids(image_ids)
    if not image_ids:
        return 0
    ids: list[int] = ids_for_images(image_ids)
    if not ids:
        return 0
    with translate_errors("drop_for_images"):
        OpenRegion.delete().where(OpenRegion.id.in_(ids)).execute()
    remove_vectors(ids)
    return len(ids)


def recover() -> tuple[int, int]:
    """Repair the rows and the index after a stop. Return `(vectors removed, boxes deleted)`.

    A vector that has no row is removed. A box that has no vector is deleted: the open search is an extra,
    so the frame does not go back to `pending`.
    """
    index: VectorIndex = _require_index()
    with translate_errors("open recover"):
        row_ids: set[int] = {row.id for row in OpenRegion.select(OpenRegion.id)}
        vector_ids: set[int] = index.ids()
        orphans: list[int] = sorted(vector_ids - row_ids)
        gaps: list[int] = sorted(row_ids - vector_ids)
        if gaps:
            OpenRegion.delete().where(OpenRegion.id.in_(gaps)).execute()
    removed: int = index.remove(orphans)
    index.checkpoint_if_due()
    if removed or gaps:
        logger.warning(
            "Open index recovery: %d orphan vector(s) removed, %d box(es) without vector deleted",
            removed,
            len(gaps),
        )
    return removed, len(gaps)


def checkpoint() -> None:
    """Write the index to disk now. Raise `VectorIndexError` if the write fails."""
    _require_index().save()


def count() -> int:
    """Return the number of vectors in the index."""
    return _require_index().count


__all__ = [
    "DbError",
    "InvalidRecordError",
    "NewOpenRegion",
    "add_open_regions",
    "checkpoint",
    "configure",
    "count",
    "drop_for_images",
    "get_open_regions",
    "get_vectors",
    "ids_for_images",
    "list_open_regions",
    "recover",
    "remove_vectors",
]
