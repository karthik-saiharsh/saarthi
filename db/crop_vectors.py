"""The crop vectors of the regions: the SigLIP vector and the DINOv2 key of the crop of each box.

The table `region` holds all boxes of a frame (the candidates and the boxes of the broad detector). Each box
can have two vectors of its crop, each in its own FAISS index with the region id as the vector id:

- the **crop** index: the SigLIP vector. The matching mode `keys` and the open search use it.
- the **key** index: the DINOv2 key (the CLS token of the crop). The matching mode `keys` uses it.

The columns `Region.crop_model` and `Region.key_model` name the model of each vector. Null means no vector.
This module keeps the columns and the indexes in step. It runs no model and reads no file.

Rule: the crop vectors are an extra of a region. The recovery at start-up removes a vector without a marked
region and clears a mark without a vector. It never sends a frame back to `pending`.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from db.errors import DbError, VectorIndexError
from db.schema import Image, ImageStatus, Region, RegionSource, database_proxy
from db.support import translate_errors
from db.vector_index import VectorIndex

logger: logging.Logger = logging.getLogger(__name__)

# The most ids in one call, and the most rows that a search reads.
MAX_IDS: int = 20000

_crop_index: VectorIndex | None = None
_key_index: VectorIndex | None = None


def configure(
    crop_path: Path,
    crop_dimension: int,
    key_path: Path,
    key_dimension: int,
    checkpoint_every: int,
) -> None:
    """Open both indexes. `db.api.init_db` calls it. Save the old indexes first."""
    global _crop_index, _key_index
    for index in (_crop_index, _key_index):
        if index is not None and index.unsaved:
            index.save()
    _crop_index = VectorIndex(crop_path, crop_dimension, checkpoint_every)
    _key_index = VectorIndex(key_path, key_dimension, checkpoint_every)


def _indexes() -> tuple[VectorIndex, VectorIndex]:
    if _crop_index is None or _key_index is None:
        raise VectorIndexError("The crop indexes are not open. Call init_db first")
    return _crop_index, _key_index


def _check_ids(ids: list[int]) -> None:
    assert isinstance(ids, list), "ids must be a list"
    assert len(ids) <= MAX_IDS, f"at most {MAX_IDS} ids"
    assert all(isinstance(i, int) and i > 0 for i in ids), "ids must be positive"


def store(
    region_ids: list[int],
    crops: np.ndarray | None,
    crop_model: str | None,
    keys: np.ndarray | None,
    key_model: str | None,
) -> None:
    """Add the crop vectors of stored regions and mark the regions. Do all steps or none.

    `crops` and `keys` have one float32 row of unit length for each id, in the same order. Either can be None
    (no vector of that kind), and then its model must be None too.
    """
    _check_ids(region_ids)
    assert region_ids, "region_ids must not be empty"
    assert (crops is None) == (crop_model is None), "a crop model goes with crops"
    assert (keys is None) == (key_model is None), "a key model goes with keys"
    crop_index, key_index = _indexes()
    added: list[VectorIndex] = []
    try:
        if crops is not None:
            crop_index.add(region_ids, crops)
            added.append(crop_index)
        if keys is not None:
            key_index.add(region_ids, keys)
            added.append(key_index)
        with translate_errors("store crop vectors"), database_proxy.atomic():
            changed: int = (
                Region.update(crop_model=crop_model, key_model=key_model)
                .where(Region.id.in_(set(region_ids)))
                .execute()
            )
            if changed != len(set(region_ids)):
                raise DbError(
                    f"{len(set(region_ids)) - changed} region id(s) do not exist"
                )
    except (DbError, VectorIndexError):
        for index in added:
            index.remove(region_ids)  # Keep the indexes and the table in step
        raise
    for index in added:
        index.checkpoint_if_due()
    logger.debug("Stored the crop vectors of %d region(s)", len(region_ids))


def get_crop_vectors(region_ids: list[int]) -> np.ndarray:
    """Return the SigLIP crop vectors of these regions, one row for each id. Raise `VectorIndexError` if one is missing."""
    _check_ids(region_ids)
    return _indexes()[0].get_vectors(region_ids)


def get_key_vectors(region_ids: list[int]) -> np.ndarray:
    """Return the DINOv2 keys of these regions, one row for each id. Raise `VectorIndexError` if one is missing."""
    _check_ids(region_ids)
    return _indexes()[1].get_vectors(region_ids)


def remove(region_ids: list[int]) -> int:
    """Remove the crop vectors of these regions from both indexes. Unknown ids are ignored. Return the count."""
    _check_ids(region_ids)
    removed: int = 0
    for index in _indexes():
        removed += index.remove(region_ids)
        index.checkpoint_if_due()
    return removed


def _window(query, since: datetime, until: datetime):
    assert isinstance(since, datetime) and isinstance(until, datetime), "datetimes"
    assert since <= until, "since must not be after until"
    return query.where(
        Image.status == ImageStatus.DONE.value,
        Image.captured_at > since,
        Image.captured_at <= until,
    )


def list_broad_regions(
    since: datetime, until: datetime, crop_model: str, limit: int = 5000
) -> list[Region]:
    """Return the boxes of the broad detector in `done` frames, with a crop vector of `crop_model`, newest frame first.

    The open search uses them. A frame is in the window if `since` < `captured_at` <= `until` (naive UTC times).
    Each region has its frame in `region.image`.
    """
    assert isinstance(crop_model, str) and crop_model, "model name needed"
    assert isinstance(limit, int) and 0 < limit <= MAX_IDS, "limit out of range"
    with translate_errors("list_broad_regions"):
        query = Region.select(Region, Image).join(Image)
        return list(
            _window(query, since, until)
            .where(
                Region.source == RegionSource.BROAD.value,
                Region.crop_model == crop_model,
            )
            .order_by(Image.captured_at.desc(), Image.id.desc(), Region.id)
            .limit(limit)
        )


def list_keyed_regions(
    since: datetime, until: datetime, crop_model: str, key_model: str, limit: int = 5000
) -> list[Region]:
    """Return the regions of `done` frames with both crop vectors of these models, newest frame first.

    The archive search of the matching mode `keys` uses them. The regions of one frame are next to each other.
    A region that failed the crop gate is listed too: the mode `keys` caps it at weak.
    """
    assert isinstance(crop_model, str) and crop_model, "model name needed"
    assert isinstance(key_model, str) and key_model, "model name needed"
    assert isinstance(limit, int) and 0 < limit <= MAX_IDS, "limit out of range"
    with translate_errors("list_keyed_regions"):
        query = Region.select(Region, Image).join(Image)
        return list(
            _window(query, since, until)
            .where(Region.crop_model == crop_model, Region.key_model == key_model)
            .order_by(Image.captured_at.desc(), Image.id.desc(), Region.id)
            .limit(limit)
        )


def recover() -> tuple[int, int]:
    """Repair the columns and the indexes after a stop. Return `(vectors removed, marks cleared)`."""
    crop_index, key_index = _indexes()
    removed: int = 0
    cleared: int = 0
    with translate_errors("crop vector recovery"):
        for index, column, name in (
            (crop_index, Region.crop_model, "crop_model"),
            (key_index, Region.key_model, "key_model"),
        ):
            marked: set[int] = {
                row.id for row in Region.select(Region.id).where(column.is_null(False))
            }
            stored: set[int] = index.ids()
            removed += index.remove(sorted(stored - marked))
            gaps: list[int] = sorted(marked - stored)
            if gaps:
                cleared += (
                    Region.update(**{name: None}).where(Region.id.in_(gaps)).execute()
                )
            index.checkpoint_if_due()
    if removed or cleared:
        logger.warning(
            "Crop vector recovery: %d orphan vector(s) removed, %d mark(s) without vector cleared",
            removed,
            cleared,
        )
    return removed, cleared


def checkpoint() -> None:
    """Write both indexes to disk now. Raise `VectorIndexError` if a write fails."""
    for index in _indexes():
        index.save()


def counts() -> tuple[int, int]:
    """Return the number of vectors in the crop index and in the key index."""
    crop_index, key_index = _indexes()
    return crop_index.count, key_index.count


__all__ = [
    "checkpoint",
    "configure",
    "counts",
    "get_crop_vectors",
    "get_key_vectors",
    "list_broad_regions",
    "list_keyed_regions",
    "recover",
    "remove",
    "store",
]
