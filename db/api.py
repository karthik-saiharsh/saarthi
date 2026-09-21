"""Public interface of the db component.

`init_db` reads `config.toml`, opens the SQLite database, and creates the tables.
"""

import logging
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
from peewee import IntegrityError, PeeweeException, SqliteDatabase

from db.errors import (
    ConfigError,
    DbError,
    DuplicateImageError,
    InvalidRecordError,
    InvalidStateError,
    VectorIndexError,
)
from db.schema import (
    Image,
    ImageStatus,
    Region,
    RegionSource,
    database_proxy,
    utc_now,
)
from db.vector_index import VectorIndex

logger: logging.Logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler: logging.StreamHandler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s-%(asctime)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

DB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = DB_DIR / "config.toml"

# Longest `status_detail` text that the database stores.
MAX_STATUS_DETAIL_LENGTH: int = 500

_PATH_KEYS: tuple[str, ...] = ("sqlite", "faiss_index")
_VECTOR_KEYS: tuple[str, ...] = ("dimension", "checkpoint_every")
_SQLITE_PRAGMAS: dict[str, object] = {
    "foreign_keys": 1,
    "journal_mode": "wal",
    "busy_timeout": 5000,
    "synchronous": "normal",
}


@dataclass(frozen=True)
class NewRegion:
    """Data for one region to insert. Coordinates are pixels in the image frame."""

    x0: int
    y0: int
    x1: int
    y1: int
    det_score: float
    source: RegionSource
    phrases: list[str]
    in_hand: bool = False
    quality_ok: bool = True

    def __post_init__(self) -> None:
        assert all(isinstance(v, int) for v in (self.x0, self.y0, self.x1, self.y1))
        assert 0 <= self.x0 < self.x1, "x1 must be more than x0, and x0 at least 0"
        assert 0 <= self.y0 < self.y1, "y1 must be more than y0, and y0 at least 0"
        assert isinstance(self.det_score, float), "det_score must be a float"
        assert isinstance(self.source, RegionSource), "source must be a RegionSource"
        assert isinstance(self.phrases, list)
        assert all(isinstance(phrase, str) for phrase in self.phrases)


@dataclass(frozen=True)
class RecoveryReport:
    """Result of `recover_interrupted`. The backend acts on the FAISS index."""

    # Images that were `processing` and are `pending` again.
    reset_image_ids: list[int]
    # Deleted regions that had a vector. They must be removed from the index.
    stale_vector_ids: list[int]
    # Regions of `done` images with no vector.
    unembedded_region_ids: list[int]


@dataclass(frozen=True)
class StartupReport:
    """Result of `startup_recovery`. The backend logs it. No action is needed."""

    # Images that were `processing` and are `pending` again.
    interrupted_image_ids: list[int]
    # Done images that had a region with no vector. They are `pending` again.
    requeued_image_ids: list[int]
    # Vectors that the index had and the tables did not know. They are removed.
    orphan_vectors_removed: int
    # Marks that the tables had and the index did not. They are cleared.
    missing_vectors_cleared: int


@dataclass(frozen=True)
class DbConfig:
    """Resolved settings of the db component."""

    sqlite_path: Path
    faiss_path: Path
    vector_dimension: int
    checkpoint_every: int


def _resolve_inside_db_dir(key: str, value: object) -> Path:
    """Turn a relative config path into an absolute path inside the db folder."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"paths.{key} must be a non-empty string")
    if Path(value).is_absolute():
        raise ConfigError(f"paths.{key} must be relative to the db folder")
    resolved: Path = (DB_DIR / value).resolve()
    if not resolved.is_relative_to(DB_DIR):
        raise ConfigError(f"paths.{key} must stay inside the db folder")
    return resolved


def _positive_int(table: str, key: str, value: object) -> int:
    """Return `value` if it is a positive integer. Raise `ConfigError` if not."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{table}.{key} must be a positive integer")
    return value


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> DbConfig:
    """Read and validate the config file."""
    assert isinstance(config_path, Path), "config_path must be a Path"
    try:
        with config_path.open("rb") as file:
            raw: dict = tomllib.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"Config file not found: {config_path}") from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"Cannot read config {config_path}: {error}") from error

    paths: object = raw.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError("Missing [paths] table in config")
    for key in _PATH_KEYS:
        if key not in paths:
            raise ConfigError(f"Missing key paths.{key} in config")

    vectors: object = raw.get("vectors")
    if not isinstance(vectors, dict):
        raise ConfigError("Missing [vectors] table in config")
    for key in _VECTOR_KEYS:
        if key not in vectors:
            raise ConfigError(f"Missing key vectors.{key} in config")

    config: DbConfig = DbConfig(
        sqlite_path=_resolve_inside_db_dir("sqlite", paths["sqlite"]),
        faiss_path=_resolve_inside_db_dir("faiss_index", paths["faiss_index"]),
        vector_dimension=_positive_int("vectors", "dimension", vectors["dimension"]),
        checkpoint_every=_positive_int(
            "vectors", "checkpoint_every", vectors["checkpoint_every"]
        ),
    )
    logger.info("Loaded config from %s", config_path)
    return config


def close_connection() -> None:
    """Close the database connection of the current thread, if it is open.

    A thread that uses the database for a long time calls this when it ends.
    """
    with _translate_errors("close_connection"):
        if not database_proxy.is_closed():
            database_proxy.close()


# The open vector index. `init_db` sets it.
_vector_index: VectorIndex | None = None


def _replace_vector_index(config: DbConfig) -> None:
    """Open the vector index of the config. Save the old one first, if any."""
    global _vector_index
    if _vector_index is not None and _vector_index.unsaved:
        checkpoint_vectors()
    try:
        _vector_index = VectorIndex(
            config.faiss_path, config.vector_dimension, config.checkpoint_every
        )
    except VectorIndexError as error:
        logger.error("Cannot open the vector index: %s", error)
        raise


def _require_index() -> VectorIndex:
    """Return the open vector index. Raise if `init_db` was not called."""
    if _vector_index is None:
        raise VectorIndexError("The vector index is not open. Call init_db first")
    return _vector_index


def init_db(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[SqliteDatabase, DbConfig]:
    """Open the database named in the config and create missing tables.

    Also open the vector index named in the config. Return the database and
    the resolved config. This function is safe to call more than once.
    """
    assert isinstance(config_path, Path), "config_path must be a Path"
    try:
        config: DbConfig = load_config(config_path)
    except ConfigError as error:
        logger.error("Invalid config: %s", error)
        raise

    try:
        config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        database: SqliteDatabase = SqliteDatabase(
            str(config.sqlite_path), pragmas=_SQLITE_PRAGMAS
        )
        database_proxy.initialize(database)
        database.connect(reuse_if_open=True)
        logger.info("Opened database %s", config.sqlite_path)
        database.create_tables([Image, Region], safe=True)
    except (OSError, PeeweeException) as error:
        logger.error(
            "Cannot open database %s: %s: %s",
            config.sqlite_path,
            type(error).__name__,
            error,
        )
        raise
    logger.info("Tables ready: image, region")
    _replace_vector_index(config)
    return database, config


@contextmanager
def _translate_errors(action: str) -> Iterator[None]:
    """Log peewee errors and raise them as the typed errors of this module."""
    try:
        yield
    except IntegrityError as error:
        logger.error("%s failed: %s: %s", action, type(error).__name__, error)
        if "UNIQUE" in str(error).upper():
            raise DuplicateImageError(f"{action}: value already exists") from error
        raise InvalidRecordError(f"{action}: database refused the value") from error
    except PeeweeException as error:
        logger.error("%s failed: %s: %s", action, type(error).__name__, error)
        raise DbError(f"{action}: database error") from error


def add_image(
    file_path: str, sha256: str, width: int, height: int, room: str | None = None
) -> Image:
    """Insert a new image with the status `pending`.

    `file_path` is relative to the storage root. `captured_at` is the time at
    which this function runs. Raise `DuplicateImageError` if `file_path` or
    `sha256` is already in the database.
    """
    assert isinstance(file_path, str) and file_path, "file_path must be a string"
    assert not file_path.startswith(("/", "\\")), "file_path must be relative"
    assert ".." not in PurePosixPath(file_path).parts, "file_path must not use .."
    assert isinstance(sha256, str) and len(sha256) == 64, "sha256 must have 64 chars"
    assert isinstance(width, int) and width > 0, "width must be a positive int"
    assert isinstance(height, int) and height > 0, "height must be a positive int"
    assert room is None or isinstance(room, str), "room must be a string or None"

    with _translate_errors("add_image"):
        image: Image = Image.create(
            file_path=file_path,
            sha256=sha256,
            width=width,
            height=height,
            room=room,
            captured_at=utc_now(),
        )
    logger.debug("Added image %d with status pending", image.id)
    return image


def find_image_by_sha256(sha256: str) -> Image | None:
    """Return the image with this hash, or None. Use it to detect a duplicate upload."""
    assert isinstance(sha256, str) and len(sha256) == 64, "sha256 must have 64 chars"
    with _translate_errors("find_image_by_sha256"):
        return Image.get_or_none(Image.sha256 == sha256)


def count_images_by_status(status: ImageStatus) -> int:
    """Return how many images have this status. Use it to limit the queue of `pending` images."""
    assert isinstance(status, ImageStatus), "status must be an ImageStatus"
    with _translate_errors("count_images_by_status"):
        return Image.select().where(Image.status == status.value).count()


def list_recent_phashes(since: datetime | None = None, limit: int = 200) -> list[str]:
    """Return the perceptual hashes of `done` images, newest first.

    Use it to check for near-duplicate uploads. SQLite cannot compare hashes
    by Hamming distance, so the caller compares the list. `since` is a naive
    UTC time. Images from before `since` are left out.
    """
    assert since is None or isinstance(since, datetime), "since must be a datetime"
    assert isinstance(limit, int) and limit > 0, "limit must be positive"
    with _translate_errors("list_recent_phashes"):
        query = Image.select(Image.phash).where(
            Image.status == ImageStatus.DONE.value, Image.phash.is_null(False)
        )
        if since is not None:
            query = query.where(Image.captured_at >= since)
        query = query.order_by(Image.captured_at.desc(), Image.id.desc()).limit(limit)
        return [row.phash for row in query]


def get_image(image_id: int) -> Image | None:
    """Return the image with this id, or None."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    with _translate_errors("get_image"):
        return Image.get_or_none(Image.id == image_id)


def claim_next_pending() -> Image | None:
    """Set the oldest `pending` image to `processing` and return it.

    The read and the update run in one write transaction. Two workers cannot
    get the same image. Return None when no image is `pending`.
    """
    with _translate_errors("claim_next_pending"), database_proxy.atomic("IMMEDIATE"):
        image: Image | None = (
            Image.select()
            .where(Image.status == ImageStatus.PENDING.value)
            .order_by(Image.captured_at, Image.id)
            .first()
        )
        if image is None:
            return None
        changed: int = (
            Image.update(status=ImageStatus.PROCESSING.value)
            .where(
                Image.id == image.id,
                Image.status == ImageStatus.PENDING.value,
            )
            .execute()
        )
        assert changed == 1, "claim must change exactly one row"
        image.status = ImageStatus.PROCESSING.value
    logger.info("Claimed image %d for processing", image.id)
    return image


def _finish_image(image_id: int, status: ImageStatus, detail: str | None) -> Image:
    """Move an image from `processing` to a final status."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    assert detail is None or isinstance(detail, str), "detail must be a string"
    assert status in (ImageStatus.DONE, ImageStatus.REJECTED, ImageStatus.FAILED)

    cut_detail: str | None = (
        None if detail is None else detail[:MAX_STATUS_DETAIL_LENGTH]
    )
    with _translate_errors(f"mark image {status.value}"):
        changed: int = (
            Image.update(
                status=status.value, status_detail=cut_detail, processed_at=utc_now()
            )
            .where(
                Image.id == image_id,
                Image.status == ImageStatus.PROCESSING.value,
            )
            .execute()
        )
        if changed != 1:
            raise InvalidStateError(
                f"Image {image_id} does not exist or is not in status processing"
            )
        image: Image = Image.get_by_id(image_id)
    logger.info("Image %d is now %s", image_id, status.value)
    return image


def mark_image_done(image_id: int) -> Image:
    """Set a `processing` image to `done`."""
    return _finish_image(image_id, ImageStatus.DONE, None)


def mark_image_rejected(image_id: int, reason: str) -> Image:
    """Set a `processing` image to `rejected`. `reason` must not hold personal data."""
    assert isinstance(reason, str) and reason, "reason must be a non-empty string"
    return _finish_image(image_id, ImageStatus.REJECTED, reason)


def mark_image_failed(image_id: int, error: str) -> Image:
    """Set a `processing` image to `failed`. `error` must not hold personal data."""
    assert isinstance(error, str) and error, "error must be a non-empty string"
    return _finish_image(image_id, ImageStatus.FAILED, error)


def _require_processing(image_id: int) -> None:
    """Raise `InvalidStateError` unless the image exists and is `processing`."""
    is_processing: bool = (
        Image.select()
        .where(
            Image.id == image_id,
            Image.status == ImageStatus.PROCESSING.value,
        )
        .exists()
    )
    if not is_processing:
        raise InvalidStateError(
            f"Image {image_id} does not exist or is not in status processing"
        )


def set_image_analysis(image_id: int, phash: str, blur_score: float) -> Image:
    """Save the quality-gate values of a `processing` image."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    assert isinstance(phash, str) and 0 < len(phash) <= 16, "phash: 1 to 16 chars"
    assert isinstance(blur_score, float) and blur_score >= 0, "blur_score must be >= 0"

    with _translate_errors("set_image_analysis"):
        changed: int = (
            Image.update(phash=phash, blur_score=blur_score)
            .where(
                Image.id == image_id,
                Image.status == ImageStatus.PROCESSING.value,
            )
            .execute()
        )
        if changed != 1:
            raise InvalidStateError(
                f"Image {image_id} does not exist or is not in status processing"
            )
        return Image.get_by_id(image_id)


def add_regions(image_id: int, regions: list[NewRegion]) -> list[Region]:
    """Insert the regions of a `processing` image in one transaction.

    Return the new rows, with ids set and `embedding_model` empty. The ids are
    the vector ids for the FAISS index. An empty list is allowed.
    """
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    assert isinstance(regions, list), "regions must be a list"
    assert all(isinstance(region, NewRegion) for region in regions)

    with _translate_errors("add_regions"), database_proxy.atomic():
        _require_processing(image_id)
        rows: list[Region] = [
            Region.create(
                image=image_id,
                x0=region.x0,
                y0=region.y0,
                x1=region.x1,
                y1=region.y1,
                det_score=region.det_score,
                source=region.source.value,
                phrases=region.phrases,
                in_hand=region.in_hand,
                quality_ok=region.quality_ok,
            )
            for region in regions
        ]
    logger.debug("Added %d region(s) to image %d", len(rows), image_id)
    return rows


def get_regions(image_id: int) -> list[Region]:
    """Return the regions of an image, ordered by id."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    with _translate_errors("get_regions"):
        return list(Region.select().where(Region.image == image_id).order_by(Region.id))


def mark_regions_embedded(region_ids: list[int], embedding_model: str) -> int:
    """Record that the vectors of these regions are in the FAISS index.

    Call it after the vectors are written. Change all rows or none: if one id
    does not exist, raise `InvalidStateError` and change nothing.
    """
    assert isinstance(region_ids, list) and region_ids, "region_ids must not be empty"
    assert all(isinstance(i, int) and i > 0 for i in region_ids), "ids must be positive"
    assert isinstance(embedding_model, str) and embedding_model, "model name needed"

    unique_ids: set[int] = set(region_ids)
    with _translate_errors("mark_regions_embedded"), database_proxy.atomic():
        changed: int = (
            Region.update(embedding_model=embedding_model)
            .where(Region.id.in_(unique_ids))
            .execute()
        )
        if changed != len(unique_ids):
            raise InvalidStateError(
                f"{len(unique_ids) - changed} region id(s) do not exist"
            )
    logger.debug("Marked %d region(s) as embedded", changed)
    return changed


def list_embedded_region_ids() -> list[int]:
    """Return the ids of all regions that have a vector in the FAISS index.

    The backend compares this list with the ids in FAISS to find differences.
    """
    with _translate_errors("list_embedded_region_ids"):
        query = (
            Region.select(Region.id)
            .where(Region.embedding_model.is_null(False))
            .order_by(Region.id)
        )
        return [row.id for row in query]


def _reset_images_to_pending(image_ids: list[int]) -> list[int]:
    """Delete the regions of these images and set the images to `pending`.

    Call it inside a transaction. Return the ids of the deleted regions that
    had a vector. The caller removes them from the vector index.
    """
    if not image_ids:
        return []
    embedded_ids: list[int] = [
        row.id
        for row in Region.select(Region.id)
        .where(Region.image.in_(image_ids), Region.embedding_model.is_null(False))
        .order_by(Region.id)
    ]
    Region.delete().where(Region.image.in_(image_ids)).execute()
    Image.update(
        status=ImageStatus.PENDING.value, status_detail=None, processed_at=None
    ).where(Image.id.in_(image_ids)).execute()
    return embedded_ids


def recover_interrupted() -> RecoveryReport:
    """Repair the database after a stop. Call it once at start-up.

    This function changes the database only and leaves the vector index
    alone. Most callers need `startup_recovery`, which also repairs the index.

    1. Each `processing` image goes back to `pending`.
    2. The regions of these images are deleted, so the retry cannot make
       duplicates. The ids of those that had a vector are reported.
    3. Regions of `done` images with no vector are reported and not changed.
    """
    with _translate_errors("recover_interrupted"), database_proxy.atomic("IMMEDIATE"):
        interrupted: list[int] = [
            row.id
            for row in Image.select(Image.id)
            .where(Image.status == ImageStatus.PROCESSING.value)
            .order_by(Image.id)
        ]
        stale_vector_ids: list[int] = _reset_images_to_pending(interrupted)
        unembedded: list[int] = [
            row.id
            for row in Region.select(Region.id)
            .join(Image)
            .where(
                Image.status == ImageStatus.DONE.value,
                Region.embedding_model.is_null(),
            )
            .order_by(Region.id)
        ]

    report: RecoveryReport = RecoveryReport(interrupted, stale_vector_ids, unembedded)
    logger.info(
        "Recovery: %d image(s) reset, %d stale vector id(s), %d region(s) without vector",
        len(interrupted),
        len(stale_vector_ids),
        len(unembedded),
    )
    if stale_vector_ids or unembedded:
        logger.warning(
            "Recovery needs action by backend: remove %d id(s) from FAISS, "
            "check %d region(s) without vector",
            len(stale_vector_ids),
            len(unembedded),
        )
    return report


def clear_regions_embedded(region_ids: list[int]) -> int:
    """Set `embedding_model` to empty for these regions. Unknown ids are ignored.

    Return the number of rows changed.
    """
    assert isinstance(region_ids, list), "region_ids must be a list"
    assert all(isinstance(i, int) and i > 0 for i in region_ids), "ids must be positive"
    if not region_ids:
        return 0
    with _translate_errors("clear_regions_embedded"):
        return (
            Region.update(embedding_model=None)
            .where(Region.id.in_(set(region_ids)))
            .execute()
        )


def store_region_vectors(
    region_ids: list[int], vectors: np.ndarray, embedding_model: str
) -> None:
    """Add the vectors of regions to the index and mark the regions as embedded.

    Do both steps or none. `vectors` has one float32 row of unit length for
    each id, in the same order. The index is written to disk when enough
    changes are unsaved (see `checkpoint_every` in `config.toml`).
    """
    assert isinstance(region_ids, list) and region_ids, "region_ids must not be empty"
    assert isinstance(embedding_model, str) and embedding_model, "model name needed"

    index: VectorIndex = _require_index()
    index.add(region_ids, vectors)
    try:
        mark_regions_embedded(region_ids, embedding_model)
    except DbError:
        index.remove(region_ids)  # Keep the index and the tables in step
        raise
    logger.debug("Stored %d vector(s)", len(region_ids))
    index.checkpoint_if_due()


def remove_region_vectors(region_ids: list[int]) -> int:
    """Remove the vectors of regions from the index and clear their marks.

    Ids that are not in the index are ignored. Return how many vectors were
    removed from the index.
    """
    assert isinstance(region_ids, list), "region_ids must be a list"
    assert all(isinstance(i, int) and i > 0 for i in region_ids), "ids must be positive"

    index: VectorIndex = _require_index()
    removed: int = index.remove(region_ids)
    clear_regions_embedded(region_ids)
    logger.info("Removed %d vector(s) from the index", removed)
    index.checkpoint_if_due()
    return removed


def checkpoint_vectors() -> None:
    """Write the vector index to disk now.

    The application calls this when it stops. Raise `VectorIndexError` if the
    write fails.
    """
    _require_index().save()


def vector_count() -> int:
    """Return the number of vectors in the index."""
    return _require_index().count


# Statuses of images that no worker uses. Only these are old enough to be deleted.
_FINAL_STATUSES: tuple[str, ...] = (
    ImageStatus.DONE.value,
    ImageStatus.REJECTED.value,
    ImageStatus.FAILED.value,
)


def list_images_older_than(cutoff: datetime, limit: int = 100) -> list[Image]:
    """Return the oldest images that were captured before `cutoff`.

    Only `done`, `rejected` and `failed` images are listed, so an image that
    waits for a worker or is in use is never a candidate. `cutoff` is a naive
    UTC time. The oldest image comes first.
    """
    assert isinstance(cutoff, datetime), "cutoff must be a datetime"
    assert isinstance(limit, int) and limit > 0, "limit must be positive"
    with _translate_errors("list_images_older_than"):
        return list(
            Image.select()
            .where(Image.status.in_(_FINAL_STATUSES), Image.captured_at < cutoff)
            .order_by(Image.captured_at, Image.id)
            .limit(limit)
        )


def list_image_paths() -> list[str]:
    """Return the `file_path` of every image, sorted. Use it to compare with the files on disk."""
    with _translate_errors("list_image_paths"):
        return [
            row.file_path
            for row in Image.select(Image.file_path).order_by(Image.file_path)
        ]


def delete_images(image_ids: list[int]) -> list[str]:
    """Delete images, their regions and their vectors. Return their `file_path` values.

    The caller deletes the files after this call. The rows go first on
    purpose: a stop in between leaves files that no row owns (found by
    `storage.find_orphan_files`), and not rows without a file.
    Unknown ids are ignored. If one image is `processing`, raise
    `InvalidStateError` and delete nothing.
    """
    assert isinstance(image_ids, list), "image_ids must be a list"
    assert all(isinstance(i, int) and i > 0 for i in image_ids), "ids must be positive"
    if not image_ids:
        return []

    index: VectorIndex = _require_index()
    with _translate_errors("delete_images"), database_proxy.atomic("IMMEDIATE"):
        images: list[Image] = list(Image.select().where(Image.id.in_(set(image_ids))))
        busy: list[int] = [
            i.id for i in images if i.status == ImageStatus.PROCESSING.value
        ]
        if busy:
            raise InvalidStateError(f"Images are in use and cannot be deleted: {busy}")
        found_ids: list[int] = [i.id for i in images]
        vector_ids: list[int] = [
            row.id
            for row in Region.select(Region.id).where(
                Region.image.in_(found_ids), Region.embedding_model.is_null(False)
            )
        ]
        paths: list[str] = [i.file_path for i in images]
        Image.delete().where(Image.id.in_(found_ids)).execute()  # Regions go by cascade

    index.remove(vector_ids)
    index.checkpoint_if_due()
    logger.info("Deleted %d image(s) and %d vector(s)", len(paths), len(vector_ids))
    return paths


def list_vector_ids() -> list[int]:
    """Return the region ids that have a vector in the index, sorted."""
    return sorted(_require_index().ids())


def _requeue_images_with_gaps() -> tuple[list[int], list[int]]:
    """Set `done` images with a region without a vector back to `pending`.

    Return the image ids and the ids of the deleted regions that had a vector.
    """
    with (
        _translate_errors("requeue images with gaps"),
        database_proxy.atomic("IMMEDIATE"),
    ):
        gap_images = Region.select(Region.image).where(Region.embedding_model.is_null())
        image_ids: list[int] = [
            row.id
            for row in Image.select(Image.id)
            .where(Image.status == ImageStatus.DONE.value, Image.id.in_(gap_images))
            .order_by(Image.id)
        ]
        return image_ids, _reset_images_to_pending(image_ids)


def startup_recovery() -> StartupReport:
    """Repair the tables and the vector index after a stop.

    Call it once after `init_db` and before the workers start. The steps:

    1. Each `processing` image goes back to `pending`. Its regions and
       vectors are deleted, so the retry cannot make duplicates.
    2. A vector that has no embedded region in the tables is removed.
    3. An embedded mark that has no vector in the index is cleared.
    4. A `done` image that has a region without a vector goes back to
       `pending`. Its regions and vectors are deleted. The workers process
       the image again.

    Rule for the caller: every region of a `done` image must have a vector.
    A region that never gets a vector would send its image back to `pending`
    at every start.
    """
    index: VectorIndex = _require_index()

    interrupted: RecoveryReport = recover_interrupted()
    index.remove(interrupted.stale_vector_ids)

    marked_ids: set[int] = set(list_embedded_region_ids())
    vector_ids: set[int] = index.ids()
    orphan_vectors_removed: int = index.remove(sorted(vector_ids - marked_ids))
    missing_vectors_cleared: int = clear_regions_embedded(
        sorted(marked_ids - vector_ids)
    )

    requeued_image_ids, gap_vector_ids = _requeue_images_with_gaps()
    index.remove(gap_vector_ids)
    index.checkpoint_if_due()

    report: StartupReport = StartupReport(
        interrupted_image_ids=interrupted.reset_image_ids,
        requeued_image_ids=requeued_image_ids,
        orphan_vectors_removed=orphan_vectors_removed,
        missing_vectors_cleared=missing_vectors_cleared,
    )
    logger.info(
        "Start-up recovery: %d interrupted image(s), %d requeued image(s), "
        "%d orphan vector(s) removed, %d missing vector(s) cleared",
        len(report.interrupted_image_ids),
        len(report.requeued_image_ids),
        report.orphan_vectors_removed,
        report.missing_vectors_cleared,
    )
    if requeued_image_ids or orphan_vectors_removed or missing_vectors_cleared:
        logger.warning("Start-up recovery repaired the tables or the vector index")
    return report
