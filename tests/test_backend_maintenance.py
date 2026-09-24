"""Tests of the maintenance jobs. They use the actual storage, database and index.

The retention tests use a retention time of 9000 days and images that were captured
in the year 2000, so only the test images are old enough to be deleted.
"""

import dataclasses
import os
import time
import uuid
from collections.abc import Iterator
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from backend.maintenance import Maintenance, MaintenanceReport
from backend.settings import BackendConfig, MaintenanceConfig, load_config
from db.schema import Image, ImageStatus
from tests.db_support import MODEL, unit_vectors

OLD: datetime = datetime(2000, 1, 1)
RETENTION_DAYS: int = 9000


def picture() -> bytes:
    rng: np.random.Generator = np.random.default_rng(uuid.uuid4().int % (2**32))
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)).save(
        buffer, "JPEG"
    )
    return buffer.getvalue()


def make(**changes: int) -> Maintenance:
    base: BackendConfig = load_config()
    config: BackendConfig = dataclasses.replace(
        base, maintenance=dataclasses.replace(base.maintenance, **changes)
    )
    return Maintenance(config)


class Made:
    """The images and files that a test made. They are removed at the end."""

    def __init__(self) -> None:
        self.image_ids: list[int] = []
        self.orphan_paths: list[str] = []
        self.object_ids: list[int] = []

    def image(
        self, status: ImageStatus = ImageStatus.DONE, captured_at: datetime = OLD
    ) -> Image:
        prepared = storage.prepare_image(picture())
        path: str = storage.save_image(prepared)
        row: Image = db.add_image(
            path, prepared.sha256, prepared.width, prepared.height
        )
        Image.update(status=status.value, captured_at=captured_at).where(
            Image.id == row.id
        ).execute()
        self.image_ids.append(row.id)
        return row

    def reference_photo(self, age_seconds: int) -> tuple[int, str]:
        """A reference photo with its row and file. The file has a given age."""
        obj = db.gallery.create_object(f"__test__ {uuid.uuid4().hex[:8]}", "a bottle")
        self.object_ids.append(obj.id)
        prepared = storage.prepare_image(picture())
        path: str = storage.save_image(prepared)
        self.orphan_paths.append(
            path
        )  # Removed at the end, if the test did not remove it
        photo = db.gallery.add_reference_photo(
            obj.id, path, prepared.sha256, prepared.width, prepared.height
        )
        old: float = time.time() - age_seconds
        os.utime(storage.current_config().root / path, (old, old))
        return photo.id, path

    def orphan(self, age_seconds: int) -> str:
        """A file that no row owns, with a given age."""
        path: str = storage.save_image(storage.prepare_image(picture()))
        old: float = time.time() - age_seconds
        os.utime(storage.current_config().root / path, (old, old))
        self.orphan_paths.append(path)
        return path


@pytest.fixture(autouse=True)
def actual_system(monkeypatch: pytest.MonkeyPatch) -> Iterator[Made]:
    config = storage.init_storage()
    database, _ = db.init_db()
    if storage.find_orphan_files(db.list_all_file_paths()):
        pytest.skip(
            "The actual storage has orphan files. The job would delete some of them"
        )
    monkeypatch.setattr(
        storage, "_config", dataclasses.replace(config, retention_days=0)
    )
    made: Made = Made()
    yield made
    existing: list[Image] = list(Image.select().where(Image.id.in_(made.image_ids)))
    if existing:
        storage.delete_images(db.delete_images([i.id for i in existing]))
    for object_id in made.object_ids:
        storage.delete_images(db.gallery.delete_object(object_id))
    storage.delete_images(made.orphan_paths)
    db.checkpoint_vectors()  # Keep the actual index file free of test vectors
    database.close()


def set_retention(monkeypatch: pytest.MonkeyPatch, days: int) -> None:
    monkeypatch.setattr(
        storage,
        "_config",
        dataclasses.replace(storage.current_config(), retention_days=days),
    )


def exists(path: str) -> bool:
    return (storage.current_config().root / path).exists()


# ---- Retention ----


def test_retention_is_off_when_the_days_are_zero(actual_system: Made) -> None:
    old: Image = actual_system.image()
    report: MaintenanceReport = make().run_once()
    assert report.expired_images_deleted == 0
    assert db.get_image(old.id) is not None and exists(old.file_path)


def test_retention_deletes_old_finished_images_with_their_files_and_vectors(
    actual_system: Made, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_retention(monkeypatch, RETENTION_DAYS)
    done_old: Image = actual_system.image(ImageStatus.DONE)
    failed_old: Image = actual_system.image(ImageStatus.FAILED)
    rejected_old: Image = actual_system.image(ImageStatus.REJECTED)
    pending_old: Image = actual_system.image(ImageStatus.PENDING)
    recent: Image = actual_system.image(ImageStatus.DONE, captured_at=datetime.now())

    # Give the old done image a region with a vector.
    set_processing = Image.update(status=ImageStatus.PROCESSING.value).where(
        Image.id == done_old.id
    )
    set_processing.execute()
    region = db.add_regions(
        done_old.id, [db.NewRegion(1, 1, 20, 20, 0.5, db.RegionSource.FULL, ["a cup"])]
    )[0]
    db.store_region_vectors([region.id], unit_vectors(1), MODEL)
    Image.update(status=ImageStatus.DONE.value).where(Image.id == done_old.id).execute()
    assert region.id in db.list_vector_ids()

    report: MaintenanceReport = make().run_once()

    assert report.expired_images_deleted == 3
    for gone in (done_old, failed_old, rejected_old):
        assert db.get_image(gone.id) is None and not exists(gone.file_path)
    assert region.id not in db.list_vector_ids()
    for kept in (pending_old, recent):  # Not finished, or not old
        assert db.get_image(kept.id) is not None and exists(kept.file_path)


def test_retention_works_in_groups(
    actual_system: Made, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_retention(monkeypatch, RETENTION_DAYS)
    images: list[Image] = [actual_system.image() for _ in range(5)]
    report: MaintenanceReport = make(retention_batch=2).run_once()
    assert report.expired_images_deleted == 5
    assert all(db.get_image(i.id) is None for i in images)


def test_a_failure_of_retention_does_not_stop_the_other_jobs(
    actual_system: Made, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_retention(monkeypatch, RETENTION_DAYS)
    old_orphan: str = actual_system.orphan(age_seconds=7200)

    def fail(*args: object, **kwargs: object) -> list[Image]:
        raise db.DbError("database error")

    monkeypatch.setattr(db, "list_images_older_than", fail)
    report: MaintenanceReport = make().run_once()
    assert report.expired_images_deleted == 0
    assert report.orphan_files_deleted == 1 and not exists(old_orphan)
    assert report.checkpoint_saved


# ---- Orphan files ----


def test_old_orphan_files_are_deleted_and_new_ones_and_known_files_stay(
    actual_system: Made,
) -> None:
    known: Image = actual_system.image(captured_at=datetime.now())
    old_orphan: str = actual_system.orphan(age_seconds=7200)
    new_orphan: str = actual_system.orphan(
        age_seconds=10
    )  # Maybe an upload that has no row yet
    report: MaintenanceReport = make().run_once()
    assert (report.orphan_files_found, report.orphan_files_deleted) == (1, 1)
    assert not exists(old_orphan)
    assert exists(new_orphan) and exists(known.file_path)


def test_the_orphan_limit_stops_a_mass_delete(actual_system: Made) -> None:
    paths: list[str] = [actual_system.orphan(age_seconds=7200) for _ in range(3)]
    report: MaintenanceReport = make(orphan_delete_limit=1).run_once()
    assert (report.orphan_files_found, report.orphan_files_deleted) == (3, 1)
    assert sum(exists(p) for p in paths) == 2


def test_a_limit_of_zero_deletes_no_orphan_file(actual_system: Made) -> None:
    path: str = actual_system.orphan(age_seconds=7200)
    report: MaintenanceReport = make(orphan_delete_limit=0).run_once()
    assert (report.orphan_files_found, report.orphan_files_deleted) == (
        1,
        0,
    ) and exists(path)


# ---- Reference photos ----


def test_a_reference_photo_is_not_an_orphan_while_its_row_exists(
    actual_system: Made,
) -> None:
    photo_id, path = actual_system.reference_photo(
        age_seconds=7200
    )  # Older than the grace time
    report: MaintenanceReport = make().run_once()
    assert report.orphan_files_found == 0 and report.missing_files == 0 and exists(path)
    db.gallery.remove_photo(photo_id)  # Now no row owns the file
    report = make().run_once()
    assert (
        report.orphan_files_found == 1
        and report.orphan_files_deleted == 1
        and not exists(path)
    )


def test_retention_never_deletes_reference_photos(
    actual_system: Made, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A long time, so that a test never expires a real image.
    set_retention(monkeypatch, RETENTION_DAYS)
    _, path = actual_system.reference_photo(age_seconds=10 * 24 * 3600)
    report: MaintenanceReport = make().run_once()
    assert report.expired_images_deleted == 0 and exists(path)


def test_a_missing_reference_photo_file_is_reported(actual_system: Made) -> None:
    _, path = actual_system.reference_photo(age_seconds=10)
    storage.delete_images([path])
    assert make().run_once().missing_files == 1


# ---- Missing files and the checkpoint ----


def test_a_missing_file_is_reported_and_nothing_is_changed(actual_system: Made) -> None:
    image: Image = actual_system.image(captured_at=datetime.now())
    storage.delete_images([image.file_path])
    report: MaintenanceReport = make().run_once()
    assert report.missing_files == 1 and db.get_image(image.id) is not None


def test_the_index_is_saved_and_a_save_error_does_not_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert make().run_once().checkpoint_saved is True

    def fail() -> None:
        raise db.VectorIndexError("disk full")

    with monkeypatch.context() as patch:  # The fixture saves the index again at the end
        patch.setattr(db, "checkpoint_vectors", fail)
        assert make().run_once().checkpoint_saved is False


# ---- The thread ----


def test_the_thread_runs_at_once_and_stops_quickly() -> None:
    maintenance: Maintenance = make(interval_minutes=60)
    assert maintenance.last_report is None
    maintenance.start()
    try:
        deadline: float = time.monotonic() + 10
        while maintenance.last_report is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (
            maintenance.last_report is not None
        )  # The first run did not wait for the interval
        with pytest.raises(RuntimeError):
            maintenance.start()
    finally:
        started: float = time.monotonic()
        maintenance.stop(timeout=10)
    assert time.monotonic() - started < 3 and not maintenance.is_running
    maintenance.stop(timeout=1)  # A second stop does no harm


def test_the_parameters_are_checked() -> None:
    with pytest.raises(AssertionError):
        Maintenance("config")  # type: ignore[arg-type]
    assert MaintenanceConfig(1, 0, 1).orphan_delete_limit == 0
    _: Path = Path()
