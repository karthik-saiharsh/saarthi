"""Tests of the upload step. They use the actual storage root and the actual database.

Each test removes the files and rows that it made.
"""

import dataclasses
from collections.abc import Iterator
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from backend.errors import QueueFullError
from backend.ingest import IngestResult, IngestService, IngestStatus
from backend.settings import load_config
from db.schema import Image, ImageStatus


def picture(seed: int = 0, fmt: str = "JPEG", **kwargs: object) -> bytes:
    rng: np.random.Generator = np.random.default_rng(seed)
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
        buffer, format=fmt, **kwargs
    )
    return buffer.getvalue()


@pytest.fixture(scope="module", autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    yield
    database.close()


@pytest.fixture()
def service() -> Iterator[IngestService]:
    """A service. At the end, remove every image that has the hash of a test picture."""
    seeds_before: set[str] = set(db.list_image_paths())
    yield IngestService(load_config())
    made: list[Image] = [i for i in Image.select() if i.file_path not in seeds_before]
    if made:
        paths: list[str] = db.delete_images([i.id for i in made])
        storage.delete_images(paths)


def stored_files() -> set[str]:
    return set(storage.list_stored_paths())


def test_a_new_image_is_stored_and_waits_for_the_worker(service: IngestService) -> None:
    result: IngestResult = service.submit(picture(1))
    row: Image | None = db.get_image(result.image_id)
    assert result.status == IngestStatus.STORED
    assert row is not None and row.status == ImageStatus.PENDING.value
    assert (row.width, row.height) == (80, 60)
    assert row.file_path in stored_files()
    assert db.find_image_by_sha256(row.sha256).id == row.id  # type: ignore[union-attr]
    assert storage.read_image(row.file_path)[:2] == b"\xff\xd8"  # A JPEG


def test_the_same_upload_twice_gives_one_image(service: IngestService) -> None:
    first: IngestResult = service.submit(picture(2))
    files_before: set[str] = stored_files()
    second: IngestResult = service.submit(picture(2))
    assert second.status == IngestStatus.DUPLICATE and second.image_id == first.image_id
    assert stored_files() == files_before  # No second file


def test_the_same_picture_with_other_metadata_is_a_duplicate(
    service: IngestService,
) -> None:
    exif: PilImage.Exif = PilImage.Exif()
    exif[0x010F] = "Some Maker"
    first: IngestResult = service.submit(picture(3))
    second: IngestResult = service.submit(picture(3, exif=exif.tobytes()))
    assert second.status == IngestStatus.DUPLICATE and second.image_id == first.image_id


def test_a_png_and_a_different_picture_are_new_images(service: IngestService) -> None:
    first: IngestResult = service.submit(picture(4))
    other: IngestResult = service.submit(picture(5, "PNG"))
    assert other.status == IngestStatus.STORED and other.image_id != first.image_id


def test_a_file_that_is_not_an_image_is_refused_and_leaves_nothing(
    service: IngestService,
) -> None:
    files_before: set[str] = stored_files()
    count_before: int = Image.select().count()
    with pytest.raises(storage.UnsupportedImageError):
        service.submit(b"not an image")
    assert stored_files() == files_before and Image.select().count() == count_before


def test_a_file_that_is_too_large_is_refused(
    service: IngestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    small: storage.StorageConfig = dataclasses.replace(storage._config, max_bytes=100)  # type: ignore[arg-type]
    monkeypatch.setattr(storage, "_config", small)
    with pytest.raises(storage.ImageTooLargeError):
        service.submit(picture(6))


def test_a_full_queue_refuses_new_uploads(service: IngestService) -> None:
    waiting: int = db.count_images_by_status(ImageStatus.PENDING)
    limited: IngestService = IngestService(
        dataclasses.replace(load_config(), max_pending=waiting + 1, token_env_var="X")
    )
    assert limited.submit(picture(7)).status == IngestStatus.STORED
    with pytest.raises(QueueFullError):
        limited.submit(picture(8))
    # A duplicate also needs a free place, because the check comes first.
    with pytest.raises(QueueFullError):
        limited.submit(picture(7))


def test_a_failed_row_deletes_the_file_again(
    service: IngestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    files_before: set[str] = stored_files()

    def fail(*args: object, **kwargs: object) -> None:
        raise db.DbError("database error")

    monkeypatch.setattr(db, "add_image", fail)
    with pytest.raises(db.DbError):
        service.submit(picture(9))
    assert stored_files() == files_before


def test_a_full_disk_leaves_no_row(
    service: IngestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    count_before: int = Image.select().count()

    def fail(prepared: object) -> str:
        raise storage.StorageFullError("full")

    monkeypatch.setattr(storage, "save_image", fail)
    with pytest.raises(storage.StorageFullError):
        service.submit(picture(10))
    assert Image.select().count() == count_before


def test_two_uploads_of_the_same_picture_at_the_same_time_give_one_image(
    service: IngestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    winner: IngestResult = service.submit(picture(11))
    files_before: set[str] = stored_files()
    real_find = db.find_image_by_sha256
    calls: list[int] = []

    def blind_first_time(sha256: str) -> Image | None:
        calls.append(1)
        return (
            None if len(calls) == 1 else real_find(sha256)
        )  # The first look misses the winner

    monkeypatch.setattr(db, "find_image_by_sha256", blind_first_time)
    late: IngestResult = service.submit(picture(11))
    assert late.status == IngestStatus.DUPLICATE and late.image_id == winner.image_id
    assert stored_files() == files_before  # The late file was deleted


def test_parameters_are_checked(service: IngestService) -> None:
    with pytest.raises(AssertionError):
        service.submit("text")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        IngestService("config")  # type: ignore[arg-type]


def test_a_hook_runs_for_a_new_image_only_and_its_failure_does_not_fail_the_upload(
    service: IngestService,  # Only for its clean-up of the rows and files
) -> None:
    calls: list[int] = []

    def hook() -> None:
        calls.append(1)
        raise RuntimeError("the hook is broken")

    service: IngestService = IngestService(load_config(), on_stored=hook)
    first: IngestResult = service.submit(picture(40))
    assert first.status == IngestStatus.STORED and calls == [1]
    assert service.submit(picture(40)).status == IngestStatus.DUPLICATE
    assert calls == [1]  # No call for a duplicate
    with pytest.raises(AssertionError):
        IngestService(load_config(), on_stored="no")  # type: ignore[arg-type]
