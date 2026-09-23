"""Tests of the worker. They use the actual storage, database and index.

The ai models are fakes, so the tests are fast. The quality gate, the image
decoding and all the post-processing are the real code.
"""

import dataclasses
import time
from collections.abc import Callable, Iterator
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import InferenceError
from ai.types import Box, RawDetection
from backend.ingest import IngestService
from backend.settings import BackendConfig, load_config
from backend.worker import REASON_BLURRY, REASON_DUPLICATE, Worker
from db.schema import Image, ImageStatus, Region
from tests.ai_support import FakeDetector, FakeEmbedder, make_engine
from tests.db_support import skip_if_actual_data_has_problems


def noise(seed: int, jitter: int = 0) -> bytes:
    """A sharp picture. `jitter` adds a little noise: a near duplicate of the same seed."""
    rng: np.random.Generator = np.random.default_rng(seed)
    pixels: np.ndarray = rng.integers(0, 255, (60, 80, 3), dtype=np.int16)
    if jitter:
        pixels = pixels + np.random.default_rng(99).integers(
            -jitter, jitter + 1, pixels.shape
        )
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(
        buffer, "JPEG", quality=95
    )
    return buffer.getvalue()


def flat() -> bytes:
    buffer: BytesIO = BytesIO()
    PilImage.new("RGB", (80, 60), (120, 120, 120)).save(buffer, "JPEG")
    return buffer.getvalue()


@dataclasses.dataclass
class Setup:
    worker: Worker
    detector: FakeDetector
    embedder: FakeEmbedder
    ingest: IngestService
    config: BackendConfig

    def upload(self, data: bytes) -> int:
        return self.ingest.submit(data).image_id


def make_setup(**worker_changes: int) -> Setup:
    base: BackendConfig = load_config()
    config: BackendConfig = dataclasses.replace(
        base, worker=dataclasses.replace(base.worker, **worker_changes)
    )
    detector, embedder = FakeDetector(), FakeEmbedder()
    engine: AiEngine = make_engine(detector, embedder)
    worker: Worker = Worker(config, engine)
    return Setup(
        worker, detector, embedder, IngestService(config, on_stored=worker.wake), config
    )


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    if Image.select().where(Image.status == ImageStatus.PENDING.value).exists():
        pytest.skip(
            "Actual database has pending images. The worker would claim one of them"
        )
    before: set[str] = set(db.list_image_paths())
    yield
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))
    database.close()


@pytest.fixture()
def setup() -> Setup:
    return make_setup(generic_pass_every=0)


def state(image_id: int) -> Image:
    image: Image | None = db.get_image(image_id)
    assert image is not None
    return image


def wait_for(condition: Callable[[], bool], seconds: float = 15.0) -> None:
    deadline: float = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("The condition did not become true in time")


# ---- One image ----


def test_an_image_is_analysed_and_marked_done(setup: Setup) -> None:
    setup.detector.detections = [
        RawDetection(Box(10, 10, 40, 40), 0.6, "bottle", "full"),
        RawDetection(Box(45, 20, 75, 55), 0.5, "keys", "full"),
    ]
    image_id: int = setup.upload(noise(1))
    assert setup.worker.process_next() is True

    image: Image = state(image_id)
    assert image.status == ImageStatus.DONE.value and image.processed_at is not None
    assert image.phash is not None and image.blur_score is not None
    regions: list[Region] = db.get_regions(image_id)
    assert len(regions) == 2
    assert {r.embedding_model for r in regions} == {"fake-embedder@00000000"}
    assert {r.id for r in regions} <= set(db.list_vector_ids())
    assert sorted(r.phrases[0] for r in regions) == ["a bottle", "a set of keys"]
    assert (regions[0].x0, regions[0].y0, regions[0].x1, regions[0].y1) == (
        10,
        10,
        40,
        40,
    )


def test_an_image_without_candidates_is_done_with_no_regions(setup: Setup) -> None:
    image_id: int = setup.upload(noise(2))
    assert setup.worker.process_next() is True
    assert (
        state(image_id).status == ImageStatus.DONE.value
        and db.get_regions(image_id) == []
    )


def test_no_waiting_image_means_nothing_to_do(setup: Setup) -> None:
    assert setup.worker.process_next() is False


# ---- Gates ----


def test_a_blurry_image_is_rejected_before_detection(setup: Setup) -> None:
    image_id: int = setup.upload(flat())
    assert setup.worker.process_next() is True
    image: Image = state(image_id)
    assert (
        image.status == ImageStatus.REJECTED.value
        and image.status_detail == REASON_BLURRY
    )
    assert image.blur_score == 0.0 and setup.detector.generic_flags == []


def test_a_near_duplicate_of_a_recent_image_is_rejected(setup: Setup) -> None:
    first: int = setup.upload(noise(3))
    second: int = setup.upload(
        noise(3, jitter=2)
    )  # Other bytes, nearly the same picture
    assert first != second
    assert setup.worker.process_next() and setup.worker.process_next()
    assert state(first).status == ImageStatus.DONE.value
    rejected: Image = state(second)
    assert (
        rejected.status == ImageStatus.REJECTED.value
        and rejected.status_detail == REASON_DUPLICATE
    )
    assert len(setup.detector.generic_flags) == 1  # Only the first image was analysed


def test_different_pictures_are_not_duplicates(setup: Setup) -> None:
    ids: list[int] = [setup.upload(noise(seed)) for seed in (4, 5, 6)]
    while setup.worker.process_next():
        pass
    assert [state(i).status for i in ids] == [ImageStatus.DONE.value] * 3


# ---- Generic pass ----


@pytest.mark.parametrize(
    ("every", "expected"),
    [(2, [False, True, False, True]), (1, [True] * 4), (0, [False] * 4)],
)
def test_the_generic_pass_runs_on_every_nth_analysed_image(
    every: int, expected: list[bool]
) -> None:
    setup: Setup = make_setup(generic_pass_every=every)
    for seed in (10, 11, 12, 13):
        setup.upload(noise(seed))
    while setup.worker.process_next():
        pass
    assert setup.detector.generic_flags == expected


# ---- Failures ----


def test_a_detector_error_fails_one_image_and_the_next_one_still_works(
    setup: Setup,
) -> None:
    setup.detector.error = InferenceError("model broke: /secret/path")
    bad: int = setup.upload(noise(20))
    good: int = setup.upload(noise(21))
    assert setup.worker.process_next() and setup.worker.process_next()
    failed: Image = state(bad)
    assert failed.status == ImageStatus.FAILED.value
    assert failed.status_detail == "InferenceError"  # The class name, not the message
    assert db.get_regions(bad) == [] and state(good).status == ImageStatus.DONE.value


def test_an_embedder_error_leaves_no_regions(setup: Setup) -> None:
    setup.detector.detections = [
        RawDetection(Box(10, 10, 40, 40), 0.6, "bottle", "full")
    ]
    setup.embedder.error = InferenceError("no memory")
    image_id: int = setup.upload(noise(22))
    assert setup.worker.process_next() is True
    assert state(image_id).status == ImageStatus.FAILED.value
    assert db.get_regions(image_id) == []  # The vectors come before the region rows


def test_a_missing_file_fails_the_image(setup: Setup) -> None:
    image_id: int = setup.upload(noise(23))
    storage.delete_images([state(image_id).file_path])
    assert setup.worker.process_next() is True
    failed: Image = state(image_id)
    assert (
        failed.status == ImageStatus.FAILED.value
        and failed.status_detail == "StorageNotFoundError"
    )


def test_an_unexpected_error_fails_the_image_and_does_not_escape(setup: Setup) -> None:
    setup.detector.error = RuntimeError("surprise")
    image_id: int = setup.upload(noise(24))
    assert setup.worker.process_next() is True
    failed: Image = state(image_id)
    assert (
        failed.status == ImageStatus.FAILED.value
        and failed.status_detail == "RuntimeError"
    )


# ---- The thread ----


def test_the_thread_processes_an_upload_at_once_because_of_the_wake_up() -> None:
    setup: Setup = make_setup(
        idle_wait_seconds=60, generic_pass_every=0
    )  # The safety net is far away
    setup.worker.start()
    try:
        time.sleep(0.3)  # The worker is idle now
        started: float = time.monotonic()
        image_id: int = setup.upload(noise(30))
        wait_for(lambda: state(image_id).status == ImageStatus.DONE.value, seconds=10)
        assert time.monotonic() - started < 10
    finally:
        setup.worker.stop(timeout=15)
    assert not setup.worker.is_running


def test_stop_returns_quickly_when_the_worker_is_idle() -> None:
    setup: Setup = make_setup(idle_wait_seconds=60)
    setup.worker.start()
    time.sleep(0.3)
    started: float = time.monotonic()
    setup.worker.stop(timeout=10)
    assert time.monotonic() - started < 3 and not setup.worker.is_running
    setup.worker.stop(timeout=1)  # A second stop does no harm


def test_the_worker_cannot_start_twice() -> None:
    setup: Setup = make_setup()
    setup.worker.start()
    try:
        with pytest.raises(RuntimeError):
            setup.worker.start()
    finally:
        setup.worker.stop(timeout=10)


def test_the_loop_survives_an_error_in_its_own_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup: Setup = make_setup(
        idle_wait_seconds=1, error_pause_seconds=0, generic_pass_every=0
    )
    real_claim = db.claim_next_pending
    calls: list[int] = []

    def flaky() -> Image | None:
        calls.append(1)
        if len(calls) == 1:
            raise db.DbError("the database is busy")
        return real_claim()

    monkeypatch.setattr(db, "claim_next_pending", flaky)
    image_id: int = setup.upload(noise(31))
    setup.worker.start()
    try:
        wait_for(lambda: state(image_id).status == ImageStatus.DONE.value, seconds=10)
    finally:
        setup.worker.stop(timeout=15)


def test_the_worker_checks_its_parameters() -> None:
    with pytest.raises(AssertionError):
        Worker("config", None)  # type: ignore[arg-type]
