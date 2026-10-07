"""Tests of the UI adapter and the Backend class.

They use the actual storage and database. Each test removes what it made.
"""

import asyncio
import time
from collections.abc import Iterator
from io import BytesIO

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from backend import ui_adapter
from backend.api import Backend
from backend.errors import BackendConfigError, QueueFullError
from backend.http import UPLOAD_ROUTE
from backend.ingest import IngestResult, IngestService, IngestStatus
from backend.settings import load_config
from backend.ui_adapter import UiUploadService
from db.schema import Image
from tests.ai_support import FakeEmbedder, FakeSceneClassifier, make_engine
from ui.models import ItemStatus
from ui.ports import OutcomeStatus, UploadOutcome, UploadService
from ui.upload_session import UploadSession


def picture(seed: int = 0) -> bytes:
    rng: np.random.Generator = np.random.default_rng(seed)
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
        buffer, "JPEG"
    )
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    before: set[str] = set(db.list_image_paths())
    yield
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))
    database.close()


class FakeService(IngestService):
    def __init__(
        self, error: Exception | None = None, status: IngestStatus = IngestStatus.STORED
    ) -> None:
        super().__init__(load_config())
        self.error: Exception | None = error
        self.status: IngestStatus = status

    def submit(self, data: bytes) -> IngestResult:
        if self.error is not None:
            raise self.error
        return IngestResult(self.status, 3)


def submit(service: IngestService, data: bytes = b"data") -> UploadOutcome:
    return asyncio.run(
        UiUploadService(service).submit_image("photo.jpg", "image/jpeg", data)
    )


@pytest.mark.parametrize(
    ("service", "status", "message"),
    [
        (FakeService(), OutcomeStatus.STORED, ""),
        (
            FakeService(status=IngestStatus.DUPLICATE),
            OutcomeStatus.DUPLICATE,
            ui_adapter.MESSAGE_DUPLICATE,
        ),
        (
            FakeService(storage.UnsupportedImageError("x")),
            OutcomeStatus.REJECTED,
            ui_adapter.MESSAGE_UNSUPPORTED,
        ),
        (
            FakeService(storage.ImageTooLargeError("x")),
            OutcomeStatus.REJECTED,
            ui_adapter.MESSAGE_TOO_LARGE,
        ),
        (
            FakeService(QueueFullError("x")),
            OutcomeStatus.FAILED,
            ui_adapter.MESSAGE_BUSY,
        ),
        (
            FakeService(storage.StorageFullError("x")),
            OutcomeStatus.FAILED,
            ui_adapter.MESSAGE_DISK_FULL,
        ),
        (
            FakeService(storage.StorageIOError("/secret/path")),
            OutcomeStatus.FAILED,
            ui_adapter.MESSAGE_FAILED,
        ),
        (
            FakeService(db.DbError("SELECT secret")),
            OutcomeStatus.FAILED,
            ui_adapter.MESSAGE_FAILED,
        ),
        (
            FakeService(RuntimeError("internal detail")),
            OutcomeStatus.FAILED,
            ui_adapter.MESSAGE_FAILED,
        ),
    ],
)
def test_each_result_becomes_an_outcome_without_internal_details(
    service: IngestService, status: OutcomeStatus, message: str
) -> None:
    assert submit(service) == UploadOutcome(status, message)


def test_the_adapter_checks_its_parameters() -> None:
    with pytest.raises(AssertionError):
        UiUploadService("service")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        asyncio.run(UiUploadService(FakeService()).submit_image("a.jpg", "image/jpeg", "text"))  # type: ignore[arg-type]


def test_the_ui_session_stores_images_through_the_real_backend() -> None:
    backend: Backend = Backend(make_engine())
    service: UploadService = backend.upload_service  # The adapter fits the UI interface

    async def scenario() -> list[ItemStatus]:
        session: UploadSession = UploadSession(service)
        session.add("a.jpg", "image/jpeg", picture(1))
        session.add("again.jpg", "image/jpeg", picture(1))  # Same picture: a duplicate
        session.add("broken.jpg", "image/jpeg", b"\xff\xd8 not really a jpeg")
        await session.wait_idle()
        return [item.status for item in session.items]

    assert asyncio.run(scenario()) == [
        ItemStatus.STORED,
        ItemStatus.DUPLICATE,
        ItemStatus.REJECTED,
    ]


def test_the_backend_makes_a_working_router_and_takes_the_limit_from_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAARTHI_API_TOKEN", raising=False)
    backend: Backend = Backend(make_engine())
    assert backend.max_upload_bytes == storage.current_config().max_bytes
    app: FastAPI = FastAPI()
    app.include_router(backend.create_router())
    response = TestClient(app).post(UPLOAD_ROUTE, content=picture(2))
    assert response.status_code == 201
    assert db.get_image(response.json()["image_id"]) is not None


def test_the_token_from_the_environment_protects_the_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAARTHI_API_TOKEN", "secret-token-for-the-test")
    app: FastAPI = FastAPI()
    app.include_router(Backend(make_engine()).create_router())
    client: TestClient = TestClient(app)
    assert client.post(UPLOAD_ROUTE, content=picture(3)).status_code == 401
    ok = client.post(
        UPLOAD_ROUTE,
        content=picture(3),
        headers={"Authorization": "Bearer secret-token-for-the-test"},
    )
    assert ok.status_code == 201


def test_the_backend_needs_started_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "_config", None)
    with pytest.raises(storage.StorageConfigError):
        Backend(make_engine())


def test_the_backend_refuses_a_model_with_the_wrong_vector_size() -> None:
    engine = make_engine(embedder=FakeEmbedder(dimension=128))
    with pytest.raises(BackendConfigError, match="128"):
        Backend(engine)


def test_the_backend_refuses_a_scene_model_with_the_wrong_open_vector_size() -> None:
    scene = FakeSceneClassifier()
    scene.dimension = 100
    engine = make_engine(scene_classifier=scene)
    with pytest.raises(BackendConfigError, match="open_dimension"):
        Backend(engine)


def test_the_open_vector_size_is_not_checked_when_the_broad_detector_is_off() -> None:
    scene = FakeSceneClassifier()
    scene.dimension = 100
    engine = make_engine(scene_classifier=scene)
    engine._open_detector = None
    Backend(engine)  # No error: no open vectors are made


def test_the_backend_starts_and_stops_the_worker_and_an_upload_wakes_it() -> None:
    backend: Backend = Backend(make_engine())
    assert not backend.worker.is_running
    backend.start()
    try:
        assert backend.worker.is_running
        app: FastAPI = FastAPI()
        app.include_router(backend.create_router())
        image_id: int = (
            TestClient(app).post(UPLOAD_ROUTE, content=picture(50)).json()["image_id"]
        )
        deadline: float = time.monotonic() + 15
        while time.monotonic() < deadline:
            image = db.get_image(image_id)
            if image is not None and image.status == "done":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("The worker did not finish the image")
    finally:
        backend.stop()
    assert not backend.worker.is_running
    assert backend.server.port > 0
