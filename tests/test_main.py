"""Tests of the life of the application (`main.py`).

They use the actual storage, database and index, and fake ai models. Each test
removes what it made.
"""

import time
import types
from collections.abc import Callable, Iterator
from io import BytesIO

import faiss
import numpy as np
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient
from PIL import Image as PilImage

import db.api as db
import main
import storage.api as storage
from ai.types import Box, RawDetection
from backend.errors import BackendConfigError
from backend.http import UPLOAD_ROUTE
from backend.instance_lock import InstanceRunningError
from db.schema import Image
from tests.ai_support import FakeDetector, FakeEmbedder, make_engine
from tests.db_support import skip_if_actual_data_has_problems


def picture(seed: int) -> bytes:
    rng: np.random.Generator = np.random.default_rng(seed)
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
        buffer, "JPEG"
    )
    return buffer.getvalue()


class RecordingApp:
    """A stand-in for the NiceGUI app. It records what `attach` does."""

    def __init__(self) -> None:
        self.routers: list[APIRouter] = []
        self.startup: list[Callable[[], None]] = []
        self.shutdown: list[Callable[[], None]] = []

    def include_router(self, router: APIRouter) -> None:
        self.routers.append(router)

    def on_startup(self, handler: Callable[[], None]) -> None:
        self.startup.append(handler)

    def on_shutdown(self, handler: Callable[[], None]) -> None:
        self.shutdown.append(handler)


@pytest.fixture(autouse=True)
def release_locks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Release the instance lock of every `Components` that a test starts."""
    started: list[main.Components] = []
    real_start = main.start_components

    def start(engine: object = None) -> main.Components:
        components: main.Components = real_start(engine)  # type: ignore[arg-type]
        started.append(components)
        return components

    monkeypatch.setattr(main, "start_components", start)
    yield
    for components in started:
        components.lock.release()


@pytest.fixture(autouse=True)
def clean_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    if Image.select().where(Image.status == "pending").exists():
        pytest.skip(
            "Actual database has pending images. The worker would claim one of them"
        )
    before: set[str] = set(db.list_image_paths())
    yield
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))
    db.checkpoint_vectors()  # Keep the actual index file free of test vectors
    database.close()


def test_the_components_start_in_order_and_the_recovery_report_is_kept() -> None:
    components: main.Components = main.start_components(engine=make_engine())
    assert components.recovery == db.StartupReport([], [], 0, 0)  # Nothing to repair
    assert (
        not components.backend.worker.is_running
    )  # The threads wait for the web server


def test_a_model_with_the_wrong_vector_size_stops_the_start() -> None:
    engine = make_engine(embedder=FakeEmbedder(dimension=128))
    with pytest.raises(BackendConfigError):
        main.start_components(engine=engine)


def test_attach_adds_the_route_and_ties_the_threads_to_the_server() -> None:
    components: main.Components = main.start_components(engine=make_engine())
    web_app: RecordingApp = RecordingApp()
    main.attach(components, web_app)  # type: ignore[arg-type]
    assert len(web_app.routers) == 1
    api: FastAPI = FastAPI()
    api.include_router(web_app.routers[0])
    paths: set[str] = set(api.openapi()["paths"])
    assert UPLOAD_ROUTE in paths and "/api/objects" in paths
    assert "/api/objects/{object_id}/photos" in paths

    web_app.startup[0]()
    assert (
        components.backend.worker.is_running
        and components.backend.maintenance.is_running
    )
    web_app.shutdown[0]()
    assert (
        not components.backend.worker.is_running
        and not components.backend.maintenance.is_running
    )


def test_an_upload_is_analysed_and_the_vectors_are_saved_at_shutdown() -> None:
    detector: FakeDetector = FakeDetector()
    detector.detections = [RawDetection(Box(5, 5, 40, 40), 0.6, "bottle", "full")]
    components: main.Components = main.start_components(engine=make_engine(detector))
    web_app: RecordingApp = RecordingApp()
    main.attach(components, web_app)  # type: ignore[arg-type]
    api: FastAPI = FastAPI()
    api.include_router(web_app.routers[0])

    web_app.startup[0]()
    try:
        response = TestClient(api).post(UPLOAD_ROUTE, content=picture(1))
        assert response.status_code == 201
        image_id: int = response.json()["image_id"]
        deadline: float = time.monotonic() + 15
        while time.monotonic() < deadline:
            image = db.get_image(image_id)
            if image is not None and image.status == "done":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("The worker did not finish the image")
    finally:
        web_app.shutdown[0]()

    region_ids: list[int] = [r.id for r in db.get_regions(image_id)]
    assert len(region_ids) == 1
    saved = faiss.read_index(
        str(db.current_config().faiss_path)
    )  # The shutdown saved the vectors
    assert set(faiss.vector_to_array(saved.id_map).tolist()) >= set(region_ids)


def test_a_start_error_gives_exit_code_1_and_no_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(engine: object = None) -> main.Components:
        raise storage.StorageConfigError("config is wrong")

    def no_server(*args: object, **kwargs: object) -> None:
        raise AssertionError("The server must not start")

    monkeypatch.setattr(main, "start_components", fail)
    monkeypatch.setattr(main.ui, "run", no_server)
    assert main.main() == 1


def test_the_pages_use_the_real_upload_service_and_the_home_page_leads_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components: main.Components = main.start_components(engine=make_engine())
    registered: list[object] = []
    pages: dict[str, Callable[[], object]] = {}

    def page(path: str) -> Callable[[Callable[[], object]], Callable[[], object]]:
        def register(function: Callable[[], object]) -> Callable[[], object]:
            pages[path] = function
            return function

        return register

    monkeypatch.setattr(
        main, "register_pages", lambda *services: registered.extend(services)
    )
    monkeypatch.setattr(main, "ui", types.SimpleNamespace(page=page))
    main.create_pages(components)
    assert [type(s).__name__ for s in registered] == [
        "UiUploadService",
        "UiEnrollmentService",
        "UiFindService",
        "UiVoiceService",
    ]
    answer = pages["/"]()
    assert (
        isinstance(answer, RedirectResponse) and answer.headers["location"] == "/upload"
    )


def test_a_second_copy_is_stopped_before_it_changes_anything() -> None:
    first: main.Components = main.start_components(engine=make_engine())
    with pytest.raises(InstanceRunningError):
        main.start_components(
            engine=make_engine()
        )  # It stops before the recovery and before the models
    first.lock.release()
    second: main.Components = main.start_components(
        engine=make_engine()
    )  # Free after the release
    assert second.lock.is_held


def test_the_shutdown_releases_the_lock() -> None:
    components: main.Components = main.start_components(engine=make_engine())
    assert components.lock.is_held
    main.stop_components(components)
    assert not components.lock.is_held


def test_a_failed_start_releases_the_lock() -> None:
    engine = make_engine(
        embedder=FakeEmbedder(dimension=128)
    )  # The backend refuses this model
    with pytest.raises(BackendConfigError):
        main.start_components(engine=engine)
    assert main.start_components(
        engine=make_engine()
    ).lock.is_held  # The lock was free again


def test_the_exit_code_is_1_when_another_copy_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(engine: object = None) -> main.Components:
        raise InstanceRunningError("Another copy of Saarthi is running")

    monkeypatch.setattr(main, "start_components", fail)
    monkeypatch.setattr(
        main.ui,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no server")),
    )
    assert main.main() == 1
