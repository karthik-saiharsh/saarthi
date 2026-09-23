"""Tests of the upload route.

The error answers use a fake service. The real answers use the actual storage
and database, and clean up what they make.
"""

import dataclasses
from collections.abc import Iterator
from io import BytesIO

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from backend import http
from backend.errors import BackendConfigError, QueueFullError
from backend.http import UPLOAD_ROUTE, create_router, read_token
from backend.ingest import IngestResult, IngestService, IngestStatus
from backend.settings import BackendConfig, load_config
from db.schema import Image

TOKEN: str = "correct-horse-battery-staple"


def picture(seed: int = 0) -> bytes:
    rng: np.random.Generator = np.random.default_rng(seed)
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
        buffer, "JPEG"
    )
    return buffer.getvalue()


class FakeService(IngestService):
    """Records the calls. It returns a result, or raises the error that the test sets."""

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__(load_config())
        self.error: Exception | None = error
        self.bodies: list[bytes] = []

    def submit(self, data: bytes) -> IngestResult:
        self.bodies.append(data)
        if self.error is not None:
            raise self.error
        return IngestResult(IngestStatus.STORED, 7)


def client_for(
    service: IngestService, max_bytes: int = 1_000_000, token: str | None = None
) -> TestClient:
    app: FastAPI = FastAPI()
    app.include_router(create_router(service, max_bytes, token))
    return TestClient(app)


# ---- Answers of the real service ----


@pytest.fixture()
def real_client() -> Iterator[TestClient]:
    storage.init_storage()
    database, _ = db.init_db()
    before: set[str] = set(db.list_image_paths())
    yield client_for(IngestService(load_config()))
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))
    database.close()


def test_a_new_image_gets_201_and_a_duplicate_gets_200(real_client: TestClient) -> None:
    first = real_client.post(UPLOAD_ROUTE, content=picture(1))
    assert first.status_code == 201 and first.json()["status"] == "stored"
    row: Image | None = db.get_image(first.json()["image_id"])
    assert row is not None and row.file_path in storage.list_stored_paths()
    second = real_client.post(UPLOAD_ROUTE, content=picture(1))
    assert second.status_code == 200
    assert second.json() == {
        "status": "duplicate",
        "image_id": first.json()["image_id"],
    }


@pytest.mark.parametrize(
    "content_type", ["image/jpeg", "text/plain", "application/octet-stream"]
)
def test_the_content_type_is_not_used(
    real_client: TestClient, content_type: str
) -> None:
    response = real_client.post(
        UPLOAD_ROUTE,
        content=picture(2 + len(content_type)),
        headers={"Content-Type": content_type},
    )
    assert response.status_code == 201


def test_data_that_is_not_an_image_gets_415(real_client: TestClient) -> None:
    response = real_client.post(UPLOAD_ROUTE, content=b"definitely not an image")
    assert response.status_code == 415 and response.json() == {
        "error": http.MESSAGE_UNSUPPORTED
    }


def test_an_empty_body_gets_400(real_client: TestClient) -> None:
    response = real_client.post(UPLOAD_ROUTE, content=b"")
    assert response.status_code == 400 and response.json() == {
        "error": http.MESSAGE_EMPTY
    }


def test_other_methods_are_not_allowed(real_client: TestClient) -> None:
    assert real_client.get(UPLOAD_ROUTE).status_code == 405
    assert real_client.put(UPLOAD_ROUTE, content=picture(5)).status_code == 405


# ---- Size limit ----


def test_a_declared_size_over_the_limit_gets_413_and_is_not_read() -> None:
    service: FakeService = FakeService()
    response = client_for(service, max_bytes=100).post(UPLOAD_ROUTE, content=b"x" * 101)
    assert response.status_code == 413 and service.bodies == []


def test_a_stream_over_the_limit_gets_413_without_a_content_length() -> None:
    service: FakeService = FakeService()
    chunks: Iterator[bytes] = (
        b"x" * 40 for _ in range(10)
    )  # No Content-Length: 400 bytes in chunks
    response = client_for(service, max_bytes=100).post(UPLOAD_ROUTE, content=chunks)
    assert response.status_code == 413 and service.bodies == []


def test_a_body_at_the_limit_is_read() -> None:
    service: FakeService = FakeService()
    response = client_for(service, max_bytes=100).post(UPLOAD_ROUTE, content=b"x" * 100)
    assert response.status_code == 201 and service.bodies == [b"x" * 100]


def test_a_bad_content_length_gets_400() -> None:
    service: FakeService = FakeService()
    response = client_for(service).post(
        UPLOAD_ROUTE, content=b"x", headers={"Content-Length": "abc"}
    )
    assert response.status_code == 400 and service.bodies == []


# ---- Error answers ----


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (QueueFullError("500 wait"), 503, http.MESSAGE_BUSY),
        (storage.ImageTooLargeError("too many pixels"), 413, http.MESSAGE_TOO_LARGE),
        (storage.UnsupportedImageError("gif"), 415, http.MESSAGE_UNSUPPORTED),
        (storage.StorageFullError("full"), 507, http.MESSAGE_DISK_FULL),
        (storage.StorageIOError("/secret/path failed"), 500, http.MESSAGE_FAILED),
        (db.DbError("SELECT secret FROM table"), 500, http.MESSAGE_FAILED),
        (RuntimeError("internal detail"), 500, http.MESSAGE_FAILED),
    ],
)
def test_each_error_gets_its_code_and_a_message_without_details(
    error: Exception, code: int, message: str
) -> None:
    response = client_for(FakeService(error)).post(UPLOAD_ROUTE, content=b"data")
    assert response.status_code == code and response.json() == {"error": message}


def test_a_busy_server_says_when_to_try_again() -> None:
    response = client_for(FakeService(QueueFullError("full"))).post(
        UPLOAD_ROUTE, content=b"data"
    )
    assert response.headers["Retry-After"] == str(http.RETRY_AFTER_SECONDS)


# ---- Token ----


def test_without_a_token_setting_no_header_is_needed() -> None:
    assert (
        client_for(FakeService()).post(UPLOAD_ROUTE, content=b"data").status_code == 201
    )


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": TOKEN},  # No scheme
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": "Bearer "},
        {"Authorization": f"Bearer {TOKEN}x"},
    ],
)
def test_a_missing_or_wrong_token_gets_401_and_the_body_is_not_read(
    headers: dict[str, str],
) -> None:
    service: FakeService = FakeService()
    response = client_for(service, token=TOKEN).post(
        UPLOAD_ROUTE, content=b"data", headers=headers
    )
    assert (
        response.status_code == 401 and response.headers["WWW-Authenticate"] == "Bearer"
    )
    assert response.json() == {"error": http.MESSAGE_TOKEN} and service.bodies == []


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_the_right_token_is_accepted(scheme: str) -> None:
    response = client_for(FakeService(), token=TOKEN).post(
        UPLOAD_ROUTE, content=b"data", headers={"Authorization": f"{scheme} {TOKEN}"}
    )
    assert response.status_code == 201


def test_the_token_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config: BackendConfig = dataclasses.replace(
        load_config(), max_pending=5, token_env_var="SAARTHI_TEST_TOKEN"
    )
    monkeypatch.delenv("SAARTHI_TEST_TOKEN", raising=False)
    assert read_token(config) is None
    monkeypatch.setenv("SAARTHI_TEST_TOKEN", TOKEN)
    assert read_token(config) == TOKEN
    for empty in ("", "   "):
        monkeypatch.setenv("SAARTHI_TEST_TOKEN", empty)
        with pytest.raises(BackendConfigError):
            read_token(config)


def test_parameters_are_checked() -> None:
    with pytest.raises(AssertionError):
        create_router("service", 100)  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        create_router(FakeService(), 0)
    with pytest.raises(AssertionError):
        create_router(FakeService(), 100, token=" ")
