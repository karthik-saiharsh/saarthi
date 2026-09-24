"""Tests of the gallery routes. They use the actual database and storage, and a fake ai engine."""

import json
import uuid
from collections.abc import Iterator
from io import BytesIO

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiError
from backend import http_common
from backend.enrollment import EnrollmentService
from backend.errors import (
    BackendError,
    InvalidInputError,
    LimitReachedError,
    NotFoundError,
)
from backend.http_objects import (
    MAX_JSON_BYTES,
    MESSAGE_BODY,
    MESSAGE_FAILED,
    MESSAGE_FILE_MISSING,
    MESSAGE_NAME_EXISTS,
    MESSAGE_NOT_FOUND,
    MESSAGE_PHOTO_EXISTS,
    MESSAGE_QUERY,
    MESSAGE_WRONG_STATE,
    OBJECTS_ROUTE,
    create_objects_router,
)
from backend.settings import load_config
from db.schema import PhotoStatus, ReferencePhoto
from tests.ai_support import make_engine

PREFIX: str = "__test__ "
MODEL: str = "fake-embedder@00000000"
TOKEN: str = "correct-horse-battery-staple"
DIMENSION: int = 384


def picture(seed: int) -> bytes:
    rng: np.random.Generator = np.random.default_rng(seed)
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
        buffer, "JPEG"
    )
    return buffer.getvalue()


def near(seed: int, noise: int) -> np.ndarray:
    base: np.ndarray = np.random.default_rng(seed).normal(size=DIMENSION)
    noisy: np.ndarray = base + np.random.default_rng(noise).normal(
        scale=0.3, size=DIMENSION
    )
    return (noisy / np.linalg.norm(noisy)).astype(np.float32)


def client_for(
    service: EnrollmentService, max_bytes: int = 1_000_000, token: str | None = None
) -> TestClient:
    app: FastAPI = FastAPI()
    app.include_router(create_objects_router(service, max_bytes, token))
    return TestClient(app)


def body(name: str | None = None, **changes: object) -> bytes:
    data: dict[str, object] = {
        "name": name or f"{PREFIX}{uuid.uuid4().hex[:8]}",
        "category": "a bottle",
    }
    data.update(changes)
    return json.dumps(data).encode()


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    yield
    for obj in db.gallery.list_objects():
        if obj.name.startswith(PREFIX):
            storage.delete_images(db.gallery.delete_object(obj.id))
    database.close()


@pytest.fixture()
def client() -> TestClient:
    return client_for(EnrollmentService(load_config(), make_engine()))


def make_object(client: TestClient, **changes: object) -> int:
    response = client.post(OBJECTS_ROUTE, content=body(**changes))
    assert response.status_code == 201, response.text
    return response.json()["id"]


def located_photo(client: TestClient, object_id: int, seed: int) -> int:
    """Add a photo through the route, then give it a view as the worker would."""
    response = client.post(f"{OBJECTS_ROUTE}/{object_id}/photos", content=picture(seed))
    assert response.status_code == 201, response.text
    photo_id: int = response.json()["id"]
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo_id
    ).execute()
    db.gallery.store_photo_views(
        photo_id, (5, 5, 40, 40), 0.5, [("original", near(1, seed))], MODEL
    )
    return photo_id


# ---- Objects ----


def test_an_object_is_created_listed_shown_and_deleted(client: TestClient) -> None:
    created = client.post(
        OBJECTS_ROUTE, content=body(synonyms=["a flask"], tracked=False)
    )
    assert created.status_code == 201
    data: dict = created.json()
    assert (
        data["status"] == "draft"
        and data["synonyms"] == ["a flask"]
        and data["tracked"] is False
    )
    assert data["note"] == "Needs 3 more confirmed photo(s)."
    object_id: int = data["id"]

    assert object_id in [o["id"] for o in client.get(OBJECTS_ROUTE).json()]
    detail = client.get(f"{OBJECTS_ROUTE}/{object_id}").json()
    assert detail["summary"]["id"] == object_id and detail["photos"] == []

    assert client.delete(f"{OBJECTS_ROUTE}/{object_id}").status_code == 204
    assert client.get(f"{OBJECTS_ROUTE}/{object_id}").status_code == 404
    assert client.delete(f"{OBJECTS_ROUTE}/{object_id}").status_code == 404


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"[1, 2]",
        b"{}",
        json.dumps({"name": 5, "category": "a bottle"}).encode(),
        json.dumps({"name": "x", "category": ["a bottle"]}).encode(),
        json.dumps({"name": "x", "category": "a bottle", "tracked": "yes"}).encode(),
        json.dumps(
            {"name": "x", "category": "a bottle", "synonyms": "a flask"}
        ).encode(),
        json.dumps({"name": "x", "category": "a bottle", "synonyms": [1]}).encode(),
        json.dumps(
            {"name": "x", "category": "a bottle", "synonyms": ["a"] * 51}
        ).encode(),
        b"\xff\xfe",
    ],
)
def test_a_body_that_is_not_valid_gets_400(client: TestClient, content: bytes) -> None:
    response = client.post(OBJECTS_ROUTE, content=content)
    assert response.status_code == 400 and response.json() == {"error": MESSAGE_BODY}


def test_a_bad_phrase_or_name_gets_400_with_the_rule(client: TestClient) -> None:
    response = client.post(OBJECTS_ROUTE, content=body(category="a bottle. a cup"))
    assert (
        response.status_code == 400 and "lowercase letters" in response.json()["error"]
    )
    response = client.post(OBJECTS_ROUTE, content=body(name="   "))
    assert response.status_code == 400 and "printable" in response.json()["error"]


def test_a_duplicate_name_gets_409(client: TestClient) -> None:
    name: str = f"{PREFIX}{uuid.uuid4().hex[:8]}"
    make_object(client, name=name)
    response = client.post(OBJECTS_ROUTE, content=body(name=name))
    assert response.status_code == 409 and response.json() == {
        "error": MESSAGE_NAME_EXISTS
    }


def test_a_json_body_that_is_too_large_gets_413(client: TestClient) -> None:
    response = client.post(OBJECTS_ROUTE, content=b" " * (MAX_JSON_BYTES + 1))
    assert response.status_code == 413


@pytest.mark.parametrize("text", ["abc", "0", "-1", "1.5", "99999999999", "%20", "٣"])
def test_an_id_that_is_not_a_positive_number_gets_404(
    client: TestClient, text: str
) -> None:
    for call in (client.get, client.delete):
        assert call(f"{OBJECTS_ROUTE}/{text}").status_code == 404
    assert (
        client.post(f"{OBJECTS_ROUTE}/{text}/photos", content=picture(1)).status_code
        == 404
    )
    assert client.post(f"{OBJECTS_ROUTE}/1/photos/{text}/confirm").status_code == 404


# ---- Photos ----


def test_a_photo_is_added_as_pending(client: TestClient) -> None:
    object_id: int = make_object(client)
    response = client.post(f"{OBJECTS_ROUTE}/{object_id}/photos", content=picture(10))
    assert response.status_code == 201
    assert response.json()["status"] == "pending" and response.json()["box"] is None
    assert (
        client.get(f"{OBJECTS_ROUTE}/{object_id}").json()["summary"]["photo_counts"][
            "pending"
        ]
        == 1
    )


def test_bad_photos_get_the_right_answers(client: TestClient) -> None:
    object_id: int = make_object(client)
    url: str = f"{OBJECTS_ROUTE}/{object_id}/photos"
    files_before: set[str] = set(storage.list_stored_paths())
    assert client.post(url, content=b"").status_code == 400
    assert client.post(url, content=b"not an image").status_code == 415
    assert (
        client.post(f"{OBJECTS_ROUTE}/{2**30}/photos", content=picture(11)).status_code
        == 404
    )
    assert set(storage.list_stored_paths()) == files_before  # Nothing was stored
    assert client.post(url, content=picture(12)).status_code == 201
    again = client.post(url, content=picture(12))
    assert again.status_code == 409 and again.json() == {"error": MESSAGE_PHOTO_EXISTS}


def test_a_photo_over_the_size_limit_gets_413() -> None:
    small: TestClient = client_for(
        EnrollmentService(load_config(), make_engine()), max_bytes=100
    )
    object_id: int = make_object(small)
    response = small.post(f"{OBJECTS_ROUTE}/{object_id}/photos", content=picture(13))
    assert response.status_code == 413


def test_the_crop_is_a_private_jpeg(client: TestClient) -> None:
    object_id: int = make_object(client)
    photo_id: int = located_photo(client, object_id, 20)
    response = client.get(f"{OBJECTS_ROUTE}/{object_id}/photos/{photo_id}/crop")
    assert (
        response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    )
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert PilImage.open(BytesIO(response.content)).size == (
        35,
        35,
    )  # The box (5, 5, 40, 40)
    small = client.get(
        f"{OBJECTS_ROUTE}/{object_id}/photos/{photo_id}/crop?max_side=32"
    )
    assert PilImage.open(BytesIO(small.content)).size == (32, 32)


def test_there_is_no_crop_before_the_object_is_located_or_for_a_bad_request(
    client: TestClient,
) -> None:
    object_id, other_id = make_object(client), make_object(client)
    pending: int = client.post(
        f"{OBJECTS_ROUTE}/{object_id}/photos", content=picture(21)
    ).json()["id"]
    assert (
        client.get(f"{OBJECTS_ROUTE}/{object_id}/photos/{pending}/crop").status_code
        == 404
    )
    located: int = located_photo(client, object_id, 22)
    assert (
        client.get(f"{OBJECTS_ROUTE}/{other_id}/photos/{located}/crop").status_code
        == 404
    )
    for bad in ("0", "31", "1025", "abc", "-5", ""):
        response = client.get(
            f"{OBJECTS_ROUTE}/{object_id}/photos/{located}/crop?max_side={bad}"
        )
        assert response.status_code == 400 and response.json() == {
            "error": MESSAGE_QUERY
        }


def test_three_confirmed_photos_make_an_object_ready(client: TestClient) -> None:
    object_id: int = make_object(client)
    for seed in (30, 31, 32):
        photo_id: int = located_photo(client, object_id, seed)
        response = client.post(f"{OBJECTS_ROUTE}/{object_id}/photos/{photo_id}/confirm")
        assert response.status_code == 200 and response.json()["status"] == "confirmed"
    summary: dict = client.get(f"{OBJECTS_ROUTE}/{object_id}").json()["summary"]
    assert (
        summary["status"] == "ready"
        and 0 < summary["tau"] <= summary["tau_strong"] <= 1.0
    )


def test_a_photo_in_the_wrong_state_gets_409(client: TestClient) -> None:
    object_id: int = make_object(client)
    pending: int = client.post(
        f"{OBJECTS_ROUTE}/{object_id}/photos", content=picture(40)
    ).json()["id"]
    for action in ("confirm", "retry"):
        response = client.post(f"{OBJECTS_ROUTE}/{object_id}/photos/{pending}/{action}")
        assert response.status_code == 409 and response.json() == {
            "error": MESSAGE_WRONG_STATE
        }


def test_a_failed_photo_can_be_tried_again(client: TestClient) -> None:
    object_id: int = make_object(client)
    photo_id: int = client.post(
        f"{OBJECTS_ROUTE}/{object_id}/photos", content=picture(41)
    ).json()["id"]
    ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
        ReferencePhoto.id == photo_id
    ).execute()
    db.gallery.mark_photo_failed(photo_id, "object not found")
    response = client.post(f"{OBJECTS_ROUTE}/{object_id}/photos/{photo_id}/retry")
    assert response.status_code == 200 and response.json()["status"] == "pending"


def test_a_photo_is_removed_with_its_file(client: TestClient) -> None:
    object_id: int = make_object(client)
    photo_id: int = located_photo(client, object_id, 50)
    path: str = db.gallery.get_photo(photo_id).file_path  # type: ignore[union-attr]
    assert (
        client.delete(f"{OBJECTS_ROUTE}/{object_id}/photos/{photo_id}").status_code
        == 204
    )
    assert path not in storage.list_stored_paths()
    assert (
        client.delete(f"{OBJECTS_ROUTE}/{object_id}/photos/{photo_id}").status_code
        == 404
    )


def test_a_photo_of_another_object_is_not_found(client: TestClient) -> None:
    first, second = make_object(client), make_object(client)
    photo_id: int = located_photo(client, first, 60)
    for call, suffix in (
        (client.post, "/confirm"),
        (client.post, "/retry"),
        (client.delete, ""),
    ):
        assert (
            call(f"{OBJECTS_ROUTE}/{second}/photos/{photo_id}{suffix}").status_code
            == 404
        )


# ---- Error mapping ----


class FailingService(EnrollmentService):
    """Every call raises the error that the test sets."""

    def __init__(self, error: Exception) -> None:
        super().__init__(load_config(), make_engine())
        self.error: Exception = error

    def list_objects(self):
        raise self.error


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (InvalidInputError("A fixed text."), 400, "A fixed text."),
        (NotFoundError("secret detail"), 404, MESSAGE_NOT_FOUND),
        (
            LimitReachedError("The gallery has 50 objects"),
            409,
            "The gallery has 50 objects",
        ),
        (db.DuplicateObjectError("x"), 409, MESSAGE_NAME_EXISTS),
        (db.DuplicateImageError("x"), 409, MESSAGE_PHOTO_EXISTS),
        (db.InvalidStateError("x"), 409, MESSAGE_WRONG_STATE),
        (storage.ImageTooLargeError("x"), 413, http_common.MESSAGE_TOO_LARGE),
        (storage.UnsupportedImageError("x"), 415, http_common.MESSAGE_UNSUPPORTED),
        (storage.StorageFullError("x"), 507, http_common.MESSAGE_DISK_FULL),
        (storage.StorageNotFoundError("/secret/path"), 404, MESSAGE_FILE_MISSING),
        (storage.StorageIOError("/secret/path failed"), 500, MESSAGE_FAILED),
        (db.DbError("SELECT secret FROM table"), 500, MESSAGE_FAILED),
        (BackendError("internal detail"), 500, MESSAGE_FAILED),
        (AiError("model path"), 500, MESSAGE_FAILED),
        (RuntimeError("internal detail"), 500, MESSAGE_FAILED),
    ],
)
def test_each_error_gets_its_code_and_a_fixed_message(
    error: Exception, code: int, message: str
) -> None:
    response = client_for(FailingService(error)).get(OBJECTS_ROUTE)
    assert response.status_code == code and response.json() == {"error": message}


# ---- Token ----


ROUTES: list[tuple[str, str]] = [
    ("POST", OBJECTS_ROUTE),
    ("GET", OBJECTS_ROUTE),
    ("GET", f"{OBJECTS_ROUTE}/1"),
    ("DELETE", f"{OBJECTS_ROUTE}/1"),
    ("POST", f"{OBJECTS_ROUTE}/1/photos"),
    ("GET", f"{OBJECTS_ROUTE}/1/photos/1/crop"),
    ("GET", f"{OBJECTS_ROUTE}/1/photos/1/crop?max_side=abc"),
    ("POST", f"{OBJECTS_ROUTE}/1/photos/1/confirm"),
    ("POST", f"{OBJECTS_ROUTE}/1/photos/1/retry"),
    ("DELETE", f"{OBJECTS_ROUTE}/1/photos/1"),
    ("GET", f"{OBJECTS_ROUTE}/abc"),
]


@pytest.mark.parametrize(("method", "url"), ROUTES)
def test_every_route_needs_the_token_when_one_is_set(method: str, url: str) -> None:
    secured: TestClient = client_for(
        EnrollmentService(load_config(), make_engine()), token=TOKEN
    )
    count: int = db.gallery.count_objects()
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}):
        response = secured.request(method, url, content=body(), headers=headers)
        assert (
            response.status_code == 401
            and response.headers["WWW-Authenticate"] == "Bearer"
        )
        assert response.json() == {"error": http_common.MESSAGE_TOKEN}
    assert db.gallery.count_objects() == count  # The refused request changed nothing


def test_the_right_token_is_accepted() -> None:
    secured: TestClient = client_for(
        EnrollmentService(load_config(), make_engine()), token=TOKEN
    )
    headers: dict[str, str] = {"Authorization": f"Bearer {TOKEN}"}
    created = secured.post(OBJECTS_ROUTE, content=body(), headers=headers)
    assert created.status_code == 201
    assert secured.get(OBJECTS_ROUTE, headers=headers).status_code == 200


def test_the_parameters_are_checked() -> None:
    service: EnrollmentService = EnrollmentService(load_config(), make_engine())
    with pytest.raises(AssertionError):
        create_objects_router("service", 100)  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        create_objects_router(service, 0)
    with pytest.raises(AssertionError):
        create_objects_router(service, 100, token=" ")
