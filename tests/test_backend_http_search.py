"""Tests of the search routes. They use the actual database and storage, and fake ai models.

Every object has a name that starts with `__test__ `. The frames are real files
in the storage, so the picture routes return real pixels. The cleanup removes the
objects first (their last-seen records keep frames), then the frames and the files.
"""

import dataclasses
import json
import logging
from collections.abc import Iterator
from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from backend.http_objects import MAX_JSON_BYTES, OBJECTS_ROUTE
from backend.http_search import (
    MESSAGE_VIEW,
    SEARCH_ROUTE,
    create_search_router,
)
from backend.query import QueryResolver
from backend.search import SearchService
from backend.settings import BackendConfig, load_config
from db.schema import EnrolledObject, Image
from tests.ai_support import make_engine
from tests.db_support import skip_if_actual_data_has_problems
from tests.search_support import (
    BOX,
    FRAME_SIZE,
    OTHER,
    PREFIX,
    make_object,
    store_frame,
)

TOKEN: str = "correct-horse-battery-staple"


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    if db.gallery.gallery_phrases():
        pytest.skip("The actual gallery has ready objects")
    before: set[str] = set(db.list_image_paths())
    yield
    # The objects go first. A last-seen record keeps its frame.
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))
    database.close()


def build_client(token: str | None = None) -> TestClient:
    base: BackendConfig = load_config()
    query = dataclasses.replace(
        base.query, matched_min_score=0.4, matched_min_gap=0.1, suggest_min_score=0.2
    )
    config: BackendConfig = dataclasses.replace(base, query=query)
    engine = make_engine()
    app: FastAPI = FastAPI()
    app.include_router(
        create_search_router(
            QueryResolver(config, engine), SearchService(config, engine), token
        )
    )
    return TestClient(app)


def jpeg_size(data: bytes) -> tuple[int, int]:
    return PilImage.open(BytesIO(data)).size


def post(client: TestClient, query: object, **headers: str):
    return client.post(
        SEARCH_ROUTE, content=json.dumps({"query": query}).encode(), headers=headers
    )


# ---- POST /api/search ----


def test_clear_words_give_the_object_and_its_answer() -> None:
    object_id: int = make_object(name="blue bottle", phrase="a bottle").id
    make_object(name="house keys", phrase="a set of keys", views=OTHER)
    frame: Image = store_frame(10)
    response = post(build_client(), "where is my blue bottle")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "matched"
    assert [c["object_id"] for c in data["candidates"]] == [object_id]
    answer = data["answer"]
    assert answer["found"] and answer["source"] == "archive"
    assert answer["image_id"] == frame.id and answer["box"] == list(BOX)
    assert answer["seen_at"].endswith("Z") and 590 <= answer["age_seconds"] <= 700
    assert answer["strong"] is True and answer["stale"] is False
    assert response.headers["cache-control"] == "no-store"


def test_a_matched_object_that_was_not_seen_gives_an_answer_with_found_false() -> None:
    make_object(name="blue bottle", phrase="a bottle")
    data = post(build_client(), "my blue bottle").json()
    assert data["status"] == "matched" and data["answer"]["found"] is False
    assert data["answer"]["seen_at"] is None and data["answer"]["complete"] is True


def test_words_that_fit_two_objects_give_a_choice_and_no_answer() -> None:
    make_object(name="red cup", phrase="a cup")
    make_object(name="blue cup", phrase="a cup")
    store_frame()
    data = post(build_client(), "my cup").json()
    assert data["status"] == "ambiguous" and data["answer"] is None
    assert {c["name"].removeprefix(PREFIX) for c in data["candidates"]} == {
        "red cup",
        "blue cup",
    }


def test_words_that_fit_nothing_give_none() -> None:
    make_object(name="blue bottle", phrase="a bottle")
    data = post(build_client(), "xylophone zebra").json()
    assert data == {"status": "none", "candidates": [], "answer": None}


@pytest.mark.parametrize(
    "content",
    [b"not json", b"[]", b'{"q": "x"}', b'{"query": 5}', b'{"query": null}', b""],
)
def test_a_body_that_is_not_valid_is_refused(content: bytes) -> None:
    response = build_client().post(SEARCH_ROUTE, content=content)
    assert response.status_code == 400


@pytest.mark.parametrize("query", ["", "   ", "x" * 301, "bell\x07"])
def test_words_that_cannot_be_used_are_refused_with_a_fixed_text(query: str) -> None:
    response = post(build_client(), query)
    assert response.status_code == 400
    assert response.json() == {
        "error": "Write what you are looking for, in a few words."
    }


def test_a_body_over_the_json_limit_is_refused() -> None:
    big = json.dumps({"query": "a" * (MAX_JSON_BYTES + 1)}).encode()
    assert build_client().post(SEARCH_ROUTE, content=big).status_code == 413


def test_the_words_are_not_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    make_object(name="blue bottle", phrase="a bottle")
    with caplog.at_level(logging.DEBUG):
        post(build_client(), "my private blue bottle")
    assert "private" not in caplog.text


# ---- GET /api/objects/{id}/last-seen ----


def test_last_seen_gives_the_answer_for_an_object() -> None:
    object_id: int = make_object().id
    store_frame(5, in_hand=True)
    response = build_client().get(f"{OBJECTS_ROUTE}/{object_id}/last-seen")
    assert response.status_code == 200
    answer = response.json()
    assert answer["found"] and answer["in_hand"] is True
    assert answer["object_id"] == object_id and answer["source"] == "archive"


def test_last_seen_of_an_object_never_seen_is_found_false() -> None:
    object_id: int = make_object().id
    response = build_client().get(f"{OBJECTS_ROUTE}/{object_id}/last-seen")
    assert response.status_code == 200 and response.json()["found"] is False


def test_last_seen_errors_have_fixed_texts() -> None:
    client: TestClient = build_client()
    draft: int = make_object(ready=False).id
    twin: int = make_object(tracked=False).id
    for object_id in (draft, twin):
        response = client.get(f"{OBJECTS_ROUTE}/{object_id}/last-seen")
        assert response.status_code == 409
        assert response.json() == {"error": "This object cannot be searched yet."}
    assert client.get(f"{OBJECTS_ROUTE}/2147483000/last-seen").status_code == 404
    for bad in ("abc", "0", "-1", "1.5", "99999999999"):
        assert client.get(f"{OBJECTS_ROUTE}/{bad}/last-seen").status_code == 404


# ---- GET /api/objects/{id}/last-seen/picture ----


def test_the_crop_shows_the_object_with_space_around_it() -> None:
    object_id: int = make_object().id
    store_frame()
    response = build_client().get(f"{OBJECTS_ROUTE}/{object_id}/last-seen/picture")
    assert (
        response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    )
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    # The box is 40 x 30. A padding of 0.25 adds 10 and 8 pixels on each side.
    assert jpeg_size(response.content) == (60, 46)


def test_the_frame_view_shows_the_whole_frame() -> None:
    object_id: int = make_object().id
    store_frame()
    url: str = f"{OBJECTS_ROUTE}/{object_id}/last-seen/picture?view=frame"
    assert jpeg_size(build_client().get(url).content) == FRAME_SIZE


def test_max_side_shrinks_the_picture() -> None:
    object_id: int = make_object().id
    store_frame()
    url: str = f"{OBJECTS_ROUTE}/{object_id}/last-seen/picture?view=frame&max_side=40"
    assert max(jpeg_size(build_client().get(url).content)) == 40


def test_a_picture_of_an_object_never_seen_is_not_found() -> None:
    object_id: int = make_object().id
    response = build_client().get(f"{OBJECTS_ROUTE}/{object_id}/last-seen/picture")
    assert response.status_code == 404 and response.json() == {"error": "Not found."}


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("view=full", MESSAGE_VIEW),
        ("view=", MESSAGE_VIEW),
        ("max_side=31", None),
        ("max_side=1025", None),
        ("max_side=abc", None),
    ],
)
def test_a_wrong_option_is_refused(query: str, message: str | None) -> None:
    object_id: int = make_object().id
    store_frame()
    response = build_client().get(
        f"{OBJECTS_ROUTE}/{object_id}/last-seen/picture?{query}"
    )
    assert response.status_code == 400
    if message is not None:
        assert response.json() == {"error": message}


def test_a_picture_whose_frame_file_is_gone_gives_a_fixed_text() -> None:
    object_id: int = make_object().id
    frame: Image = store_frame()
    storage.delete_images([frame.file_path])
    response = build_client().get(f"{OBJECTS_ROUTE}/{object_id}/last-seen/picture")
    assert response.status_code == 404
    assert response.json() == {"error": "The file of the photo is missing."}


# ---- The token ----


def test_every_route_needs_the_token_when_one_is_set() -> None:
    object_id: int = make_object().id
    store_frame()
    client: TestClient = build_client(TOKEN)
    base: str = f"{OBJECTS_ROUTE}/{object_id}/last-seen"
    for response in (
        client.post(SEARCH_ROUTE, content=b'{"query": "bottle"}'),
        client.get(base),
        client.get(base + "/picture"),
        client.get(
            OBJECTS_ROUTE + "/abc/last-seen"
        ),  # A bad id does not skip the check
        client.get(base + "/picture?view=bad"),  # A bad option does not skip it either
    ):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
    good = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get(base, headers=good).status_code == 200
    assert client.get(base + "/picture", headers=good).status_code == 200
    assert post(client, "bottle", **good).status_code == 200


def test_an_unexpected_error_gives_a_plain_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_id: int = make_object().id

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret path /home/x")

    monkeypatch.setattr(SearchService, "find", broken)
    response = build_client().get(f"{OBJECTS_ROUTE}/{object_id}/last-seen")
    assert response.status_code == 500
    assert "secret" not in response.text
