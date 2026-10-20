"""Tests of the question of the wearable: the talk service and its routes.

They use the actual database and storage, and fake ai models. Every object has a name that starts with
`__test__ `. The objects go first (a last-seen record keeps its frame), then the frames and their files.
"""

import dataclasses
import logging
import time
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import db.api as db
import storage.api as storage
from backend.describe import DescribeService
from backend.errors import QueueFullError
from backend.http_talk import (
    MESSAGE_NO_TOKEN,
    MESSAGE_OFF,
    TALK_ROUTE,
    create_talk_router,
)
from backend.query import QueryResolver
from backend.search import SearchService
from backend.settings import BackendConfig, load_config
from backend.talk import (
    FAILED_TEXT,
    NOT_HEARD_TEXT,
    UNKNOWN_TEXT,
    TalkJob,
    TalkOutcome,
    TalkService,
    TalkStatus,
    choose_text,
)
from backend.voice import VoiceService
from db.schema import EnrolledObject, Image
from tests.ai_support import FakeSpeechRecognizer, make_engine
from tests.db_support import skip_if_actual_data_has_problems
from tests.search_support import BOTTLE, PREFIX, WEAK, make_object, store_frame
from tests.test_backend_voice import wav_bytes

TOKEN: str = "correct-horse-battery-staple"
AUTH: dict[str, str] = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    if db.gallery.gallery_phrases():
        pytest.skip("The actual gallery has ready objects")
    before: set[str] = set(db.list_image_paths())
    yield
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))
    database.close()


def make_service(
    text: str = "where is my blue bottle",
    max_jobs: int = 4,
    keep_seconds: int = 300,
    speech: bool = True,
) -> tuple[TalkService, FakeSpeechRecognizer]:
    base: BackendConfig = load_config()
    config: BackendConfig = dataclasses.replace(
        base,
        query=dataclasses.replace(
            base.query,
            matched_min_score=0.4,
            matched_min_gap=0.1,
            suggest_min_score=0.2,
        ),
        talk=dataclasses.replace(
            base.talk, max_jobs=max_jobs, keep_seconds=keep_seconds
        ),
    )
    fake = FakeSpeechRecognizer(text)
    engine = make_engine(speech_recognizer=fake)
    if not speech:
        engine._speech = None  # The switch "speech off" of the engine
    service = TalkService(
        config,
        VoiceService(config, engine),
        QueryResolver(config, engine),
        SearchService(config, engine),
        DescribeService(config, engine),
    )
    return service, fake


def wait_done(service: TalkService, job_id: str) -> TalkJob:
    deadline: float = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = service.get(job_id)
        assert job is not None
        if job.status == TalkStatus.DONE:
            return job
        time.sleep(0.02)
    raise AssertionError("the job did not finish")


# ---- The answer ----


def test_a_sure_sighting_is_said_without_a_picture() -> None:
    make_object(name="blue bottle", phrase="a bottle")
    store_frame(12, BOTTLE)
    outcome, text = make_service()[0].answer("where is my blue bottle")
    assert outcome == TalkOutcome.FOUND
    assert "blue bottle" in text and "12 minutes ago" in text
    assert "picture" not in text


def test_a_weak_sighting_says_not_sure_and_never_asks_to_look() -> None:
    make_object(name="blue bottle")
    store_frame(5, WEAK)
    outcome, text = make_service()[0].answer("where is my blue bottle")
    assert outcome == TalkOutcome.MAYBE
    assert "not sure" in text and "picture" not in text


def test_a_known_object_that_was_not_seen_says_so() -> None:
    make_object(name="blue bottle")
    outcome, text = make_service()[0].answer("where is my blue bottle")
    assert outcome == TalkOutcome.NOT_SEEN and "not seen" in text


def test_words_that_fit_two_objects_name_them_and_ask_again() -> None:
    make_object(name="red cup", phrase="a cup")
    make_object(name="blue cup", phrase="a cup")
    outcome, text = make_service()[0].answer("my cup")
    assert outcome == TalkOutcome.CHOOSE
    assert f"your {PREFIX}red cup" in text and f"your {PREFIX}blue cup" in text
    assert text.endswith("Please press the button and say the name.")


def test_choose_text_lists_three_names() -> None:
    assert choose_text(["watch", "book", "cup"]) == (
        "Do you mean your watch, your book, or your cup? "
        "Please press the button and say the name."
    )


def test_no_object_and_no_open_search_is_unknown() -> None:
    assert make_service()[0].answer("where is my umbrella") == (
        TalkOutcome.UNKNOWN,
        UNKNOWN_TEXT,
    )


def test_empty_words_are_not_heard() -> None:
    assert make_service()[0].answer("   ") == (TalkOutcome.NOT_HEARD, NOT_HEARD_TEXT)


# ---- The jobs ----


def test_a_recording_becomes_an_answer_and_the_words_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    make_object(name="blue bottle")
    store_frame(12, BOTTLE)
    service, fake = make_service("where is my blue bottle")
    with caplog.at_level(logging.DEBUG):
        job = service.submit(wav_bytes(1.0))
        assert job.status == TalkStatus.PENDING and len(job.id) == 22
        done = wait_done(service, job.id)
    service.close()
    assert done.outcome == TalkOutcome.FOUND and "blue bottle" in (done.text or "")
    assert fake.calls  # The recording went to the speech model
    assert "blue bottle" not in caplog.text and done.id not in caplog.text


def test_a_recording_that_is_not_a_wav_file_is_not_heard_with_a_detail() -> None:
    service, _ = make_service()
    done = wait_done(service, service.submit(b"not a wav file").id)
    service.close()
    assert done.outcome == TalkOutcome.NOT_HEARD and done.text == NOT_HEARD_TEXT
    assert done.detail is not None


def test_silence_is_not_heard() -> None:
    service, fake = make_service()
    done = wait_done(service, service.submit(wav_bytes(1.0, amplitude=0.0)).id)
    service.close()
    assert done.outcome == TalkOutcome.NOT_HEARD and not fake.calls


def test_an_error_of_the_model_gives_a_sentence() -> None:
    service, fake = make_service()
    fake.error = RuntimeError("model broke")
    done = wait_done(service, service.submit(wav_bytes(1.0)).id)
    service.close()
    assert done.outcome == TalkOutcome.FAILED and done.text == FAILED_TEXT


def test_too_many_waiting_questions_are_refused() -> None:
    service, _ = make_service(max_jobs=1)
    waiting = TalkJob("a" * 22, time.monotonic(), TalkStatus.PENDING)
    with service._lock:
        service._jobs[waiting.id] = waiting
    with pytest.raises(QueueFullError):
        service.submit(wav_bytes(1.0))
    service.close()


def test_only_the_newest_answers_are_kept() -> None:
    service, _ = make_service(max_jobs=2)
    ids: list[str] = []
    for _ in range(3):
        job = service.submit(wav_bytes(1.0))
        wait_done(service, job.id)
        ids.append(job.id)
    kept: list[str] = [i for i in ids if service.get(i) is not None]
    service.close()
    assert kept == ids[1:]


def test_a_question_after_the_shutdown_leaves_no_job() -> None:
    service, _ = make_service()
    service.close()
    with pytest.raises(RuntimeError):
        service.submit(wav_bytes(1.0))
    assert not service._jobs


def test_an_old_job_is_deleted() -> None:
    service, _ = make_service(keep_seconds=10)
    job = service.submit(wav_bytes(1.0))
    wait_done(service, job.id)
    with service._lock:
        service._jobs[job.id] = dataclasses.replace(
            service._jobs[job.id], created=time.monotonic() - 11
        )
    assert service.get(job.id) is None
    service.close()


# ---- The routes ----


def build_client(
    token: str | None = TOKEN, speech: bool = True
) -> tuple[TestClient, TalkService]:
    service, _ = make_service(speech=speech)
    app: FastAPI = FastAPI()
    app.include_router(create_talk_router(service, token))
    return TestClient(app), service


def test_the_routes_give_the_answer_with_the_token() -> None:
    make_object(name="blue bottle")
    store_frame(12, BOTTLE)
    client, service = build_client()
    posted = client.post(TALK_ROUTE, content=wav_bytes(1.0), headers=AUTH)
    assert posted.status_code == 202
    assert posted.headers["cache-control"] == "no-store"
    job_id: str = posted.json()["id"]
    wait_done(service, job_id)
    got = client.get(f"{TALK_ROUTE}/{job_id}", headers=AUTH)
    service.close()
    assert got.status_code == 200 and got.headers["cache-control"] == "no-store"
    data = got.json()
    assert data["status"] == "done" and data["outcome"] == "found"
    assert "blue bottle" in data["text"]


def test_without_a_token_on_the_server_the_routes_are_off() -> None:
    client, service = build_client(token=None)
    posted = client.post(TALK_ROUTE, content=wav_bytes(1.0))
    got = client.get(f"{TALK_ROUTE}/{'a' * 22}")
    service.close()
    assert posted.status_code == 503 and posted.json()["error"] == MESSAGE_NO_TOKEN
    assert got.status_code == 503


@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic " + TOKEN])
def test_a_wrong_or_missing_token_is_refused(header: str | None) -> None:
    client, service = build_client()
    headers: dict[str, str] = {} if header is None else {"Authorization": header}
    posted = client.post(TALK_ROUTE, content=wav_bytes(1.0), headers=headers)
    got = client.get(f"{TALK_ROUTE}/{'a' * 22}", headers=headers)
    service.close()
    assert posted.status_code == 401 and got.status_code == 401


def test_listening_off_is_refused() -> None:
    client, service = build_client(speech=False)
    posted = client.post(TALK_ROUTE, content=wav_bytes(1.0), headers=AUTH)
    service.close()
    assert posted.status_code == 503 and posted.json()["error"] == MESSAGE_OFF


def test_an_empty_or_too_long_recording_is_refused() -> None:
    client, service = build_client()
    empty = client.post(TALK_ROUTE, content=b"", headers=AUTH)
    too_long = client.post(
        TALK_ROUTE, content=b"\0" * (service.max_bytes + 1), headers=AUTH
    )
    service.close()
    assert empty.status_code == 400 and too_long.status_code == 413


@pytest.mark.parametrize("job_id", ["a" * 22, "short", "../../etc/passwd0000000"])
def test_an_unknown_or_bad_id_is_not_found(job_id: str) -> None:
    client, service = build_client()
    got = client.get(f"{TALK_ROUTE}/{job_id}", headers=AUTH)
    service.close()
    assert got.status_code == 404
