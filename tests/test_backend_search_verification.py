"""Tests of the check of weak hits in the search (stage I5). They use the actual database and storage.

The matching is the real code with fake models. The keypoint model is a fake that gives the number
that a test sets. The frames and the reference photos are real files.
Every object has a name that starts with `__test__ `. The objects go first, then the frames and files.
"""

import dataclasses
from collections.abc import Iterator
from datetime import timedelta

import numpy as np
import pytest

import db.api as db
import storage.api as storage
from ai.errors import InferenceError
from backend.search import Answer, AnswerSource, SearchService
from backend.settings import BackendConfig, SearchConfig, load_config
from db.api import sightings
from db.schema import EnrolledObject, Image, utc_now
from tests.ai_support import FakeVerifier, make_engine
from tests.db_support import skip_if_actual_data_has_problems
from tests.search_support import (
    BOTTLE,
    OTHER,
    PREFIX,
    WEAK,
    direction,
    make_object_with_references,
    store_frame,
)

# Reference views. WEAK has a cosine of 0.91 to BOTTLE: above tau (0.8), below tau_strong (0.99).
VIEW: np.ndarray = direction(1, 0)


@pytest.fixture(autouse=True)
def files() -> Iterator[list[str]]:
    """Give the test a list. The test appends the paths of the reference files that it stored."""
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    before: set[str] = set(db.list_image_paths())
    files: list[str] = []
    yield files
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))
    storage.delete_images(files)
    database.close()


def make_service(verifier: FakeVerifier, **changes: object) -> SearchService:
    base: BackendConfig = load_config()
    search: SearchConfig = dataclasses.replace(base.search, **changes)
    return SearchService(
        dataclasses.replace(base, search=search), make_engine(verifier=verifier)
    )


def find(service: SearchService, obj: EnrolledObject) -> Answer:
    return service.find(obj.id, utc_now())


def test_a_weak_hit_that_passes_the_check_is_a_firm_answer(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, WEAK)
    verifier: FakeVerifier = FakeVerifier(answer=100)
    answer: Answer = find(make_service(verifier), obj)
    assert answer.found and answer.source == AnswerSource.ARCHIVE
    assert answer.strong is True  # The page does not say "maybe"
    assert len(verifier.calls) == 1


def test_the_firm_answer_keeps_the_similarity_of_the_match(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, WEAK)
    answer: Answer = find(make_service(FakeVerifier(100)), obj)
    assert answer.similarity is not None and 0.8 <= answer.similarity < 0.99


def test_a_weak_hit_that_fails_the_check_stays_a_maybe(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, WEAK)
    answer: Answer = find(make_service(FakeVerifier(answer=3)), obj)
    assert answer.found and answer.strong is False  # The check never rejects


def test_a_strong_hit_is_not_checked(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, BOTTLE)
    verifier: FakeVerifier = FakeVerifier()
    answer: Answer = find(make_service(verifier), obj)
    assert answer.strong is True and verifier.calls == []


def test_the_check_is_off_when_verify_weak_is_false(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, WEAK)
    verifier: FakeVerifier = FakeVerifier(answer=100)
    answer: Answer = find(make_service(verifier, verify_weak=False), obj)
    assert answer.strong is False and verifier.calls == []


def test_the_check_uses_at_most_verify_max_references_photos(files: list[str]) -> None:
    obj = make_object_with_references(
        files,
        [
            (side, direction(*[1 if i == n else 0 for i in range(5)]))
            for n, side in enumerate((20, 30, 40, 50))
        ],
    )
    store_frame(10, WEAK)
    verifier: FakeVerifier = FakeVerifier(answer=0)
    find(make_service(verifier, verify_max_references=2), obj)
    assert len(verifier.calls) == 2


def test_an_error_of_the_verifier_leaves_the_hit_weak_and_the_search_works(
    files: list[str],
) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, WEAK)
    verifier: FakeVerifier = FakeVerifier()
    verifier.error = InferenceError("test")
    answer: Answer = find(make_service(verifier), obj)
    assert answer.found and answer.strong is False


def test_a_frame_that_cannot_be_read_leaves_the_hit_weak(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    frame: Image = store_frame(10, WEAK)
    storage.delete_images([frame.file_path])
    verifier: FakeVerifier = FakeVerifier(answer=100)
    answer: Answer = find(make_service(verifier), obj)
    assert answer.found and answer.strong is False and verifier.calls == []


def test_reference_photos_that_cannot_be_read_leave_the_hit_weak(
    files: list[str],
) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, WEAK)
    storage.delete_images(list(files))
    verifier: FakeVerifier = FakeVerifier(answer=100)
    answer: Answer = find(make_service(verifier), obj)
    assert answer.found and answer.strong is False and verifier.calls == []


def test_a_stale_record_is_replaced_by_a_firm_archive_answer(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    old: Image = store_frame(60 * 30, BOTTLE)
    sightings.set_last_seen(
        obj.id,
        old.id,
        (20, 15, 60, 45),
        utc_now() - timedelta(hours=30),
        0.9,
        0.5,
        True,
        False,
    )
    newer: Image = store_frame(10, WEAK)
    answer: Answer = find(make_service(FakeVerifier(100)), obj)
    assert answer.image_id == newer.id and answer.strong is True and not answer.stale


def test_a_fresh_record_is_not_checked(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    frame: Image = store_frame(10, WEAK)
    sightings.set_last_seen(
        obj.id,
        frame.id,
        (20, 15, 60, 45),
        utc_now() - timedelta(minutes=10),
        0.9,
        0.5,
        False,
        False,
    )
    verifier: FakeVerifier = FakeVerifier(answer=100)
    answer: Answer = find(make_service(verifier), obj)
    assert answer.source == AnswerSource.LAST_SEEN and verifier.calls == []


def test_a_frame_with_another_thing_gives_no_check_and_no_answer(
    files: list[str],
) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, OTHER)
    verifier: FakeVerifier = FakeVerifier(answer=100)
    assert not find(make_service(verifier), obj).found and verifier.calls == []


def test_the_search_does_not_change_any_data(files: list[str]) -> None:
    obj = make_object_with_references(files, [(40, VIEW)])
    store_frame(10, WEAK)
    before: int = db.vector_count()
    find(make_service(FakeVerifier(100)), obj)
    assert sightings.get_last_seen(obj.id) is None and db.vector_count() == before
