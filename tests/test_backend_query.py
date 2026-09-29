"""Tests of the resolver: from the words of a user to an object. They use the actual database.

The query encoder is a fake that hashes words, so texts that share words are close.
Every object has a name that starts with `__test__ `.
"""

import dataclasses
from collections.abc import Iterator

import pytest

import db.api as db
from backend.errors import InvalidQueryError
from backend.query import (
    QueryResolver,
    QueryResult,
    QueryStatus,
    clean_query,
)
from backend.settings import BackendConfig, load_config
from db.api import gallery
from db.schema import EnrolledObject
from tests.ai_support import FakeQueryEncoder, make_engine

PREFIX: str = "__test__ "


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, _ = db.init_db()
    if gallery.gallery_phrases():
        pytest.skip("The actual gallery has ready objects")
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    yield
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()


def make_object(
    name: str,
    phrase: str,
    synonyms: list[str] | None = None,
    ready: bool = True,
    tracked: bool = True,
) -> EnrolledObject:
    obj: EnrolledObject = gallery.create_object(
        PREFIX + name, phrase, synonyms, tracked
    )
    if ready:
        gallery.set_calibration(obj.id, 0.8, 0.85, 0.03, "test-model")
    return obj


def make_resolver(
    encoder: FakeQueryEncoder | None = None, **changes: float
) -> QueryResolver:
    base: BackendConfig = load_config()
    query = dataclasses.replace(
        base.query,
        matched_min_score=0.4,
        matched_min_gap=0.1,
        suggest_min_score=0.2,
        **changes,
    )
    config: BackendConfig = dataclasses.replace(base, query=query)
    return QueryResolver(config, make_engine(query_encoder=encoder))


def names(result: QueryResult) -> list[str]:
    return [c.name.removeprefix(PREFIX) for c in result.candidates]


def test_words_that_name_one_object_match_it() -> None:
    make_object("blue bottle", "a bottle", ["a water bottle"])
    make_object("house keys", "a set of keys")
    make_object("tea cup", "a cup", ["a mug"])
    result: QueryResult = make_resolver().resolve("where is my blue bottle")
    assert result.status == QueryStatus.MATCHED and names(result) == ["blue bottle"]
    assert result.candidates[0].score > 0.4


def test_a_description_of_the_object_matches_it() -> None:
    make_object("blue bottle", "a bottle", ["a water bottle"])
    make_object("house keys", "a set of keys")
    result: QueryResult = make_resolver().resolve("the water bottle that is blue")
    assert result.status == QueryStatus.MATCHED and names(result) == ["blue bottle"]


def test_words_that_fit_two_objects_give_a_choice_best_first() -> None:
    make_object("red cup", "a cup")
    make_object("blue cup", "a cup")
    make_object("house keys", "a set of keys")
    result: QueryResult = make_resolver().resolve("my cup")
    assert result.status == QueryStatus.AMBIGUOUS
    assert set(names(result)) == {"red cup", "blue cup"}
    scores: list[float] = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_the_choice_has_at_most_max_candidates() -> None:
    for colour in ("red", "blue", "green", "black"):
        make_object(f"{colour} cup", "a cup")
    result: QueryResult = make_resolver(max_candidates=2).resolve("my cup")
    assert result.status == QueryStatus.AMBIGUOUS and len(result.candidates) == 2


def test_words_that_fit_nothing_give_none() -> None:
    make_object("blue bottle", "a bottle")
    result: QueryResult = make_resolver().resolve("xylophone zebra")
    assert result.status == QueryStatus.NONE and result.candidates == ()


def test_a_draft_object_and_a_look_alike_are_never_offered() -> None:
    make_object("blue bottle", "a bottle", ready=False)
    make_object("blue bottle twin", "a bottle", tracked=False)
    result: QueryResult = make_resolver().resolve("my blue bottle")
    assert result.status == QueryStatus.NONE


def test_no_objects_give_none_without_encoding() -> None:
    encoder: FakeQueryEncoder = FakeQueryEncoder()
    assert make_resolver(encoder).resolve("my bottle").status == QueryStatus.NONE
    assert encoder.calls == []


def test_a_description_is_encoded_once() -> None:
    encoder: FakeQueryEncoder = FakeQueryEncoder()
    make_object("blue bottle", "a bottle")
    make_object("house keys", "a set of keys")
    resolver: QueryResolver = make_resolver(encoder)
    resolver.resolve("my bottle")
    assert len(encoder.calls) == 2  # The descriptions, then the question
    resolver.resolve("my keys")
    assert encoder.calls[2:] == [["my keys"]]  # Only the question


def test_a_new_object_is_found_and_a_deleted_object_is_forgotten() -> None:
    encoder: FakeQueryEncoder = FakeQueryEncoder()
    keys = make_object("house keys", "a set of keys")
    resolver: QueryResolver = make_resolver(encoder)
    assert names(resolver.resolve("my keys")) == ["house keys"]
    make_object("tea cup", "a cup")
    assert names(resolver.resolve("my cup")) == ["tea cup"]  # The new object counts
    gallery.delete_object(keys.id)
    after: QueryResult = resolver.resolve("my keys")
    assert "house keys" not in names(after)


def test_the_words_are_cleaned_and_bad_words_are_refused() -> None:
    assert clean_query("  where   is\nmy bottle ", 300) == "where is my bottle"
    for bad in ("", "   ", "x" * 301, "bell\x07", "line\x00"):
        with pytest.raises(InvalidQueryError):
            clean_query(bad, 300)
    with pytest.raises(InvalidQueryError):
        make_resolver().resolve("")


def test_the_words_of_the_user_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    make_object("blue bottle", "a bottle")
    with caplog.at_level("DEBUG"):
        make_resolver().resolve("my secret blue bottle")
    assert "secret" not in caplog.text and "blue" not in caplog.text
