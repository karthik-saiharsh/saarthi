"""Tests of the search (stage Q1). They use the actual database and index.

The matching is the real code, with fake models. Every object has a name that
starts with `__test__ `. The objects go first (the last-seen records and the
sightings go by cascade), then the frames and their vectors.
"""

import dataclasses
from collections.abc import Iterator
from datetime import datetime, timedelta

import numpy as np
import pytest

import db.api as db
from backend.errors import NotFoundError, ObjectNotReadyError
from backend.search import Answer, AnswerSource, SearchService
from backend.settings import BackendConfig, load_config
from db.api import sightings
from db.schema import (
    EnrolledObject,
    Image,
    ImageStatus,
    Region,
)
from tests.ai_support import make_engine
from tests.db_support import (
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
)
from tests.search_support import (
    BOTTLE,
    OTHER,
    PREFIX,
    TAU,
    TAU_STRONG,
    WEAK,
    make_object,
    model,
)

NOW: datetime = datetime(2000, 6, 1, 12, 0, 0)
BOX: tuple[int, int, int, int] = (10, 20, 60, 80)


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    yield
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    db.delete_images(
        [i.id for i in Image.select().where(Image.file_path.startswith(TEST_PREFIX))]
    )


def make_service(**changes: int) -> SearchService:
    base: BackendConfig = load_config()
    config: BackendConfig = dataclasses.replace(
        base, search=dataclasses.replace(base.search, **changes)
    )
    return SearchService(config, make_engine())


def make_frame(
    seen_at: datetime,
    vector: np.ndarray = BOTTLE,
    in_hand: bool = False,
    quality_ok: bool = True,
) -> Image:
    """A `done` frame with one region and its vector."""
    image: Image = make_image()
    region: Region = make_regions(image, 1)[0]
    Region.update(in_hand=in_hand, quality_ok=quality_ok).where(
        Region.id == region.id
    ).execute()
    db.store_region_vectors([region.id], vector.reshape(1, -1), model())
    set_status(image, ImageStatus.DONE)
    Image.update(captured_at=seen_at).where(Image.id == image.id).execute()
    return image


def set_record(obj: EnrolledObject, image: Image, seen_at: datetime) -> None:
    sightings.set_last_seen(obj.id, image.id, BOX, seen_at, 0.9, 0.5, True, True)


def minutes(count: int) -> datetime:
    return NOW - timedelta(minutes=count)


# ---- The record ----


def test_a_fresh_record_is_the_answer_and_the_archive_is_not_searched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = make_object()
    image = make_frame(minutes(30))
    set_record(obj, image, minutes(30))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the archive must not be searched")

    monkeypatch.setattr(db, "list_searchable_regions", forbidden)
    answer: Answer = make_service().find(obj.id, NOW)
    assert answer.found and answer.source == AnswerSource.LAST_SEEN
    assert answer.image_id == image.id and answer.box == BOX
    assert answer.age_seconds == 30 * 60 and answer.in_hand is True
    assert answer.strong is True and not answer.stale and answer.complete


def test_a_stale_record_is_replaced_by_a_newer_sighting_in_the_archive() -> None:
    obj = make_object()
    old = make_frame(minutes(60 * 30))
    set_record(obj, old, minutes(60 * 30))  # 30 hours: stale
    newer = make_frame(minutes(40))
    answer: Answer = make_service().find(obj.id, NOW)
    assert answer.source == AnswerSource.ARCHIVE and answer.image_id == newer.id
    assert not answer.stale


def test_a_stale_record_stays_the_answer_when_the_archive_has_nothing_newer() -> None:
    obj = make_object()
    old = make_frame(minutes(60 * 30))
    set_record(obj, old, minutes(60 * 30))
    make_frame(minutes(20), OTHER)  # A frame without the object
    answer: Answer = make_service().find(obj.id, NOW)
    assert answer.source == AnswerSource.LAST_SEEN and answer.image_id == old.id
    assert answer.stale and answer.found


def test_the_archive_does_not_return_a_sighting_older_than_the_record() -> None:
    obj = make_object()
    make_frame(minutes(60 * 40))  # Before the record
    record_frame = make_frame(minutes(60 * 30))
    set_record(obj, record_frame, minutes(60 * 30))
    answer: Answer = make_service().find(obj.id, NOW)
    assert answer.image_id == record_frame.id and answer.stale


# ---- The archive ----


def test_the_newest_matching_frame_wins() -> None:
    obj = make_object()
    make_frame(minutes(50))
    newest = make_frame(minutes(10))
    make_frame(minutes(5), OTHER)  # Newer, but another thing
    answer: Answer = make_service().find(obj.id, NOW)
    assert answer.source == AnswerSource.ARCHIVE and answer.image_id == newest.id
    assert answer.seen_at == minutes(10) and answer.age_seconds == 600
    assert answer.box is not None and answer.strong is True


def test_the_search_goes_back_over_several_windows() -> None:
    obj = make_object()
    far = make_frame(minutes(60 * 5 + 30))
    answer: Answer = make_service(window_minutes=60).find(obj.id, NOW)
    assert answer.image_id == far.id


def test_a_weak_match_is_returned_and_marked_weak() -> None:
    obj = make_object()
    make_frame(minutes(15), WEAK, in_hand=True)
    answer: Answer = make_service().find(obj.id, NOW)
    assert answer.found and answer.strong is False and answer.in_hand is True
    assert answer.similarity is not None and TAU <= answer.similarity < TAU_STRONG


def test_a_frame_that_is_not_a_match_gives_no_answer() -> None:
    obj = make_object()
    make_frame(minutes(15), OTHER)
    answer: Answer = make_service().find(obj.id, NOW)
    assert not answer.found and answer.source is None and answer.complete
    assert answer.seen_at is None and answer.image_id is None


def test_a_region_that_is_not_good_is_ignored() -> None:
    obj = make_object()
    make_frame(minutes(15), quality_ok=False)
    assert not make_service().find(obj.id, NOW).found


def test_a_frame_older_than_the_scan_limit_is_not_found() -> None:
    obj = make_object()
    make_frame(minutes(60 * 80))
    assert not make_service(max_scan_hours=72).find(obj.id, NOW).found
    assert make_service(max_scan_hours=100).find(obj.id, NOW).found


def test_the_frame_limit_stops_the_search_and_the_answer_says_so() -> None:
    obj = make_object()
    make_frame(minutes(5), OTHER)
    make_frame(minutes(10), OTHER)
    make_frame(minutes(20))  # The match is the third frame
    stopped: Answer = make_service(max_frames=2).find(obj.id, NOW)
    assert not stopped.found and not stopped.complete
    assert make_service(max_frames=3).find(obj.id, NOW).found


def test_a_look_alike_with_the_same_views_blocks_the_sighting() -> None:
    owner = make_object()
    make_object(tracked=False)  # The same views: the margin is 0
    make_frame(minutes(15))
    assert not make_service().find(owner.id, NOW).found


def test_frames_with_vectors_of_another_model_are_not_compared() -> None:
    obj = make_object()
    image = make_frame(minutes(15))
    Region.update(embedding_model="old-model").where(Region.image == image.id).execute()
    assert not make_service().find(obj.id, NOW).found


# ---- Objects that cannot be searched ----


def test_an_unknown_object_is_not_found() -> None:
    with pytest.raises(NotFoundError):
        make_service().find(2**31 - 1, NOW)


def test_a_draft_object_cannot_be_searched() -> None:
    draft = make_object(ready=False)
    with pytest.raises(ObjectNotReadyError):
        make_service().find(draft.id, NOW)


def test_a_look_alike_cannot_be_searched() -> None:
    twin = make_object(tracked=False)
    with pytest.raises(ObjectNotReadyError):
        make_service().find(twin.id, NOW)


def test_the_search_does_not_change_any_data() -> None:
    obj = make_object()
    make_frame(minutes(15))
    before: int = db.vector_count()
    make_service().find(obj.id, NOW)
    assert sightings.get_last_seen(obj.id) is None and db.vector_count() == before


def test_find_with_picture_searches_once_and_keeps_the_answer_without_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = make_object()
    make_frame(minutes(15))
    service: SearchService = make_service()
    calls: list[int] = []
    original = service.find
    monkeypatch.setattr(
        service, "find", lambda *a, **k: calls.append(1) or original(*a, **k)
    )
    # The frame of this test has no real file: the answer stays, the picture is None.
    answer, picture = service.find_with_picture(obj.id, 128, NOW)
    assert answer.found and picture is None
    assert calls == [1]


def test_find_with_picture_gives_none_when_the_object_was_not_seen() -> None:
    obj = make_object()
    answer, picture = make_service().find_with_picture(obj.id, 128, NOW)
    assert not answer.found and picture is None
