"""Tests of the UI adapter for the find page. They use the actual database and storage, and fake ai models.

Every object has a name that starts with `__test__ `. The objects go first (a last-seen
record keeps its frame), then the frames and their files.
"""

import asyncio
import dataclasses
import logging
from collections.abc import Iterator

import pytest

import db.api as db
import storage.api as storage
from backend.enrollment import EnrollmentService
from backend.query import QueryResolver
from backend.search import SearchService
from backend.settings import BackendConfig, load_config
from backend.ui_adapter import (
    MESSAGE_FIND_FAILED,
    MESSAGE_FIND_NOT_FOUND,
    MESSAGE_FIND_NOT_READY,
    UiFindService,
)
from db.schema import EnrolledObject, Image
from tests.ai_support import make_engine
from tests.db_support import skip_if_actual_data_has_problems
from tests.search_support import (
    BOTTLE,
    PREFIX,
    WEAK,
    make_object,
    store_frame,
)
from ui.ports import FindError, FindResult, FindStatus


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


def make_service() -> tuple[UiFindService, QueryResolver]:
    base: BackendConfig = load_config()
    query = dataclasses.replace(
        base.query, matched_min_score=0.4, matched_min_gap=0.1, suggest_min_score=0.2
    )
    config: BackendConfig = dataclasses.replace(base, query=query)
    engine = make_engine()
    resolver: QueryResolver = QueryResolver(config, engine)
    service: UiFindService = UiFindService(
        resolver,
        SearchService(config, engine),
        EnrollmentService(config, engine),
    )
    return service, resolver


def ask(service: UiFindService, text: str) -> FindResult:
    return asyncio.run(service.ask(text))


def test_only_ready_objects_that_are_not_look_alikes_can_be_chosen() -> None:
    ready = make_object(name="blue bottle")
    make_object(name="draft bottle", ready=False)
    make_object(name="twin bottle", tracked=False)
    choices = asyncio.run(make_service()[0].list_choices())
    mine = [c for c in choices if c.name.startswith(PREFIX)]
    assert [(c.object_id, c.name) for c in mine] == [(ready.id, PREFIX + "blue bottle")]


def test_words_for_a_seen_object_give_the_sighting_with_a_picture() -> None:
    make_object(name="blue bottle", phrase="a bottle")
    store_frame(12, BOTTLE, in_hand=True)
    result: FindResult = ask(make_service()[0], "where is my blue bottle")
    assert result.status == FindStatus.FOUND and result.complete
    assert result.object_name == PREFIX + "blue bottle"
    sighting = result.sighting
    assert sighting is not None and sighting.strong and sighting.in_hand
    assert not sighting.stale and 12 * 60 <= sighting.age_seconds < 12 * 60 + 60
    assert sighting.picture is not None and sighting.picture.startswith(b"\xff\xd8")


def test_a_weak_match_is_marked_weak() -> None:
    object_id: int = make_object().id
    store_frame(5, WEAK)
    result: FindResult = asyncio.run(make_service()[0].find(object_id))
    assert result.sighting is not None and result.sighting.strong is False


def test_choosing_an_object_uses_its_name_from_the_gallery() -> None:
    obj = make_object(name="blue bottle")
    result: FindResult = asyncio.run(make_service()[0].find(obj.id))
    assert result.status == FindStatus.NOT_SEEN and result.sighting is None
    assert result.object_name == PREFIX + "blue bottle" and result.complete


def test_words_that_fit_two_objects_give_a_choice() -> None:
    make_object(name="red cup", phrase="a cup")
    make_object(name="blue cup", phrase="a cup")
    result: FindResult = ask(make_service()[0], "my cup")
    assert result.status == FindStatus.CHOOSE and result.sighting is None
    assert {c.name for c in result.choices} == {PREFIX + "red cup", PREFIX + "blue cup"}


def test_words_that_fit_nothing_give_no_match() -> None:
    make_object(name="blue bottle", phrase="a bottle")
    assert ask(make_service()[0], "xylophone zebra").status == FindStatus.NO_MATCH


def test_an_answer_stays_when_the_file_of_the_frame_is_gone() -> None:
    object_id: int = make_object().id
    frame: Image = store_frame(10)
    storage.delete_images([frame.file_path])
    result: FindResult = asyncio.run(make_service()[0].find(object_id))
    assert result.status == FindStatus.FOUND
    assert result.sighting is not None and result.sighting.picture is None


def test_errors_become_fixed_texts() -> None:
    service, _ = make_service()
    draft: int = make_object(ready=False).id
    with pytest.raises(FindError) as not_ready:
        asyncio.run(service.find(draft))
    assert str(not_ready.value) == MESSAGE_FIND_NOT_READY
    with pytest.raises(FindError) as unknown:
        asyncio.run(service.find(2**31 - 1))
    assert str(unknown.value) == MESSAGE_FIND_NOT_FOUND
    with pytest.raises(FindError) as empty:
        ask(service, "   ")
    assert str(empty.value) == "Write what you are looking for, in a few words."


def test_an_unexpected_error_gives_a_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, resolver = make_service()

    def broken(text: str) -> None:
        raise RuntimeError("secret internal text /home/x")

    monkeypatch.setattr(resolver, "resolve", broken)
    with pytest.raises(FindError) as error:
        ask(service, "my bottle")
    assert str(error.value) == MESSAGE_FIND_FAILED
    assert "secret" not in str(error.value)


def test_the_words_of_the_user_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    make_object(name="blue bottle", phrase="a bottle")
    with caplog.at_level(logging.DEBUG):
        ask(make_service()[0], "my private blue bottle")
    assert "private" not in caplog.text


def test_the_backend_gives_a_working_service_to_the_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.api import Backend

    monkeypatch.delenv("SAARTHI_API_TOKEN", raising=False)
    backend: Backend = Backend(make_engine())
    make_object(name="blue bottle")
    service = backend.find_service
    assert isinstance(service, UiFindService)
    choices = asyncio.run(service.list_choices())
    assert PREFIX + "blue bottle" in [c.name for c in choices]
