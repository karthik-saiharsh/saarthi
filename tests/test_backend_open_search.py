"""Tests of the open search: objects that nobody registered. They use the actual database and storage, and fake ai models.

The frames are made with `store_frame` (a real file and a registered region). The boxes of the broad detector are added
with vectors that the fake scene model gives, so a test says which phrase fits a box.
"""

import asyncio
import dataclasses
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import db.api as db
import storage.api as storage
from ai.errors import InferenceError
from backend.describe import DescribeService
from backend.enrollment import EnrollmentService
from backend.http_search import SEARCH_ROUTE, create_search_router
from backend.open_search import OpenSearchService, phrase_from_words
from backend.query import QueryResolver
from backend.search import AnswerSource, SearchService
from backend.settings import BackendConfig, load_config
from backend.ui_adapter import UiFindService
from db.open_regions import NewOpenRegion
from db.schema import EnrolledObject, Image
from tests.ai_support import FakeLanguageModel, FakeSceneClassifier, make_engine
from tests.db_support import skip_if_actual_data_has_problems
from tests.search_support import PREFIX, store_frame
from ui.ports import FindStatus

SCENE = FakeSceneClassifier()


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    db.init_db()
    skip_if_actual_data_has_problems()
    if db.gallery.gallery_phrases():
        pytest.skip("The actual gallery has ready objects")
    before: set[str] = set(db.list_image_paths())
    yield
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))


def make_service(
    language: FakeLanguageModel | None = None, **changes: object
) -> OpenSearchService:
    base: BackendConfig = load_config()
    config = dataclasses.replace(
        base, open_search=dataclasses.replace(base.open_search, **changes)
    )
    engine = make_engine(scene_classifier=SCENE, language_model=language)
    return OpenSearchService(config, engine)


def add_box(frame: Image, label: str, shows: str | None = None) -> None:
    """A box with a name from the detector. `shows` is the phrase that its picture fits (default: the name)."""
    vector = SCENE._vector(shows or label).reshape(1, -1)
    db.open_regions.add_open_regions(
        frame.id,
        [NewOpenRegion(10, 10, 50, 40, label, 0.9)],
        vector,
        SCENE.name,
    )


# ---- The phrase ----


@pytest.mark.parametrize(
    ("words", "phrase"),
    [
        ("laptop", "laptop"),
        ("Where is my laptop?", "laptop"),
        ("where did I leave my reading glasses", "reading glasses"),
        ("I can't find my keys", "keys"),
        ("I cant find the remote", "remote"),
        ("have you seen my wallet please", "wallet"),
        ("I'm looking for a blue notebook", "blue notebook"),
        ("show me the umbrella", "umbrella"),
        ("that thing I check my mails on", "thing i check my mails on"),
        ("where is my", None),
        ("   ?  ", None),
    ],
)
def test_the_phrase_has_no_question_words(words: str, phrase: str | None) -> None:
    assert phrase_from_words(words) == phrase


# ---- Step 1: the names ----


def test_a_name_that_fits_gives_the_answer_with_the_age_and_the_box() -> None:
    frame = store_frame(12)
    add_box(frame, "laptop")
    found = make_service().find("where is my laptop")
    assert found is not None and found.via == "label" and found.label == "laptop"
    assert found.answer.source == AnswerSource.OPEN and found.answer.strong is False
    assert found.answer.image_id == frame.id and found.answer.box == (10, 10, 50, 40)
    assert 11 * 60 <= (found.answer.age_seconds or 0) <= 13 * 60


def test_the_newest_frame_wins() -> None:
    old, new = store_frame(50), store_frame(5)
    add_box(old, "laptop")
    add_box(new, "laptop")
    found = make_service().find("laptop")
    assert found is not None and found.answer.image_id == new.id


def test_a_less_exact_name_does_not_beat_an_exact_name_in_an_older_frame() -> None:
    old, new = store_frame(50), store_frame(5)
    add_box(old, "laptop")
    add_box(
        new, "laptop keyboard"
    )  # The fake score is 0.71, outside the tier of the best name
    found = make_service().find("laptop")
    assert found is not None and found.answer.image_id == old.id


def test_a_name_that_does_not_fit_is_not_used() -> None:
    add_box(
        store_frame(5), "laptop keyboard"
    )  # Score 0.71 for "keyboard", but not for "umbrella"
    assert make_service(visual_min_probability=1.0).find("umbrella") is None


def test_boxes_older_than_the_window_are_not_searched() -> None:
    add_box(store_frame(5 * 60), "laptop")
    assert make_service(max_scan_hours=2).find("laptop") is None
    assert make_service(max_scan_hours=10).find("laptop") is not None


# ---- Step 2: the pictures ----


def test_the_picture_finds_a_box_when_no_name_fits() -> None:
    frame = store_frame(8)
    add_box(frame, "stuff", shows="umbrella")  # The name is wrong, the picture fits
    found = make_service().find("umbrella")
    assert found is not None and found.via == "visual" and found.label == "stuff"
    assert found.answer.image_id == frame.id and found.score > 0.9


def test_the_picture_needs_a_fit_better_than_the_neutral_phrases() -> None:
    add_box(store_frame(8), "stuff", shows="a wall")  # It fits a neutral phrase exactly
    assert make_service().find("umbrella") is None


def test_no_box_that_fits_gives_none() -> None:
    add_box(store_frame(8), "stuff", shows="piano")
    assert make_service().find("umbrella") is None


def test_a_description_is_searched_as_it_is() -> None:
    frame = store_frame(8)
    add_box(frame, "stuff", shows="thing i check my mails on")
    found = make_service().find("that thing I check my mails on")
    assert found is not None and found.via == "visual"


# ---- Step 3: the rewrite ----

DESCRIPTION = "that thing I check my mails on"
PHRASE = "thing i check my mails on"  # The words without "that", as the search gives them to the model


def test_a_description_is_rewritten_into_names_and_a_name_fits() -> None:
    frame = store_frame(8)
    add_box(frame, "laptop")
    language = FakeLanguageModel({PHRASE: ["computer", "laptop", "tablet"]})
    found = make_service(language).find(DESCRIPTION)
    assert found is not None and found.via == "rewrite" and found.label == "laptop"
    assert found.answer.image_id == frame.id and language.calls == [PHRASE]


def test_the_best_name_wins() -> None:
    old, new = store_frame(50), store_frame(5)
    add_box(old, "laptop")
    add_box(new, "tablet")
    language = FakeLanguageModel({PHRASE: ["laptop", "tablet"]})
    found = make_service(language).find(DESCRIPTION)
    # Both names fit with the score 1.0. The first name found with that score stays.
    assert found is not None and found.label == "laptop"


def test_a_name_is_not_rewritten() -> None:
    # "piano" is a name. The model would give a neighbour ("keyboard"), and a wrong box would fit it.
    add_box(store_frame(8), "laptop keyboard")
    language = FakeLanguageModel({"piano": ["keyboard"]})
    assert make_service(language).find("where is my piano") is None
    assert language.calls == []


def test_a_name_from_the_model_must_fit_almost_exactly() -> None:
    add_box(store_frame(8), "laptop keyboard")
    language = FakeLanguageModel(
        {PHRASE: ["laptop"]}
    )  # The score with "laptop keyboard" is 0.71
    assert make_service(language).find(DESCRIPTION) is None
    assert make_service(language, rewrite_min_score=0.7).find(DESCRIPTION) is not None


def test_the_rewrite_runs_only_when_the_other_steps_fail() -> None:
    add_box(store_frame(8), "laptop")
    language = FakeLanguageModel({"laptop": ["computer"]})
    found = make_service(language).find("laptop")
    assert found is not None and found.via == "label" and language.calls == []


def test_names_that_fit_no_box_give_none() -> None:
    add_box(store_frame(8), "laptop")
    language = FakeLanguageModel({PHRASE: ["piano", "umbrella"]})
    assert make_service(language).find(DESCRIPTION) is None
    assert language.calls == [PHRASE]


def test_a_switched_off_language_model_is_not_called() -> None:
    add_box(store_frame(8), "laptop")
    service = make_service(FakeLanguageModel({PHRASE: ["laptop"]}))
    service._engine._language = None
    assert service.find(DESCRIPTION) is None


def test_an_error_of_the_language_model_gives_none() -> None:
    add_box(store_frame(8), "laptop")
    language = FakeLanguageModel({PHRASE: ["laptop"]})
    language.error = InferenceError("test")
    assert make_service(language).find(DESCRIPTION) is None


def test_the_log_never_shows_the_words_or_the_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    add_box(store_frame(8), "laptop")
    language = FakeLanguageModel({PHRASE: ["laptop"]})
    with caplog.at_level(logging.DEBUG):
        make_service(language).find(DESCRIPTION)
    assert "mails" not in caplog.text


# ---- Other cases ----


def test_a_switched_off_detector_or_scene_model_searches_nothing() -> None:
    add_box(store_frame(8), "laptop")
    service = make_service()
    service._engine._open_detector = None
    assert not service.enabled and service.find("laptop") is None


def test_no_words_and_no_box_give_none() -> None:
    assert make_service().find("where is my") is None
    assert make_service().find("laptop") is None  # No stored box of this test


def test_the_picture_is_cut_from_the_frame() -> None:
    add_box(store_frame(8), "laptop")
    found, picture = make_service().find_with_picture("laptop", 64)
    assert found is not None and picture is not None
    assert picture[:2] == b"\xff\xd8"  # A JPEG


def test_a_missing_file_gives_the_answer_without_a_picture() -> None:
    frame = store_frame(8)
    add_box(frame, "laptop")
    storage.delete_images([frame.file_path])
    found, picture = make_service().find_with_picture("laptop")
    assert found is not None and picture is None


# ---- The find page and the route ----


def make_ui(open_search: OpenSearchService | None) -> UiFindService:
    config: BackendConfig = load_config()
    engine = make_engine(scene_classifier=SCENE)
    return UiFindService(
        QueryResolver(config, engine),
        SearchService(config, engine),
        EnrollmentService(config, engine),
        DescribeService(config, engine),
        open_search,
    )


def test_the_find_page_shows_a_weak_result_for_an_unregistered_object() -> None:
    add_box(store_frame(12), "laptop")
    result = asyncio.run(make_ui(make_service()).ask("where is my laptop"))
    assert result.status == FindStatus.FOUND and result.object_name == "Laptop"
    assert result.sighting is not None and result.sighting.strong is False
    assert result.sighting.picture is not None
    assert result.description is not None and "a laptop" in result.description
    assert "I am not sure" in result.description and "your" not in result.description


def test_the_find_page_says_no_match_when_nothing_fits() -> None:
    add_box(store_frame(12), "laptop")
    result = asyncio.run(make_ui(make_service()).ask("where is my umbrella"))
    assert result.status == FindStatus.NO_MATCH


def test_without_the_open_search_the_old_answer_stays() -> None:
    add_box(store_frame(12), "laptop")
    result = asyncio.run(make_ui(None).ask("where is my laptop"))
    assert result.status == FindStatus.NO_MATCH


def test_the_route_gives_status_open_and_the_label() -> None:
    add_box(store_frame(12), "laptop")
    config: BackendConfig = load_config()
    engine = make_engine(scene_classifier=SCENE)
    app = FastAPI()
    app.include_router(
        create_search_router(
            QueryResolver(config, engine),
            SearchService(config, engine),
            None,
            OpenSearchService(config, engine),
        )
    )
    client = TestClient(app)
    answer = client.post(SEARCH_ROUTE, json={"query": "where is my laptop"}).json()
    assert answer["status"] == "open" and answer["candidates"] == []
    assert answer["answer"]["label"] == "laptop" and answer["answer"]["strong"] is False
    assert answer["answer"]["source"] == "open"
    none = client.post(SEARCH_ROUTE, json={"query": "where is my umbrella"}).json()
    assert none["status"] == "none" and none["answer"] is None
