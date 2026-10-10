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
from ai.types import Box, RawDetection
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
from tests.ai_support import (
    FakeDetector,
    FakeLanguageModel,
    FakeSceneClassifier,
    make_engine,
)
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
    language: FakeLanguageModel | None = None,
    detector: FakeDetector | None = None,
    **changes: object,
) -> OpenSearchService:
    base: BackendConfig = load_config()
    config = dataclasses.replace(
        base, open_search=dataclasses.replace(base.open_search, **changes)
    )
    engine = make_engine(
        detector=detector, scene_classifier=SCENE, language_model=language
    )
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


# ---- The mode "verified" ----

HIT = RawDetection(Box(10, 10, 50, 40), 0.8, "laptop", "full")  # The box of add_box
FAR = RawDetection(Box(60, 2, 78, 12), 0.9, "laptop", "full")  # No overlap with it
WEAK_HIT = RawDetection(Box(10, 10, 50, 40), 0.3, "laptop", "full")  # Below 0.35


class ScriptedDetector(FakeDetector):
    """Gives one list of boxes for each call, in order. After that it gives nothing."""

    def __init__(self, *results: list[RawDetection]) -> None:
        super().__init__()
        self.results: list[list[RawDetection]] = list(results)

    def detect(self, image, phrases, *args, **kwargs):
        super().detect(
            image, phrases, *args, **kwargs
        )  # Records, and fails once if set
        return self.results.pop(0) if self.results else []


def verified(
    detector: FakeDetector,
    language: FakeLanguageModel | None = None,
    **changes: object,
) -> OpenSearchService:
    return make_service(language, detector, mode="verified", **changes)


def test_the_mode_decides_the_path() -> None:
    frame = store_frame(8)
    add_box(frame, "laptop")
    classic = make_service(detector=ScriptedDetector([HIT]))
    found = classic.find("laptop")
    assert found is not None and found.via == "label"
    assert classic._engine._detector.phrase_calls == []  # type: ignore[attr-defined]
    detector = ScriptedDetector([HIT])
    found = verified(detector).find("laptop")
    assert found is not None and found.via == "verified" and found.label == "laptop"
    assert len(detector.phrase_calls) == 1


def test_the_classic_mode_stays_the_default_and_is_unchanged() -> None:
    assert load_config().open_search.mode == "classic"
    add_box(store_frame(8), "stuff", shows="umbrella")
    detector = ScriptedDetector()
    found = make_service(detector=detector).find("umbrella")
    assert found is not None and found.via == "visual"
    assert detector.phrase_calls == []


def test_a_wrong_mode_is_refused_by_the_service() -> None:
    with pytest.raises(AssertionError):
        make_service(mode="fast")


def test_a_frame_that_passes_gives_a_weak_answer_with_the_detector_score() -> None:
    frame = store_frame(12)
    add_box(frame, "laptop")
    found = verified(ScriptedDetector([HIT])).find("where is my laptop")
    assert found is not None and found.via == "verified" and found.score == 0.8
    assert found.answer.source == AnswerSource.OPEN and found.answer.strong is False
    assert found.answer.image_id == frame.id and found.answer.box == (10, 10, 50, 40)


def test_the_newest_frame_that_passes_wins() -> None:
    old, new = store_frame(50), store_frame(5)
    add_box(old, "laptop")
    add_box(new, "laptop")
    detector = ScriptedDetector([HIT], [HIT])
    found = verified(detector).find("laptop")
    assert found is not None and found.answer.image_id == new.id
    assert len(detector.phrase_calls) == 1  # It stops at the first frame that passes


def test_a_failed_frame_gives_way_to_an_older_frame_that_passes() -> None:
    old, new = store_frame(50), store_frame(5)
    add_box(old, "laptop")
    add_box(new, "laptop")
    detector = ScriptedDetector([], [HIT])
    found = verified(detector).find("laptop")
    assert found is not None and found.answer.image_id == old.id
    assert len(detector.phrase_calls) == 2


def test_no_frame_that_passes_gives_none() -> None:
    add_box(store_frame(5), "laptop")
    add_box(store_frame(50), "laptop")
    assert verified(ScriptedDetector([], [])).find("laptop") is None


@pytest.mark.parametrize("detection", [FAR, WEAK_HIT])
def test_a_detector_box_with_a_low_score_or_far_from_the_box_fails(
    detection: RawDetection,
) -> None:
    add_box(store_frame(5), "laptop")
    assert verified(ScriptedDetector([detection])).find("laptop") is None


def test_the_thresholds_of_the_check_come_from_the_config() -> None:
    add_box(store_frame(5), "laptop")
    assert verified(ScriptedDetector([WEAK_HIT]), verify_min_score=0.3).find("laptop")
    half = RawDetection(Box(10, 10, 50, 25), 0.8, "laptop", "full")  # IoU 0.5
    assert verified(ScriptedDetector([half]), verify_min_iou=0.6).find("laptop") is None
    assert verified(ScriptedDetector([half]), verify_min_iou=0.4).find("laptop")


def test_only_a_limited_number_of_frames_is_checked() -> None:
    for minutes in (5, 20, 40, 60):
        add_box(store_frame(minutes), "laptop")
    detector = ScriptedDetector([], [], [HIT])  # It would pass on the third frame
    assert verified(detector, verify_max_frames=2).find("laptop") is None
    assert len(detector.phrase_calls) == 2
    detector = ScriptedDetector([], [], [HIT])
    assert verified(detector, verify_max_frames=3).find("laptop") is not None


def test_one_frame_is_checked_one_time_with_its_best_box() -> None:
    frame = store_frame(8)
    add_box(frame, "laptop")
    add_box(frame, "laptop", shows="piano")  # A second box of the same frame
    detector = ScriptedDetector([])
    assert verified(detector).find("laptop") is None
    assert len(detector.phrase_calls) == 1


def test_a_wrong_name_alone_never_gives_an_answer() -> None:
    # The name fits exactly, so the classic mode answers. The detector finds nothing, so the verified mode does not.
    add_box(store_frame(8), "laptop")
    assert make_service().find("laptop") is not None
    assert verified(ScriptedDetector([])).find("laptop") is None


def test_the_best_pictures_are_candidates() -> None:
    far, near = store_frame(5), store_frame(50)
    add_box(far, "stuff", shows="piano")  # Newer, but its picture does not fit
    add_box(near, "stuff", shows="umbrella")
    detector = ScriptedDetector([HIT])
    found = verified(detector, verify_top_boxes=1).find("umbrella")
    assert found is not None and found.answer.image_id == near.id
    assert len(detector.phrase_calls) == 1  # The newer frame was no candidate


def test_a_name_that_fits_adds_a_candidate_beyond_the_best_pictures() -> None:
    picture, named = store_frame(5), store_frame(50)
    add_box(picture, "stuff", shows="umbrella")
    add_box(named, "umbrella", shows="piano")  # Its picture does not fit, its name does
    detector = ScriptedDetector([], [HIT])
    found = verified(detector, verify_top_boxes=1).find("umbrella")
    assert found is not None and found.answer.image_id == named.id
    assert found.label == "umbrella" and len(detector.phrase_calls) == 2


def test_a_name_that_does_not_fit_adds_nothing() -> None:
    picture, other = store_frame(5), store_frame(50)
    add_box(picture, "stuff", shows="umbrella")
    add_box(other, "piano", shows="piano")
    detector = ScriptedDetector([HIT])
    found = verified(detector, verify_top_boxes=1).find("umbrella")
    assert found is not None and found.answer.image_id == picture.id
    assert len(detector.phrase_calls) == 1


def test_an_unreadable_frame_is_skipped() -> None:
    old, new = store_frame(50), store_frame(5)
    add_box(old, "laptop")
    add_box(new, "laptop")
    storage.delete_images([new.file_path])
    detector = ScriptedDetector([HIT])
    found = verified(detector).find("laptop")
    assert found is not None and found.answer.image_id == old.id
    assert len(detector.phrase_calls) == 1  # The lost frame never reached the detector


def test_an_error_of_the_detector_skips_the_frame() -> None:
    old, new = store_frame(50), store_frame(5)
    add_box(old, "laptop")
    add_box(new, "laptop")
    detector = ScriptedDetector([HIT], [HIT])
    detector.error = InferenceError("test")  # The first call fails, the second works
    found = verified(detector).find("laptop")
    assert found is not None and found.answer.image_id == old.id


def test_errors_on_every_frame_give_none() -> None:
    add_box(store_frame(5), "laptop")
    detector = ScriptedDetector([HIT])
    detector.error = InferenceError("test")
    assert verified(detector).find("laptop") is None


def test_an_error_of_the_scene_model_gives_none() -> None:
    add_box(store_frame(5), "laptop")
    service = verified(ScriptedDetector([HIT]))
    SCENE.error = InferenceError("test")
    try:
        assert service.find("laptop") is None
    finally:
        SCENE.error = None


def test_a_description_adds_the_names_of_the_language_model_to_the_prompts() -> None:
    add_box(store_frame(8), "laptop")
    language = FakeLanguageModel({PHRASE: ["computer", "laptop"]})
    detector = ScriptedDetector([HIT])
    found = verified(detector, language).find(DESCRIPTION)
    assert found is not None and found.via == "verified"
    assert detector.phrase_calls == [[PHRASE, "computer", "laptop"]]


def test_a_name_is_not_rewritten_in_the_verified_mode() -> None:
    add_box(store_frame(8), "laptop")
    language = FakeLanguageModel({"laptop": ["computer"]})
    detector = ScriptedDetector([HIT])
    assert verified(detector, language).find("laptop") is not None
    assert language.calls == [] and detector.phrase_calls == [["laptop"]]


def test_a_failed_language_model_leaves_the_phrase_alone() -> None:
    add_box(store_frame(8), "laptop")
    language = FakeLanguageModel({PHRASE: ["laptop"]})
    language.error = InferenceError("test")
    detector = ScriptedDetector([HIT])
    found = verified(detector, language).find(DESCRIPTION)
    assert found is not None and detector.phrase_calls == [[PHRASE]]


def test_no_stored_box_gives_none_without_a_detector_run() -> None:
    detector = ScriptedDetector([HIT])
    assert verified(detector).find("laptop") is None
    assert detector.phrase_calls == []


def test_the_log_of_the_verified_mode_never_shows_the_words(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    add_box(store_frame(8), "laptop")
    language = FakeLanguageModel({PHRASE: ["laptop"]})
    with caplog.at_level(logging.DEBUG):
        verified(ScriptedDetector([HIT]), language).find(DESCRIPTION)
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
