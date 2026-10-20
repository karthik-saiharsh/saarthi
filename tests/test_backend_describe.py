"""Tests of the description of the place of an object. They use the actual database and storage.

The matching and the scene model are fakes. The frames are real files. Every frame of a test is removed at the end.
"""

import dataclasses
import itertools
import logging
from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest

import db.api as db
import storage.api as storage
from ai.errors import InferenceError
from ai.settings import load_config
from backend.describe import (
    DescribeService,
    RoomChoice,
    build_sentence,
    choose_room,
    day_phrase,
    description_model,
    part_of_day,
    time_phrase,
)
from backend.search import Answer, AnswerSource
from backend.settings import BackendConfig, DescribeConfig
from backend.settings import load_config as backend_config
from db.schema import Image
from tests.ai_support import FakeLanguageModel, FakeSceneClassifier, make_engine
from tests.db_support import skip_if_actual_data_has_problems
from tests.search_support import BOX, store_frame

ANCHOR: datetime = datetime(2000, 1, 1, 12, 0, 0)
ROOMS: tuple[str, ...] = load_config().scene.rooms
CONFIG: DescribeConfig = backend_config().describe


def rooms(top: str, score: float = 0.7) -> dict[str, float]:
    """Room scores: `top` has `score`, the others share the rest."""
    rest: float = (1.0 - score) / (len(ROOMS) - 1)
    return {room: score if room == top else rest for room in ROOMS}


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    before: set[str] = set(db.list_image_paths())
    yield
    # The objects go first: a last-seen record keeps its frame. Their reference photos have no file.
    from db.schema import EnrolledObject
    from tests.search_support import PREFIX

    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))
    database.close()


_seeds = itertools.count(1)


def frame_at(offset_seconds: int) -> Image:
    """A real frame, captured `offset_seconds` after the anchor. Each frame has other pixels."""
    frame: Image = store_frame(next(_seeds))
    Image.update(captured_at=ANCHOR + timedelta(seconds=offset_seconds)).where(
        Image.id == frame.id
    ).execute()
    return db.get_image(frame.id)  # type: ignore[return-value]


def answer_for(frame: Image | None, **changes: object) -> Answer:
    values: dict[str, object] = {
        "object_id": 1,
        "found": True,
        "source": AnswerSource.ARCHIVE,
        "seen_at": ANCHOR,
        "age_seconds": 600,
        "image_id": None if frame is None else frame.id,
        "box": BOX,
        "similarity": 0.9,
        "strong": True,
        "in_hand": False,
        "stale": False,
        "complete": True,
    }
    values.update(changes)
    return Answer(**values)  # type: ignore[arg-type]


def make_service(
    scene: FakeSceneClassifier | None = None, **changes: object
) -> tuple[DescribeService, FakeSceneClassifier]:
    base: BackendConfig = backend_config()
    config = dataclasses.replace(
        base, describe=dataclasses.replace(base.describe, **changes)
    )
    fake: FakeSceneClassifier = scene or FakeSceneClassifier(
        rooms=rooms("a bedroom"), surfaces=[("a table", 0.7), ("the floor", 0.3)]
    )
    return DescribeService(config, make_engine(scene_classifier=fake)), fake


# ---- The time ----


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "just now"),
        (59, "just now"),
        (60, "1 minute ago"),
        (12 * 60, "12 minutes ago"),
        (3600, "1 hour ago"),
        (3 * 3600, "3 hours ago"),
        (86400, "1 day ago"),
        (2 * 86400 + 5, "2 days ago"),
    ],
)
def test_the_time_is_plain(seconds: int, text: str) -> None:
    assert time_phrase(seconds) == text


# ---- The room rule ----


def test_the_room_is_named_when_enough_frames_agree_and_the_score_is_high() -> None:
    frames = [rooms("a bedroom"), rooms("a bedroom"), rooms("a kitchen")]
    choice: RoomChoice = choose_room(frames, CONFIG)
    assert choice.room == "a bedroom" and choice.agreeing == 2 and choice.frames == 3


def test_the_rule_says_nothing_when_the_frames_disagree() -> None:
    frames = [rooms("a bedroom"), rooms("a kitchen"), rooms("a bathroom")]
    assert choose_room(frames, CONFIG).room is None


def test_the_rule_says_nothing_with_too_few_frames() -> None:
    frames = [rooms("a bedroom")] * (CONFIG.room_min_frames - 1)
    assert choose_room(frames, CONFIG).room is None
    assert choose_room([], CONFIG) == RoomChoice(None, 0.0, 0, 0)


def test_the_rule_says_nothing_when_the_average_score_is_low() -> None:
    low = [
        rooms("a bedroom", 0.2)
    ] * 3  # The best room is the same, but the score is low
    choice = choose_room(low, CONFIG)
    assert choice.room is None and choice.agreeing == 3


def test_one_wrong_frame_with_a_high_score_does_not_decide() -> None:
    # A close-up of a floor: one frame is sure of "a bathroom". The others see a corridor.
    frames = [
        rooms("a bathroom", 0.93),
        rooms("a corridor", 0.6),
        rooms("a corridor", 0.6),
    ]
    assert choose_room(frames, CONFIG).room == "a corridor"


def test_the_limits_come_from_the_settings() -> None:
    strict = dataclasses.replace(CONFIG, room_agree_fraction=1.0)
    frames = [rooms("a bedroom"), rooms("a bedroom"), rooms("a kitchen")]
    assert choose_room(frames, strict).room is None
    assert choose_room(frames, CONFIG).room == "a bedroom"


# ---- The sentence ----


def sure(**changes: object) -> Answer:
    return answer_for(None, image_id=1, **changes)


def test_a_sure_answer_names_the_time_the_surface_and_the_room() -> None:
    text = build_sentence("bottle", sure(), "a table", "a bedroom")
    assert text == (
        "Your bottle was last seen 10 minutes ago. It was on a table. The room looks like a bedroom."
    )


def test_the_surface_and_the_room_are_left_out_when_they_are_none() -> None:
    assert (
        build_sentence("bottle", sure(), None, None)
        == "Your bottle was last seen 10 minutes ago."
    )


def test_an_object_in_the_hand_does_not_name_a_surface() -> None:
    text = build_sentence("bottle", sure(in_hand=True), "a table", "a bedroom")
    assert (
        text
        == "You were holding your bottle 10 minutes ago. The room looks like a bedroom."
    )


def test_a_weak_answer_says_that_the_system_is_not_sure() -> None:
    text = build_sentence("bottle", sure(strong=False), "a table", "a bedroom")
    assert text == (
        "I am not sure, but I may have seen your bottle 10 minutes ago. It may have been on a table. "
        "The room may look like a bedroom. Please look at the picture."
    )


def test_a_weak_answer_in_the_hand_is_also_careful() -> None:
    text = build_sentence("bottle", sure(strong=False, in_hand=True), None, None)
    assert text == (
        "I am not sure, but you may have been holding your bottle 10 minutes ago. Please look at the picture."
    )


def test_an_old_sighting_says_that_it_was_a_long_time_ago() -> None:
    text = build_sentence(
        "bottle", sure(stale=True, age_seconds=2 * 86400), None, "a kitchen"
    )
    assert text == (
        "The last time that I saw your bottle was 2 days ago. That was a long time ago. "
        "The room looks like a kitchen."
    )


def test_an_object_that_was_not_seen() -> None:
    gone = answer_for(None, found=False, age_seconds=None, image_id=None, box=None)
    assert (
        build_sentence("bottle", gone, None, None)
        == "I have not seen your bottle lately."
    )
    partial = dataclasses.replace(gone, complete=False)
    assert build_sentence("bottle", partial, None, None).endswith(
        "I could not look at all the pictures."
    )


def test_the_name_of_the_object_must_not_be_empty() -> None:
    with pytest.raises(AssertionError):
        build_sentence("  ", sure(), None, None)


# ---- The service ----


def test_the_sentence_has_the_surface_and_the_room_from_the_frames_around() -> None:
    centre, before, after = frame_at(0), frame_at(-30), frame_at(40)
    service, scene = make_service()
    result = service.describe("bottle", answer_for(centre))
    assert result.surface == "a table" and result.room == "a bedroom"
    assert result.text == (
        "Your bottle was last seen 10 minutes ago. It was on a table. The room looks like a bedroom."
    )
    assert len(scene.room_calls) == 3 and len(scene.surface_calls) == 1
    assert before.id != after.id


def test_the_room_scores_are_saved_and_a_frame_is_scored_one_time() -> None:
    centre, before, after = frame_at(0), frame_at(-30), frame_at(40)
    service, scene = make_service()
    service.describe("bottle", answer_for(centre))
    model: str = make_engine().scene_model
    saved = db.scenes.get_room_scores([centre.id, before.id, after.id], model)
    assert set(saved) == {centre.id, before.id, after.id}
    service.describe("bottle", answer_for(centre))
    assert len(scene.room_calls) == 3  # No new call: the scores come from the cache


def test_scores_of_another_model_are_not_used() -> None:
    centre, before, after = frame_at(0), frame_at(-30), frame_at(40)
    for frame in (centre, before, after):
        db.scenes.set_room_scores(frame.id, "old-model/00000000", rooms("a kitchen"))
    service, scene = make_service()
    assert service.describe("bottle", answer_for(centre)).room == "a bedroom"
    assert len(scene.room_calls) == 3


def test_saved_scores_of_the_neighbours_are_used_without_a_new_call() -> None:
    centre, before, after = frame_at(0), frame_at(-30), frame_at(40)
    model: str = make_engine().scene_model
    for frame in (before, after):
        db.scenes.set_room_scores(frame.id, model, rooms("a kitchen"))
    service, scene = make_service(
        scene=FakeSceneClassifier(rooms=rooms("a kitchen"), surfaces=[("a table", 0.9)])
    )
    assert service.describe("bottle", answer_for(centre)).room == "a kitchen"
    assert len(scene.room_calls) == 1  # Only the frame of the answer


def test_no_room_is_named_when_the_frames_disagree() -> None:
    centre = frame_at(0)
    frame_at(-30), frame_at(40)
    service, scene = make_service()
    scene.room_queue = [rooms("a bedroom"), rooms("a kitchen"), rooms("a bathroom")]
    result = service.describe("bottle", answer_for(centre))
    assert result.room is None and "room" not in result.text
    assert result.surface == "a table"


def test_no_room_is_named_without_neighbours() -> None:
    centre = frame_at(0)
    result = make_service()[0].describe("bottle", answer_for(centre))
    assert result.room is None and result.surface == "a table"


def test_frames_outside_the_window_are_not_used() -> None:
    centre = frame_at(0)
    frame_at(-500), frame_at(900)
    service, scene = make_service(neighbour_window_seconds=120)
    assert service.describe("bottle", answer_for(centre)).room is None
    assert len(scene.room_calls) == 1


def test_the_limit_of_neighbours_is_used() -> None:
    centre = frame_at(0)
    for offset in (10, 20, 30, 40, 50):
        frame_at(offset)
    service, scene = make_service(neighbour_max_frames=2)
    service.describe("bottle", answer_for(centre))
    assert len(scene.room_calls) == 3  # The frame of the answer and 2 neighbours


def test_an_object_in_the_hand_has_no_surface() -> None:
    centre, _, _ = frame_at(0), frame_at(-30), frame_at(40)
    service, scene = make_service()
    result = service.describe("bottle", answer_for(centre, in_hand=True))
    assert result.surface is None and scene.surface_calls == []
    assert result.room == "a bedroom"


def test_a_surface_with_a_low_score_is_not_named() -> None:
    centre = frame_at(0)
    scene = FakeSceneClassifier(
        rooms=rooms("a bedroom"), surfaces=[("a table", 0.2), ("the floor", 0.15)]
    )
    result = make_service(scene)[0].describe("bottle", answer_for(centre))
    assert result.surface is None and "It was on" not in result.text


def test_a_switched_off_scene_gives_a_sentence_with_the_time_only() -> None:
    from ai.api import AiEngine
    from tests.ai_support import (
        FakeDetector,
        FakeEmbedder,
        FakeOpenDetector,
        FakeQueryEncoder,
        FakeSpeechRecognizer,
        FakeSpeechSynthesizer,
        FakeVerifier,
    )

    base = load_config()
    ai_config = dataclasses.replace(
        base, scene=dataclasses.replace(base.scene, enabled=False)
    )
    engine = AiEngine(
        ai_config,
        detector=FakeDetector(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        query_encoder=FakeQueryEncoder(),  # type: ignore[arg-type]
        verifier=FakeVerifier(),  # type: ignore[arg-type]
        speech_recognizer=FakeSpeechRecognizer(),
        speech_synthesizer=FakeSpeechSynthesizer(),  # type: ignore[arg-type]
        open_detector=FakeOpenDetector(),  # type: ignore[arg-type]
        language_model=FakeLanguageModel(),  # type: ignore[arg-type]
    )
    service = DescribeService(backend_config(), engine)
    result = service.describe("bottle", answer_for(frame_at(0)))
    assert result.text == "Your bottle was last seen 10 minutes ago."
    assert result.surface is None and result.room is None


def test_an_error_of_the_scene_model_gives_a_sentence_without_the_place() -> None:
    centre = frame_at(0)
    scene = FakeSceneClassifier()
    scene.error = InferenceError("test")
    result = make_service(scene)[0].describe("bottle", answer_for(centre))
    assert result.text == "Your bottle was last seen 10 minutes ago."


def test_a_neighbour_that_fails_is_left_out_and_the_others_still_count() -> None:
    centre, bad, _ = frame_at(0), frame_at(-30), frame_at(40)
    storage.delete_images([bad.file_path])  # Its file is gone
    service, scene = make_service(room_min_frames=2)
    assert service.describe("bottle", answer_for(centre)).room == "a bedroom"
    assert len(scene.room_calls) == 2


def test_a_frame_of_the_answer_that_cannot_be_read_gives_the_time_only() -> None:
    centre = frame_at(0)
    storage.delete_images([centre.file_path])
    result = make_service()[0].describe("bottle", answer_for(centre))
    assert result.text == "Your bottle was last seen 10 minutes ago."


def test_an_unknown_frame_and_an_answer_without_frame_give_the_time_only() -> None:
    service, scene = make_service()
    assert service.describe("bottle", answer_for(None, image_id=2**31 - 1)).room is None
    assert service.describe("bottle", answer_for(None)).surface is None
    assert scene.room_calls == []


def test_an_object_that_was_not_seen_has_no_place() -> None:
    gone = answer_for(None, found=False, age_seconds=None, image_id=None, box=None)
    service, scene = make_service()
    assert (
        service.describe("bottle", gone).text == "I have not seen your bottle lately."
    )
    assert scene.room_calls == []


def test_the_log_shows_no_name_and_no_word_of_the_lists(
    caplog: pytest.LogCaptureFixture,
) -> None:
    centre = frame_at(0)
    frame_at(-30), frame_at(40)
    service, _ = make_service()
    with caplog.at_level(logging.DEBUG):
        service.describe("secret-name", answer_for(centre))
    own = "\n".join(
        r.getMessage()
        for r in caplog.records
        if r.name.startswith(("backend", "ai", "db", "storage"))
    )
    assert "secret-name" not in own and "bedroom" not in own and "table" not in own


# ---- The part of the day ----

KOLKATA = dataclasses.replace(CONFIG, time_zone="Asia/Kolkata")


@pytest.mark.parametrize(
    ("hour", "part"),
    [
        (0, "night"),
        (4, "night"),
        (5, "morning"),
        (11, "morning"),
        (12, "afternoon"),
        (16, "afternoon"),
        (17, "evening"),
        (21, "evening"),
        (22, "night"),
        (23, "night"),
    ],
)
def test_part_of_day(hour: int, part: str) -> None:
    assert part_of_day(hour) == part


def utc(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute)


def test_the_day_is_the_day_in_the_time_zone() -> None:
    # 20:00 UTC is 01:30 the next day in Kolkata. The search runs at 05:00 UTC (10:30 Kolkata) on the 10th.
    seen = utc(9, 20)
    age = int((utc(10, 5) - seen).total_seconds())
    assert day_phrase(seen, age, "Asia/Kolkata") == "during the night"
    assert day_phrase(seen, age, "UTC") == "yesterday evening"


def test_older_than_yesterday_has_no_part_of_the_day() -> None:
    seen = utc(5, 10)
    assert day_phrase(seen, int((utc(9, 10) - seen).total_seconds()), "UTC") is None


def test_a_young_sighting_keeps_the_relative_time() -> None:
    seen = utc(9, 20)
    assert time_phrase(3 * 3600, seen, KOLKATA) == "3 hours ago"


def test_an_old_sighting_of_today_or_yesterday_gets_the_part_of_the_day() -> None:
    seen = utc(9, 3, 30)  # 09:00 in Kolkata
    assert time_phrase(7 * 3600, seen, KOLKATA) == "this morning"
    assert time_phrase(30 * 3600, seen, KOLKATA) == "yesterday morning"
    assert time_phrase(5 * 86400, seen, KOLKATA) == "5 days ago"


def test_the_sentence_uses_the_part_of_the_day() -> None:
    answer = dataclasses.replace(sure(), age_seconds=8 * 3600, seen_at=utc(9, 3, 30))
    assert "was last seen this morning." in build_sentence(
        "bottle", answer, None, None, KOLKATA
    )
    stale = dataclasses.replace(answer, stale=True)
    assert "was this morning. That was a long time ago." in build_sentence(
        "bottle", stale, None, None, KOLKATA
    )


def test_without_a_config_the_sentence_keeps_the_relative_time() -> None:
    answer = dataclasses.replace(sure(), age_seconds=8 * 3600, seen_at=utc(9, 3, 30))
    assert "8 hours ago" in build_sentence("bottle", answer, None, None)


def test_the_server_time_zone_is_used_when_the_name_is_empty() -> None:
    phrase = day_phrase(utc(9, 12), 8 * 3600, "")
    assert phrase is not None and (
        "this" in phrase or "night" in phrase or "yesterday" in phrase
    )


# ---- The sentence of the language model ----


def stored_description(
    frame: Image, text: str, **changes: object
) -> tuple[DescribeService, FakeSceneClassifier]:
    """A service, and a sentence in the database for the object 1 of `answer_for`. Return the service."""
    from db.schema import EnrolledObject
    from tests.search_support import make_object

    obj: EnrolledObject = make_object()
    service, scene = make_service(**changes)
    model = description_model(service._config, service._engine)
    db.scenes.set_description(frame.id, obj.id, model, text, ["Object: x"])
    service._test_object_id = obj.id  # type: ignore[attr-defined]
    return service, scene


def test_a_checked_sentence_takes_the_place_of_the_surface_and_the_room() -> None:
    frame = frame_at(0)
    service, scene = stored_description(
        frame, "The bottle is on a table next to a laptop"
    )
    answer = answer_for(frame, object_id=service._test_object_id)  # type: ignore[attr-defined]
    result = service.describe("bottle", answer)
    assert result.text == (
        "Your bottle was last seen 10 minutes ago. The bottle is on a table next to a laptop."
    )
    assert (
        scene.room_calls == [] and scene.surface_calls == []
    )  # No scene model run is needed


def test_a_refused_sentence_gives_the_template() -> None:
    frame = frame_at(0)
    service, _ = stored_description(frame, "")  # Refused
    answer = answer_for(frame, object_id=service._test_object_id)  # type: ignore[attr-defined]
    assert "It was on a table." in service.describe("bottle", answer).text


def test_a_weak_answer_does_not_use_the_sentence() -> None:
    frame = frame_at(0)
    service, _ = stored_description(frame, "The bottle is on a table")
    answer = answer_for(frame, object_id=service._test_object_id, strong=False)  # type: ignore[attr-defined]
    text = service.describe("bottle", answer).text
    assert "I am not sure" in text and "The bottle is on a table." not in text


def test_an_answer_of_an_open_object_does_not_use_a_sentence() -> None:
    frame = frame_at(0)
    service, _ = stored_description(frame, "The bottle is on a table")
    answer = answer_for(frame, object_id=0, strong=True)
    assert "The bottle is on a table." not in service.describe("bottle", answer).text


def test_a_sentence_of_another_instruction_or_check_is_not_used() -> None:
    frame = frame_at(0)
    service, _ = stored_description(frame, "The bottle is on a table")
    changed = dataclasses.replace(
        service._config, max_words=29
    )  # A change of the check changes the name
    service._config = changed
    answer = answer_for(frame, object_id=service._test_object_id)  # type: ignore[attr-defined]
    assert "It was on a table." in service.describe("bottle", answer).text


def test_the_time_of_the_answer_stays_in_the_sentence() -> None:
    frame = frame_at(0)
    service, _ = stored_description(frame, "The bottle is on a table.")
    answer = answer_for(frame, object_id=service._test_object_id, age_seconds=3 * 3600)  # type: ignore[attr-defined]
    assert "3 hours ago" in service.describe("bottle", answer).text


def test_a_newer_weak_sighting_is_said_as_a_second_place_with_i_am_not_sure() -> None:
    from backend.search import NewerMaybe

    answer = dataclasses.replace(
        sure(),
        newer_maybe=NewerMaybe(datetime(2000, 1, 1), 5 * 60, 2, (1, 1, 5, 5), False),
    )
    text = build_sentence("bottle", answer, "a table", None)
    assert text.startswith(
        "Your bottle was last seen 10 minutes ago. It was on a table."
    )
    assert text.endswith(
        "I am not sure, but I may have seen it again 5 minutes ago, in another place. "
        "Please look at the second picture."
    )
