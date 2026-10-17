"""Tests of the writer of the descriptions. They use the actual database and storage, and fake ai models.

The language model is a fake that gives the sentence that the test sets. The check of the words is the real code.
"""

import dataclasses
from collections.abc import Iterator

import pytest

import db.api as db
import storage.api as storage
from ai.errors import InferenceError
from backend.describe import DescribeService, description_model
from backend.scene_writer import FUNCTION_WORDS, SceneWriter, check_sentence, words_of
from backend.settings import BackendConfig, load_config
from db.schema import EnrolledObject, Image
from tests.ai_support import FakeLanguageModel, FakeSceneClassifier, make_engine
from tests.browse_support import full_frame, open_vectors
from tests.db_support import skip_if_actual_data_has_problems
from tests.search_support import PREFIX, add_broad_boxes

ALLOWED = {"bottle", "table", "laptop", "wall"}


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[None]:
    storage.init_storage()
    db.init_db()
    skip_if_actual_data_has_problems()
    before: set[str] = set(db.list_image_paths())
    yield
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    made: list[Image] = [i for i in Image.select() if i.file_path not in before]
    if made:
        storage.delete_images(db.delete_images([i.id for i in made]))


# ---- The check of the words ----


@pytest.mark.parametrize(
    "text",
    [
        "The bottle is on a table next to a laptop.",
        "A bottle sits on the table, near the wall.",
        "The bottles are on tables.",  # Plurals of allowed words
        "It is held in a hand next to the bottle.",
    ],
)
def test_a_sentence_with_known_words_passes(text: str) -> None:
    allowed = ALLOWED | {"hand"}
    assert check_sentence(text, allowed, 30) is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("The bottle is on a ledge.", "a word that is not in the facts"),
        ("The black bottle is on a table.", "a word that is not in the facts"),
        ("The bottle was moved 5 minutes ago.", "a digit"),
        ("The bottle is on a table " + "and a table " * 20, "too long"),
        ("???", "too long"),  # No word at all
    ],
)
def test_a_sentence_with_another_word_is_refused_with_a_fixed_reason(
    text: str, reason: str
) -> None:
    assert check_sentence(text, ALLOWED, 30) == reason


def test_the_reason_never_contains_a_word_of_the_sentence() -> None:
    reason = check_sentence("The secret ledge", ALLOWED, 30)
    assert reason is not None and "secret" not in reason and "ledge" not in reason


def test_the_function_words_have_no_claim_about_the_scene() -> None:
    # A noun of a place or a thing must not be a function word.
    for noun in (
        "table",
        "bed",
        "shelf",
        "window",
        "door",
        "person",
        "people",
        "ledge",
        "colour",
    ):
        assert noun not in FUNCTION_WORDS
    assert words_of("It's the Bottle!") == ["it's", "the", "bottle"]


# ---- The writer ----


def make_writer(
    description: str,
    surfaces: list[tuple[str, float]] | None = None,
    **changes: object,
) -> tuple[SceneWriter, FakeLanguageModel, BackendConfig]:
    base: BackendConfig = load_config()
    config = dataclasses.replace(
        base, describe=dataclasses.replace(base.describe, **changes)
    )
    language = FakeLanguageModel()
    language.description = description
    scene = FakeSceneClassifier(
        rooms={"a bedroom": 0.9, "a kitchen": 0.1},
        surfaces=surfaces or [("a table", 0.9), ("the floor", 0.1)],
    )
    engine = make_engine(scene_classifier=scene, language_model=language)
    return (
        SceneWriter(config, engine, DescribeService(config, engine)),
        language,
        config,
    )


def stored(
    frame: Image, obj: EnrolledObject, config: BackendConfig, writer: SceneWriter
) -> str | None:
    model = description_model(config.describe, writer._engine)
    return db.scenes.get_description(frame.id, obj.id, model)


def test_a_sentence_with_known_words_is_stored() -> None:
    frame, obj = full_frame(10, "bottle")
    writer, language, config = make_writer("The bottle is on a table next to a laptop.")
    assert writer.write_next() is True
    assert (
        stored(frame, obj, config, writer)
        == "The bottle is on a table next to a laptop."
    )
    assert len(language.described) == 1


def test_the_model_gets_the_checked_facts_only() -> None:
    frame, obj = full_frame(10, "bottle")
    add_broad_boxes(
        frame,
        [((1, 1, 9, 9), "cork", 0.5), ((2, 2, 12, 12), "shelf", 0.8)],
        open_vectors(2),
        "test-open-model",
    )
    writer, language, _ = make_writer("It is on a table.", room_min_frames=1)
    writer.write_next()
    ((_, facts),) = language.described
    assert facts[0] == f"Object: {obj.name}"
    assert "It is on: a table" in facts and "The room looks like: a bedroom" in facts
    others = next(f for f in facts if f.startswith("Other things"))
    assert "laptop" in others and "shelf" in others  # Confidence 0.9 and 0.8
    assert "cork" not in others  # Confidence 0.5 is below 0.7


def test_a_sentence_with_a_word_that_is_not_in_the_facts_is_refused_and_not_made_again() -> (
    None
):
    frame, obj = full_frame(10, "bottle")
    writer, language, config = make_writer(
        "The bottle is on a table with a cork nearby."
    )
    assert writer.write_next() is True
    assert (
        stored(frame, obj, config, writer) == ""
    )  # Refused. The answer uses its template
    assert writer.write_next() is False  # Nothing to do: the refusal is stored
    assert len(language.described) == 1


def test_an_error_of_the_model_is_not_raised_and_the_frame_is_skipped() -> None:
    frame, obj = full_frame(10, "bottle")
    writer, language, config = make_writer("x")
    language.error = InferenceError("test")
    assert writer.write_next() is False
    language.error = None
    assert writer.write_next() is False  # The frame is skipped until the next start
    assert stored(frame, obj, config, writer) is None


def test_a_missing_file_is_skipped() -> None:
    frame, obj = full_frame(10, "bottle")
    storage.delete_images([frame.file_path])
    writer, _, config = make_writer("The bottle is on a table.")
    assert writer.write_next() is False
    assert stored(frame, obj, config, writer) is None


def test_a_hand_is_a_fact_and_its_words_are_allowed() -> None:
    frame, obj = full_frame(10, "bottle")
    db.sightings.set_last_seen(
        obj.id, frame.id, (20, 15, 60, 45), db.get_image(frame.id).captured_at, 0.7, 0.2, True, True  # type: ignore[union-attr]
    )
    writer, language, config = make_writer("A hand holds the bottle next to a laptop.")
    writer.write_next()
    ((_, facts),) = language.described
    assert "A hand holds the object" in facts and not any(
        f.startswith("It is on") for f in facts
    )
    assert (
        stored(frame, obj, config, writer)
        == "A hand holds the bottle next to a laptop."
    )


def test_a_switched_off_writer_does_nothing() -> None:
    full_frame(10, "bottle")
    writer, language, _ = make_writer("The bottle is on a table.", write=False)
    assert not writer.enabled and writer.write_next() is False
    assert language.described == []
    writer, language, _ = make_writer("The bottle is on a table.")
    writer._engine._language = None
    assert not writer.enabled and writer.write_next() is False


def test_the_log_never_shows_the_sentence(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    full_frame(10, "bottle")
    writer, _, _ = make_writer("The bottle is on a table with a secret ledge.")
    with caplog.at_level(logging.DEBUG):
        writer.write_next()
    assert "secret" not in caplog.text and "ledge" not in caplog.text
    assert "refused (a word that is not in the facts)" in caplog.text
