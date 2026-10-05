"""Tests of the scene classifier in the engine (stage Q2). The model is a fake."""

import dataclasses

import pytest
from PIL import Image

from ai.api import AiEngine
from ai.errors import InferenceError
from ai.scene import context_box
from ai.settings import load_config
from ai.types import Box
from tests.ai_support import (
    FakeDetector,
    FakeEmbedder,
    FakeQueryEncoder,
    FakeSceneClassifier,
    FakeSpeechRecognizer,
    FakeVerifier,
    make_engine,
)

SCENE = load_config().scene


def picture(width: int = 400, height: int = 300) -> Image.Image:
    return Image.new("RGB", (width, height), (120, 120, 120))


def test_the_context_has_space_on_each_side_and_below_and_half_of_it_above() -> None:
    box: Box = Box(100, 100, 200, 160)  # 100 x 60
    around: Box = context_box(box, 1000, 1000, 1.0)
    assert (around.x0, around.x1) == (0, 300)  # 100 on each side
    assert (around.y0, around.y1) == (70, 220)  # 30 above, 60 below


def test_the_context_is_cut_at_the_edge_of_the_picture() -> None:
    corner: Box = context_box(Box(0, 0, 100, 100), 150, 150, 1.0)
    assert (corner.x0, corner.y0, corner.x1, corner.y1) == (0, 0, 150, 150)
    far: Box = context_box(Box(300, 200, 400, 300), 400, 300, 1.0)
    assert (far.x1, far.y1) == (400, 300) and (far.x0, far.y0) == (200, 150)


def test_a_context_of_zero_gives_the_box_itself() -> None:
    box: Box = Box(10, 20, 60, 90)
    assert context_box(box, 400, 300, 0.0) == box


def test_the_rooms_are_the_scores_of_the_classifier_for_the_whole_picture() -> None:
    rooms: dict[str, float] = {room: 0.0 for room in SCENE.rooms}
    rooms["a bedroom"] = 1.0
    scene: FakeSceneClassifier = FakeSceneClassifier(rooms=rooms)
    engine: AiEngine = make_engine(scene_classifier=scene)
    assert engine.score_rooms(picture(400, 300)) == rooms
    assert scene.room_calls == [(400, 300)]


def test_the_surfaces_come_best_first_from_the_box_with_space_around_it() -> None:
    scene: FakeSceneClassifier = FakeSceneClassifier(
        surfaces=[("a table", 0.2), ("the floor", 0.7), ("a bed", 0.1)]
    )
    engine: AiEngine = make_engine(scene_classifier=scene)
    box: Box = Box(100, 100, 200, 160)
    surfaces = engine.score_surfaces(picture(1000, 1000), box)
    assert [name for name, _ in surfaces] == ["the floor", "a table", "a bed"]
    around: Box = context_box(box, 1000, 1000, SCENE.surface_context)
    assert scene.surface_calls == [(around.width, around.height)]


def test_the_name_for_the_cache_has_the_model_and_a_fingerprint_of_the_words() -> None:
    name: str = make_engine().scene_model
    assert name.startswith("fake-scene@00000000/") and len(name.split("/")[1]) == 8


def test_the_name_for_the_cache_changes_when_a_word_list_or_the_prompt_changes() -> (
    None
):
    base = load_config()

    def engine_with(**changes: object) -> AiEngine:
        config = dataclasses.replace(
            base, scene=dataclasses.replace(base.scene, **changes)
        )
        return AiEngine(
            config,
            detector=FakeDetector(),  # type: ignore[arg-type]
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            query_encoder=FakeQueryEncoder(),  # type: ignore[arg-type]
            verifier=FakeVerifier(),  # type: ignore[arg-type]
            scene_classifier=FakeSceneClassifier(),  # type: ignore[arg-type]
            speech_recognizer=FakeSpeechRecognizer(),  # type: ignore[arg-type]
        )

    same = engine_with().scene_model
    assert same == make_engine().scene_model
    assert engine_with(rooms=(*base.scene.rooms, "a garage")).scene_model != same
    assert engine_with(surfaces=base.scene.surfaces[:-1]).scene_model != same
    assert engine_with(prompt="a picture of {label}").scene_model != same


def test_an_error_of_the_model_reaches_the_caller() -> None:
    scene: FakeSceneClassifier = FakeSceneClassifier()
    scene.error = InferenceError("test")
    engine: AiEngine = make_engine(scene_classifier=scene)
    with pytest.raises(InferenceError):
        engine.score_rooms(picture())
    with pytest.raises(InferenceError):
        engine.score_surfaces(picture(), Box(10, 10, 50, 50))


def test_a_switched_off_scene_loads_no_model_and_refuses_calls() -> None:
    base = load_config()
    config = dataclasses.replace(
        base, scene=dataclasses.replace(base.scene, enabled=False)
    )
    engine: AiEngine = AiEngine(
        config,
        detector=FakeDetector(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        query_encoder=FakeQueryEncoder(),  # type: ignore[arg-type]
        verifier=FakeVerifier(),  # type: ignore[arg-type]
    )  # The real model would be loaded here if the switch did not work
    assert not engine.scene_enabled
    with pytest.raises(AssertionError):
        engine.score_rooms(picture())
    with pytest.raises(AssertionError):
        engine.score_surfaces(picture(), Box(10, 10, 50, 50))
    with pytest.raises(AssertionError):
        _ = engine.scene_model
    assert make_engine().scene_enabled
