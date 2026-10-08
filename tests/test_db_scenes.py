"""Tests of the scene cache and the neighbouring frames. They use the actual database.

Every image of a test has a path that starts with `__test__/`. The cleanup removes them.
"""

import math
from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest

import db.api as db
from db.api import scenes
from db.errors import InvalidRecordError
from db.schema import EnrolledObject, FrameScene, Image, ImageStatus
from tests.db_support import (
    TEST_PREFIX,
    make_image,
    set_status,
    skip_if_actual_data_has_problems,
)
from tests.search_support import PREFIX

MODEL: str = "test-scene-model"
NOW: datetime = datetime(2000, 1, 1, 12, 0, 0)
ROOMS: dict[str, float] = {"a bedroom": 0.7, "a corridor": 0.2, "a kitchen": 0.1}


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    yield
    # A last-seen record keeps its frame, so the objects of the test go first.
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    ids: list[int] = [
        i.id for i in Image.select().where(Image.file_path.startswith(TEST_PREFIX))
    ]
    db.delete_images(ids)


def make_frame(seconds: int = 0, status: ImageStatus = ImageStatus.DONE) -> Image:
    """A frame captured `seconds` after NOW."""
    image: Image = make_image()
    set_status(image, status)
    Image.update(captured_at=NOW + timedelta(seconds=seconds)).where(
        Image.id == image.id
    ).execute()
    return db.get_image(image.id)  # type: ignore[return-value]


def ids(images: list[Image]) -> list[int]:
    return [image.id for image in images]


# ---- The scores ----


def test_the_scores_of_a_frame_are_saved_and_read_back() -> None:
    frame: Image = make_frame()
    assert scenes.get_room_scores([frame.id], MODEL) == {}  # Nothing is saved yet
    scenes.set_room_scores(frame.id, MODEL, ROOMS)
    assert scenes.get_room_scores([frame.id], MODEL) == {frame.id: ROOMS}


def test_the_scores_of_another_model_are_not_used() -> None:
    frame: Image = make_frame()
    scenes.set_room_scores(frame.id, MODEL, ROOMS)
    assert scenes.get_room_scores([frame.id], "another-model") == {}


def test_new_scores_replace_the_old_ones_and_the_model() -> None:
    frame: Image = make_frame()
    scenes.set_room_scores(frame.id, MODEL, ROOMS)
    other: dict[str, float] = {"a kitchen": 0.9, "a bedroom": 0.1}
    scenes.set_room_scores(frame.id, "new-model", other)
    assert scenes.get_room_scores([frame.id], MODEL) == {}
    assert scenes.get_room_scores([frame.id], "new-model") == {frame.id: other}
    assert FrameScene.select().where(FrameScene.image == frame.id).count() == 1


def test_the_scores_of_several_frames_are_read_together() -> None:
    first, second, third = make_frame(0), make_frame(1), make_frame(2)
    scenes.set_room_scores(first.id, MODEL, ROOMS)
    scenes.set_room_scores(third.id, MODEL, {"a kitchen": 1.0})
    found = scenes.get_room_scores([first.id, second.id, third.id], MODEL)
    assert set(found) == {first.id, third.id}  # The second frame has no scores
    assert scenes.get_room_scores([], MODEL) == {}


def test_the_scores_go_with_the_frame() -> None:
    frame: Image = make_frame()
    scenes.set_room_scores(frame.id, MODEL, ROOMS)
    db.delete_images([frame.id])
    assert not FrameScene.select().where(FrameScene.image == frame.id).exists()


def test_scores_of_a_frame_that_does_not_exist_are_refused() -> None:
    with pytest.raises(db.InvalidRecordError):
        scenes.set_room_scores(2**31 - 1, MODEL, ROOMS)


@pytest.mark.parametrize(
    "bad",
    [
        {"a bedroom": math.nan, "a kitchen": 1.0},
        {"a bedroom": math.inf},
        {"a bedroom": -0.5, "a kitchen": 1.5},
        {"a bedroom": 0.4, "a kitchen": 0.4},  # Adds up to 0.8
        {"a bedroom": 0.9, "a kitchen": 0.9},  # Adds up to 1.8
    ],
)
def test_scores_that_are_not_probabilities_are_refused(bad: dict[str, float]) -> None:
    frame: Image = make_frame()
    with pytest.raises(db.InvalidRecordError):
        scenes.set_room_scores(frame.id, MODEL, bad)
    assert scenes.get_room_scores([frame.id], MODEL) == {}


def test_a_small_rounding_error_in_the_sum_is_allowed() -> None:
    frame: Image = make_frame()
    scenes.set_room_scores(frame.id, MODEL, {"a bedroom": 0.6, "a kitchen": 0.3995})
    assert scenes.get_room_scores([frame.id], MODEL)


# ---- The neighbouring frames ----


def test_the_neighbours_are_the_frames_inside_the_window_nearest_first() -> None:
    centre = make_frame(0)
    far, near, mid = make_frame(100), make_frame(10), make_frame(-60)
    result = scenes.list_neighbour_images(centre.id, 120, 10)
    assert ids(result) == [near.id, mid.id, far.id]


def test_the_window_includes_its_ends_and_excludes_the_frame_itself() -> None:
    centre = make_frame(0)
    edge_after, edge_before = make_frame(120), make_frame(-120)
    outside = make_frame(121)
    result = ids(scenes.list_neighbour_images(centre.id, 120, 10))
    assert set(result) == {edge_after.id, edge_before.id}
    assert centre.id not in result and outside.id not in result


def test_only_done_frames_are_neighbours() -> None:
    centre = make_frame(0)
    done = make_frame(5)
    for status in (ImageStatus.PENDING, ImageStatus.REJECTED, ImageStatus.FAILED):
        make_frame(3, status)
    assert ids(scenes.list_neighbour_images(centre.id, 60, 10)) == [done.id]


def test_the_limit_keeps_the_nearest_frames() -> None:
    centre = make_frame(0)
    frames = [make_frame(s) for s in (30, -20, 10, -40)]
    result = scenes.list_neighbour_images(centre.id, 60, 2)
    assert ids(result) == [frames[2].id, frames[1].id]  # 10 s and 20 s away


def test_frames_at_the_same_distance_come_in_the_order_of_their_ids() -> None:
    centre = make_frame(0)
    after, before = make_frame(10), make_frame(-10)
    assert ids(scenes.list_neighbour_images(centre.id, 60, 10)) == [
        after.id,
        before.id,
    ]


def test_a_frame_without_neighbours_and_an_unknown_frame_give_an_empty_list() -> None:
    alone = make_frame(0)
    assert scenes.list_neighbour_images(alone.id, 60, 10) == []
    assert scenes.list_neighbour_images(2**31 - 1, 60, 10) == []


def test_a_wrong_window_or_limit_is_refused() -> None:
    frame = make_frame(0)
    with pytest.raises(AssertionError):
        scenes.list_neighbour_images(frame.id, 0, 10)
    with pytest.raises(AssertionError):
        scenes.list_neighbour_images(frame.id, 60, 0)


# ---- The descriptions ----

DESCRIPTION_MODEL: str = "test-language/abc"


def make_last_seen(seconds: int = 0, name: str = "bottle"):
    """A `done` frame that is the last-seen record of a new object. Return (frame, object)."""
    from tests.search_support import make_object

    obj = make_object(name=name)
    frame: Image = make_frame(seconds)
    db.sightings.set_last_seen(
        obj.id, frame.id, (1, 1, 5, 5), frame.captured_at, 0.7, 0.2, True, False
    )
    return frame, obj


def test_a_description_is_saved_and_read_back() -> None:
    frame, obj = make_last_seen()
    assert scenes.get_description(frame.id, obj.id, DESCRIPTION_MODEL) is None
    scenes.set_description(
        frame.id, obj.id, DESCRIPTION_MODEL, "It is on a table.", ["Object: bottle"]
    )
    assert (
        scenes.get_description(frame.id, obj.id, DESCRIPTION_MODEL)
        == "It is on a table."
    )
    assert scenes.get_description(frame.id, obj.id, "another-model") is None


def test_a_refused_description_is_an_empty_text_and_a_new_one_replaces_the_old() -> (
    None
):
    frame, obj = make_last_seen()
    scenes.set_description(frame.id, obj.id, DESCRIPTION_MODEL, "", [])
    assert scenes.get_description(frame.id, obj.id, DESCRIPTION_MODEL) == ""
    scenes.set_description(frame.id, obj.id, DESCRIPTION_MODEL, "New text.", ["x"])
    assert scenes.get_description(frame.id, obj.id, DESCRIPTION_MODEL) == "New text."


def test_two_objects_in_one_frame_have_two_descriptions() -> None:
    from tests.search_support import make_object

    frame, first = make_last_seen()
    second = make_object(name="keys")
    db.sightings.set_last_seen(
        second.id, frame.id, (2, 2, 6, 6), frame.captured_at, 0.7, 0.2, True, False
    )
    scenes.set_description(frame.id, first.id, DESCRIPTION_MODEL, "One.", [])
    scenes.set_description(frame.id, second.id, DESCRIPTION_MODEL, "Two.", [])
    assert scenes.get_description(frame.id, first.id, DESCRIPTION_MODEL) == "One."
    assert scenes.get_description(frame.id, second.id, DESCRIPTION_MODEL) == "Two."


def test_the_frames_to_describe_are_the_last_seen_frames_without_a_description_of_the_model() -> (
    None
):
    frame, obj = make_last_seen()
    todo = [
        t
        for t in scenes.list_frames_to_describe(DESCRIPTION_MODEL, 100)
        if t.object_id == obj.id
    ]
    assert [(t.image_id, t.object_name, t.box, t.in_hand) for t in todo] == [
        (frame.id, obj.name, (1, 1, 5, 5), False)
    ]
    scenes.set_description(frame.id, obj.id, DESCRIPTION_MODEL, "Text.", [])
    assert not [
        t
        for t in scenes.list_frames_to_describe(DESCRIPTION_MODEL, 100)
        if t.object_id == obj.id
    ]
    # A description of another model does not count
    assert [
        t
        for t in scenes.list_frames_to_describe("newer-model", 100)
        if t.object_id == obj.id
    ]


def test_a_frame_that_is_not_done_is_not_described() -> None:
    frame, obj = make_last_seen()
    Image.update(status=ImageStatus.REJECTED.value).where(
        Image.id == frame.id
    ).execute()
    assert not [
        t
        for t in scenes.list_frames_to_describe(DESCRIPTION_MODEL, 100)
        if t.object_id == obj.id
    ]
    Image.update(status=ImageStatus.DONE.value).where(Image.id == frame.id).execute()


def test_a_description_goes_with_its_frame_and_its_object() -> None:
    from db.schema import FrameDescription

    frame, obj = make_last_seen()
    scenes.set_description(frame.id, obj.id, DESCRIPTION_MODEL, "Text.", [])
    EnrolledObject.delete().where(
        EnrolledObject.id == obj.id
    ).execute()  # The last-seen record goes with it
    assert (
        FrameDescription.select().where(FrameDescription.image == frame.id).count() == 0
    )


def test_the_description_values_are_checked() -> None:
    frame, obj = make_last_seen()
    with pytest.raises(AssertionError):
        scenes.set_description(frame.id, obj.id, DESCRIPTION_MODEL, "x" * 601, [])
    with pytest.raises(InvalidRecordError):
        scenes.set_description(frame.id, 2_000_000_000, DESCRIPTION_MODEL, "x", [])
