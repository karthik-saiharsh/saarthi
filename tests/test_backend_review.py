"""Tests of "Is this your object?" (backend/review.py, plan part D2). They use the actual database and storage.

The fake scene model gives every crop of a noise picture the same SigLIP vector ("unknown"), so a stored box with
that vector looks exactly like the reference photos. Every object has a name that starts with `__test__ `.
Each test removes its rows and files.
"""

from collections.abc import Iterator

import numpy as np
import pytest

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from backend.enrollment import EnrollmentService
from backend.errors import InvalidInputError, LimitReachedError
from backend.memory import load_match_objects
from backend.references import ReferenceVectors
from backend.review import ReviewService
from backend.settings import BackendConfig, load_config
from db.schema import EnrolledObject, Image, Region
from tests.ai_support import FakeEmbedder, FakeSceneClassifier, make_engine
from tests.db_support import TEST_PREFIX, skip_if_actual_data_has_problems
from tests.search_support import BOTTLE, PREFIX, make_object, store_frame
from tests.search_support import make_object_with_references as ready_object

SCENE: FakeSceneClassifier = FakeSceneClassifier()


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[list[str]]:
    files: list[str] = []
    before: set[int] = {i.id for i in Image.select(Image.id)}
    yield files
    for obj in db.gallery.list_objects():
        if obj.name.startswith(PREFIX):
            paths = db.gallery.delete_object(obj.id)
            storage.delete_images([p for p in paths if not p.startswith(TEST_PREFIX)])
    new: list[int] = [i.id for i in Image.select(Image.id) if i.id not in before]
    storage.delete_images(
        [p for p in db.delete_images(new) if not p.startswith(TEST_PREFIX)]
    )
    storage.delete_images(files)


def make_review(config: BackendConfig | None = None) -> tuple[ReviewService, AiEngine]:
    engine: AiEngine = make_engine(
        embedder=FakeEmbedder(), scene_classifier=SCENE, matching_mode="keys"
    )
    base: BackendConfig = config or load_config()
    service = ReviewService(base, engine, EnrollmentService(base, engine))
    return service, engine


def keyed_frame(engine: AiEngine, minutes_ago: int, shows: str = "unknown") -> Region:
    """A `done` frame with one region and its crop vectors. `shows` is the phrase that its SigLIP vector fits."""
    frame: Image = store_frame(minutes_ago)
    region: Region = db.get_regions(frame.id)[0]
    key: np.ndarray = np.zeros((1, 384), np.float32)
    key[0, 0] = 1.0  # The DINOv2 key of the fake embedder
    db.crop_vectors.store(
        [region.id],
        SCENE._vector(shows).reshape(1, -1),
        engine.open_vector_model,
        key,
        engine.key_model,
    )
    return region


def test_the_boxes_that_look_like_the_object_are_asked_most_similar_first(
    cleanup: list[str],
) -> None:
    review, engine = make_review()
    obj = ready_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    near = keyed_frame(engine, 5)
    keyed_frame(engine, 10, shows="a laptop")  # Not like the object: not asked
    found = review.candidates(obj.id)
    assert [c.region_id for c in found] == [near.id]
    assert found[0].score == pytest.approx(1.0, abs=1e-5)
    assert review.picture(near.id)[:2] == b"\xff\xd8"  # A JPEG


def test_a_no_makes_a_negative_and_the_box_is_not_asked_again(
    cleanup: list[str],
) -> None:
    review, engine = make_review()
    obj = ready_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    near = keyed_frame(engine, 5)
    review.answer(obj.id, near.id, yes=False)
    assert review.candidates(obj.id) == []
    target = next(
        o
        for o in load_match_objects(
            engine, ReferenceVectors(engine), look_alikes_need_references=True
        )
        if o.object_id == obj.id
    )
    assert target.negative_crops is not None and target.negative_crops.shape[0] == 1
    # A new box that looks the same is refused too: it is as close to the "no" as to the object.
    keyed_frame(engine, 3)
    assert review.candidates(obj.id) == []


def test_a_yes_makes_a_confirmed_reference_photo_with_the_box(
    cleanup: list[str],
) -> None:
    review, engine = make_review()
    obj = ready_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    near = keyed_frame(engine, 5)
    before = len(db.gallery.list_photos(obj.id))
    review.answer(obj.id, near.id, yes=True)
    photos = db.gallery.list_photos(obj.id)
    assert len(photos) == before + 1
    new = photos[-1]
    assert new.status == "confirmed"
    assert (new.x0, new.y0, new.x1, new.y1) == (near.x0, near.y0, near.x1, near.y1)
    assert storage.read_image(new.file_path)  # Its own copy of the frame
    assert review.candidates(obj.id) == []  # Answered


def test_a_box_with_a_strong_sighting_is_not_asked(cleanup: list[str]) -> None:
    review, engine = make_review()
    obj = ready_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    near = keyed_frame(engine, 5)
    db.sightings.add_sighting(
        obj.id, near.image_id, near.id, near.image.captured_at, 0.9, 0.5, True, False
    )
    assert review.candidates(obj.id) == []


def test_an_object_that_is_not_ready_cannot_be_checked() -> None:
    review, _ = make_review()
    draft: EnrolledObject = make_object(ready=False)
    with pytest.raises(InvalidInputError):
        review.candidates(draft.id)


def test_a_yes_is_refused_when_the_object_has_the_most_photos(
    cleanup: list[str],
) -> None:
    import dataclasses

    base = load_config()
    full = dataclasses.replace(
        base, enrollment=dataclasses.replace(base.enrollment, max_photos_per_object=2)
    )
    review, engine = make_review(full)
    obj = ready_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    near = keyed_frame(engine, 5)
    with pytest.raises(LimitReachedError):
        review.answer(obj.id, near.id, yes=True)
    assert review.candidates(obj.id)  # Not answered: it is still asked


# ---- The self-test of the photos (plan part D1) ----


def test_photos_that_all_show_one_place_give_a_warning(cleanup: list[str]) -> None:
    """The fake embedder gives every background the same DINOv2 key, so the photos show one place."""
    from ai.calibration import NOTE_ONE_PLACE

    _, engine = make_review()
    base = load_config()
    obj = ready_object(cleanup, [(60, BOTTLE), (80, BOTTLE), (70, BOTTLE)])
    EnrollmentService(
        base, engine, references=ReferenceVectors(engine)
    ).recalibrate_all()
    note = db.gallery.get_object(obj.id).calibration_note  # type: ignore[union-attr]
    assert note is not None and NOTE_ONE_PLACE in note


def test_without_the_references_there_is_no_self_test(cleanup: list[str]) -> None:
    from ai.calibration import NOTE_ONE_PLACE

    _, engine = make_review()
    obj = ready_object(cleanup, [(60, BOTTLE), (80, BOTTLE), (70, BOTTLE)])
    EnrollmentService(load_config(), engine).recalibrate_all()
    note = db.gallery.get_object(obj.id).calibration_note  # type: ignore[union-attr]
    assert note is None or NOTE_ONE_PLACE not in note
