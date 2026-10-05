"""Tests of the rule that changes the last-seen record (stage I6). They use the actual database.

The matching is replaced by a fixed answer, so the tests check the rule only.
Every object has a name that starts with `__test__ `. The objects go first, then the frames.
"""

import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta

import numpy as np
import pytest

import db.api as db
from ai.api import AiEngine
from ai.types import Box, Candidate, MatchObject, Sighting
from backend.memory import MemoryService
from backend.settings import MemoryConfig
from db.api import gallery, sightings
from db.schema import (
    EnrolledObject,
    Image,
    ImageStatus,
    ObjectStatus,
    PhotoStatus,
    ReferencePhoto,
    Region,
)
from db.schema import Sighting as SightingRow
from tests.ai_support import make_engine
from tests.db_support import (
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
)

PREFIX: str = "__test__ "
NOW: datetime = datetime(2000, 1, 1, 12, 0, 0)
CONFIG: MemoryConfig = MemoryConfig(hysteresis_window_seconds=60, hysteresis_hits=2)
CANDIDATE: Candidate = Candidate(
    Box(10, 20, 60, 80), 0.5, "full", ("a bottle",), True, True
)


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


def make_object() -> EnrolledObject:
    return gallery.create_object(f"{PREFIX}{uuid.uuid4().hex[:8]}", "a bottle")


def make_frame(seen_at: datetime) -> tuple[Image, list[Region]]:
    image: Image = make_image()
    regions: list[Region] = make_regions(image, 1)
    set_status(image, ImageStatus.DONE)
    Image.update(captured_at=seen_at).where(Image.id == image.id).execute()
    return db.get_image(image.id), regions  # type: ignore[return-value]


def observe(
    service: MemoryService, frame: tuple[Image, list[Region]], sighting: Sighting
) -> int:
    """Run `remember` with a fixed matching answer."""
    engine: AiEngine = service._engine
    engine.match_regions = lambda *args: [sighting]  # type: ignore[method-assign]
    service.load_objects = lambda: [object()]  # type: ignore[method-assign,list-item]
    image, regions = frame
    vectors: np.ndarray = np.zeros((1, 384), dtype=np.float32)
    return service.remember(image, [CANDIDATE], regions, vectors)


@pytest.fixture()
def service() -> MemoryService:
    return MemoryService(CONFIG, make_engine())


def sighting_of(obj: EnrolledObject, strong: bool) -> Sighting:
    return Sighting(obj.id, 0, 0.9, 0.5, strong)


def test_a_strong_sighting_changes_the_record_at_once(service: MemoryService) -> None:
    obj = make_object()
    frame = make_frame(NOW)
    assert observe(service, frame, sighting_of(obj, True)) == 1
    record = sightings.get_last_seen(obj.id)
    assert record is not None and record.image_id == frame[0].id
    assert (record.x0, record.y0, record.x1, record.y1) == (10, 20, 60, 80)
    assert record.in_hand and record.seen_at == NOW


def test_one_weak_sighting_is_logged_but_does_not_change_the_record(
    service: MemoryService,
) -> None:
    obj = make_object()
    observe(service, make_frame(NOW), sighting_of(obj, False))
    assert sightings.count_recent(obj.id, NOW, NOW) == 1
    assert sightings.get_last_seen(obj.id) is None


def test_two_weak_sightings_in_the_window_change_the_record(
    service: MemoryService,
) -> None:
    obj = make_object()
    observe(service, make_frame(NOW), sighting_of(obj, False))
    second = make_frame(NOW + timedelta(seconds=40))
    observe(service, second, sighting_of(obj, False))
    record = sightings.get_last_seen(obj.id)
    assert record is not None and record.image_id == second[0].id


def test_a_weak_false_match_after_a_strong_sighting_does_not_replace_the_record(
    service: MemoryService,
) -> None:
    obj = make_object()
    strong = make_frame(NOW)
    observe(service, strong, sighting_of(obj, True))
    observe(service, make_frame(NOW + timedelta(seconds=20)), sighting_of(obj, False))
    record = sightings.get_last_seen(obj.id)
    assert record is not None and record.image_id == strong[0].id


def test_a_strong_sighting_does_not_count_as_a_hit_for_two_weak_ones(
    service: MemoryService,
) -> None:
    obj = make_object()
    observe(service, make_frame(NOW), sighting_of(obj, True))
    observe(service, make_frame(NOW + timedelta(seconds=10)), sighting_of(obj, False))
    third = make_frame(NOW + timedelta(seconds=20))
    observe(service, third, sighting_of(obj, False))  # The second weak hit
    record = sightings.get_last_seen(obj.id)
    assert record is not None and record.image_id == third[0].id


def test_two_weak_sightings_too_far_apart_do_not(service: MemoryService) -> None:
    obj = make_object()
    observe(service, make_frame(NOW), sighting_of(obj, False))
    observe(service, make_frame(NOW + timedelta(seconds=61)), sighting_of(obj, False))
    assert sightings.get_last_seen(obj.id) is None


def test_the_hits_of_another_object_do_not_count(service: MemoryService) -> None:
    first, second = make_object(), make_object()
    observe(service, make_frame(NOW), sighting_of(first, False))
    observe(service, make_frame(NOW + timedelta(seconds=5)), sighting_of(second, False))
    assert sightings.get_last_seen(first.id) is None
    assert sightings.get_last_seen(second.id) is None


def test_the_same_frame_twice_is_one_hit(service: MemoryService) -> None:
    obj = make_object()
    frame = make_frame(NOW)
    observe(service, frame, sighting_of(obj, False))
    observe(service, frame, sighting_of(obj, False))  # For example, after a repair
    assert sightings.get_last_seen(obj.id) is None


def test_one_hit_is_enough_when_the_limit_is_one() -> None:
    service = MemoryService(MemoryConfig(60, 1), make_engine())
    obj = make_object()
    observe(service, make_frame(NOW), sighting_of(obj, False))
    assert sightings.get_last_seen(obj.id) is not None


def test_a_strong_sighting_of_an_older_frame_does_not_replace_a_newer_record(
    service: MemoryService,
) -> None:
    obj = make_object()
    newer = make_frame(NOW)
    observe(service, newer, sighting_of(obj, True))
    observe(service, make_frame(NOW - timedelta(minutes=10)), sighting_of(obj, True))
    assert sightings.get_last_seen(obj.id).image_id == newer[0].id  # type: ignore[union-attr]


def test_a_deleted_object_drops_its_sighting_without_an_error(
    service: MemoryService,
) -> None:
    ghost = Sighting(2**31, 0, 0.9, 0.5, True)
    assert observe(service, make_frame(NOW), ghost) == 0


def test_no_regions_or_no_ready_object_do_nothing(service: MemoryService) -> None:
    image, regions = make_frame(NOW)
    assert service.remember(image, [], [], np.empty((0, 384), dtype=np.float32)) == 0
    service.load_objects = list  # type: ignore[method-assign]
    vectors: np.ndarray = np.zeros((1, 384), dtype=np.float32)
    assert service.remember(image, [CANDIDATE], regions, vectors) == 0
    assert not SightingRow.select().where(SightingRow.image == image.id).exists()


def test_load_objects_takes_ready_objects_with_views_of_the_current_model() -> None:
    engine: AiEngine = make_engine()
    service = MemoryService(CONFIG, engine)
    ready, draft, other_model = make_object(), make_object(), make_object()
    unit: np.ndarray = np.zeros(384, dtype=np.float32)
    unit[0] = 1.0
    for obj, model in ((ready, engine.embedding_model), (other_model, "old-model")):
        photo = gallery.add_reference_photo(
            obj.id, f"{TEST_PREFIX}{uuid.uuid4().hex}.jpg", uuid.uuid4().hex * 2, 80, 60
        )
        ReferencePhoto.update(status=PhotoStatus.PROCESSING.value).where(
            ReferencePhoto.id == photo.id
        ).execute()  # The worker would do this. A claim could take a real photo.
        gallery.store_photo_views(
            photo.id, (1, 1, 5, 5), 0.5, [("original", unit)], model
        )
        gallery.confirm_photo(photo.id)
        gallery.set_calibration(obj.id, 0.8, 0.85, 0.03, model)
    found: list[MatchObject] = service.load_objects()
    ids = {o.object_id for o in found}
    assert ready.id in ids and draft.id not in ids and other_model.id not in ids
    mine = next(o for o in found if o.object_id == ready.id)
    assert mine.phrases == ("a bottle",) and mine.views.shape == (1, 384)
    assert (mine.tau, mine.tau_strong, mine.margin_min) == (0.8, 0.85, 0.03)
    assert ObjectStatus.DRAFT.value == gallery.get_object(draft.id).status
