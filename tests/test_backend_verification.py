"""Tests of the check of weak sightings (stage I5) in the memory. They use the actual database and storage.

The matching is replaced by a fixed answer, and the keypoint model is a fake that gives the number
that a test sets. The reference photos are real files. The sizes of their boxes are different, so the
size of the crop that the verifier receives shows which photo it is.
Every object has a name that starts with `__test__ `. The objects go first, then the frames and files.
"""

import dataclasses
from collections.abc import Iterator
from datetime import datetime

import numpy as np
import pytest
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.errors import InferenceError
from ai.settings import load_config
from ai.types import Box, Candidate, Sighting
from backend.memory import MemoryService
from backend.settings import MemoryConfig
from db.api import gallery, sightings
from db.schema import (
    EnrolledObject,
    Image,
    ImageStatus,
    Region,
)
from db.schema import Sighting as SightingRow
from tests.ai_support import (
    FakeDetector,
    FakeEmbedder,
    FakeOpenDetector,
    FakeQueryEncoder,
    FakeSceneClassifier,
    FakeSpeechRecognizer,
    FakeVerifier,
    make_engine,
)
from tests.db_support import (
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
)
from tests.search_support import PREFIX, direction
from tests.search_support import make_object_with_references as make_object

NOW: datetime = datetime(2000, 1, 1, 12, 0, 0)
CANDIDATE: Candidate = Candidate(
    Box(10, 20, 60, 80), 0.5, "full", ("a bottle",), False, True
)
PADDING: float = load_config().verification.crop_padding
FRAME: PilImage.Image = PilImage.new(
    "RGB", (200, 200), (90, 90, 90)
)  # The decoded frame
CONFIG: MemoryConfig = MemoryConfig(
    hysteresis_window_seconds=60,
    hysteresis_hits=2,
    verify_weak=True,
    verify_max_references=3,
)


def crop_side(box_side: int) -> int:
    """The side of the crop that the verifier receives for a square box."""
    return box_side + 2 * round(box_side * PADDING)


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[list[str]]:
    """Give the test a list. The test appends the paths of the files that it stored."""
    files: list[str] = []
    yield files
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    db.delete_images(
        [i.id for i in Image.select().where(Image.file_path.startswith(TEST_PREFIX))]
    )
    storage.delete_images(files)


def make_frame() -> tuple[Image, list[Region]]:
    image: Image = make_image()
    regions: list[Region] = make_regions(image, 1)
    set_status(image, ImageStatus.DONE)
    Image.update(captured_at=NOW).where(Image.id == image.id).execute()
    return db.get_image(image.id), regions  # type: ignore[return-value]


def run(
    service: MemoryService,
    obj: EnrolledObject,
    strong: bool = False,
    vector: np.ndarray | None = None,
    picture: PilImage.Image | None = FRAME,
) -> int:
    """Run `remember` for one frame with a fixed matching answer."""
    image, regions = make_frame()
    sighting: Sighting = Sighting(obj.id, 0, 0.9, 0.5, strong)
    service._engine.match_regions = lambda *args: [sighting]  # type: ignore[method-assign]
    service.load_objects = lambda: [object()]  # type: ignore[method-assign,list-item]
    region_vector: np.ndarray = (
        direction(1, 0, 0) if vector is None else vector
    ).reshape(1, -1)
    return service.remember(image, [CANDIDATE], regions, region_vector, picture)


def make_service(
    verifier: FakeVerifier, config: MemoryConfig = CONFIG
) -> MemoryService:
    return MemoryService(config, make_engine(verifier=verifier))


def row_strong(obj: EnrolledObject) -> bool:
    return bool(
        sightings.get_last_seen(obj.id) and sightings.get_last_seen(obj.id).strong
    )


def test_a_weak_sighting_that_passes_the_check_changes_the_record_at_once(
    cleanup: list[str],
) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    verifier: FakeVerifier = FakeVerifier(answer=100)
    assert run(make_service(verifier), obj) == 1
    record = sightings.get_last_seen(obj.id)
    assert record is not None and record.strong  # Promoted: one weak sighting is enough
    assert len(verifier.calls) == 1


def test_the_promoted_sighting_is_logged_as_strong(cleanup: list[str]) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    run(make_service(FakeVerifier(100)), obj)
    rows = list(SightingRow.select().where(SightingRow.enrolled_object == obj.id))
    assert len(rows) == 1 and rows[0].strong


def test_a_weak_sighting_that_fails_the_check_stays_weak(cleanup: list[str]) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    assert run(make_service(FakeVerifier(answer=2)), obj) == 1
    assert sightings.get_last_seen(obj.id) is None  # One weak hit is not enough
    row = SightingRow.get(SightingRow.enrolled_object == obj.id)
    assert not row.strong  # The check never rejects: the sighting is in the log


def test_two_weak_sightings_that_fail_still_change_the_record_by_the_old_rule(
    cleanup: list[str],
) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    service: MemoryService = make_service(FakeVerifier(answer=2))
    run(service, obj)
    run(service, obj)  # Same time, two frames: the rule of the hysteresis
    record = sightings.get_last_seen(obj.id)
    assert record is not None and not record.strong


def test_a_strong_sighting_is_not_checked(cleanup: list[str]) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    verifier: FakeVerifier = FakeVerifier()
    run(make_service(verifier), obj, strong=True)
    assert verifier.calls == [] and row_strong(obj)


def test_the_most_similar_reference_is_compared_first(cleanup: list[str]) -> None:
    # Box sides 40, 80 and 120 for the views a, b and c. The region is closest to b, then c, then a.
    obj = make_object(
        cleanup,
        [(40, direction(1, 0, 0)), (80, direction(0, 1, 0)), (120, direction(0, 0, 1))],
    )
    verifier: FakeVerifier = FakeVerifier(answer=0)
    run(make_service(verifier), obj, vector=direction(0, 1, 0.1))
    first_sides: list[int] = [reference[0] for reference, _ in verifier.calls]
    assert first_sides == [crop_side(80), crop_side(120), crop_side(40)]


def test_the_check_stops_at_the_first_reference_that_passes(cleanup: list[str]) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0)), (80, direction(0, 1, 0))])
    verifier: FakeVerifier = FakeVerifier(answer=100)
    run(make_service(verifier), obj, vector=direction(0, 1, 0))
    assert len(verifier.calls) == 1 and verifier.calls[0][0][0] == crop_side(80)


def test_at_most_verify_max_references_photos_are_used(cleanup: list[str]) -> None:
    obj = make_object(
        cleanup,
        [
            (side, direction(*[1 if i == n else 0 for i in range(5)]))
            for n, side in enumerate((20, 30, 40, 50, 60))
        ],
    )
    verifier: FakeVerifier = FakeVerifier(answer=0)
    config: MemoryConfig = dataclasses.replace(CONFIG, verify_max_references=2)
    run(make_service(verifier, config), obj, vector=direction(1, 0, 0, 0, 0))
    assert len(verifier.calls) == 2


def test_the_candidate_is_cut_from_its_box_with_the_same_padding(
    cleanup: list[str],
) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    verifier: FakeVerifier = FakeVerifier(answer=0)
    run(make_service(verifier), obj)
    candidate_size: tuple[int, int] = verifier.calls[0][1]
    pad_x, pad_y = round(50 * PADDING), round(60 * PADDING)
    assert candidate_size == (50 + 2 * pad_x, 60 + 2 * pad_y)  # The box is 50 x 60


@pytest.mark.parametrize("mode", ["no picture", "verify_weak off", "verifier off"])
def test_no_check_is_made_when_it_is_off_or_cannot_be_made(
    cleanup: list[str], mode: str
) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    verifier: FakeVerifier = FakeVerifier()
    service: MemoryService = make_service(verifier)
    picture = PilImage.new("RGB", (200, 200))
    if mode == "no picture":
        picture = None  # type: ignore[assignment]
    elif mode == "verify_weak off":
        service = make_service(verifier, dataclasses.replace(CONFIG, verify_weak=False))
    else:
        base = load_config()
        config = dataclasses.replace(
            base, verification=dataclasses.replace(base.verification, enabled=False)
        )
        engine = AiEngine(
            config,
            detector=FakeDetector(),  # type: ignore[arg-type]
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            query_encoder=FakeQueryEncoder(),  # type: ignore[arg-type]
            scene_classifier=FakeSceneClassifier(),  # type: ignore[arg-type]
            speech_recognizer=FakeSpeechRecognizer(),  # type: ignore[arg-type]
            open_detector=FakeOpenDetector(),  # type: ignore[arg-type]
        )
        service = MemoryService(CONFIG, engine)
    assert run(service, obj, picture=picture) == 1
    assert verifier.calls == [] and sightings.get_last_seen(obj.id) is None


def test_an_error_of_the_verifier_leaves_the_sighting_weak_and_does_not_fail(
    cleanup: list[str],
) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    verifier: FakeVerifier = FakeVerifier()
    verifier.error = InferenceError("test")
    assert run(make_service(verifier), obj) == 1  # No exception
    assert sightings.get_last_seen(obj.id) is None


def test_a_reference_file_that_is_gone_is_left_out(cleanup: list[str]) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0)), (80, direction(0, 1, 0))])
    best = [p for p in gallery.list_photos(obj.id)][1]  # The view of the box 80
    storage.delete_images([best.file_path])
    verifier: FakeVerifier = FakeVerifier(answer=100)
    run(make_service(verifier), obj, vector=direction(0, 1, 0))
    assert len(verifier.calls) == 1 and verifier.calls[0][0][0] == crop_side(40)
    assert row_strong(obj)


def test_when_no_reference_file_can_be_read_the_sighting_stays_weak(
    cleanup: list[str],
) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    storage.delete_images([p.file_path for p in gallery.list_photos(obj.id)])
    verifier: FakeVerifier = FakeVerifier(answer=100)
    assert run(make_service(verifier), obj) == 1
    assert verifier.calls == [] and sightings.get_last_seen(obj.id) is None


def test_the_log_of_the_application_shows_no_name_or_phrase(
    cleanup: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    obj = make_object(cleanup, [(40, direction(1, 0, 0))])
    name: str = gallery.get_object(obj.id).name  # type: ignore[union-attr]
    with caplog.at_level("DEBUG"):
        run(make_service(FakeVerifier(100)), obj)
    # The SQL logger of peewee shows the rows of the test itself. Only the application counts.
    own: str = "\n".join(
        r.getMessage()
        for r in caplog.records
        if r.name.startswith(("backend", "ai", "storage"))
    )
    assert own and name not in own and "a bottle" not in own
