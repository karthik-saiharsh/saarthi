"""Tests of the matching mode `keys` in the backend: the reference vectors, the worker, the memory and the archive search.

They use the actual database and storage. The rule itself is tested in `tests/test_ai_keys.py`, so here the engine
call `match_keys` is replaced by a fixed answer. The tests check the flow around it: which boxes the backend gives,
which boxes become regions, where a sighting points, and what the open search stores.
Every object has a name that starts with `__test__ `. Each test removes its rows and files.
"""

import dataclasses
from collections.abc import Iterator
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.types import (
    Box,
    MatchObject,
    MatchRegion,
    OpenDetection,
    RawDetection,
    Sighting,
)
from backend.ingest import IngestService
from backend.references import ReferenceVectors
from backend.search import SearchService
from backend.settings import BackendConfig, load_config
from backend.worker import Worker
from db.api import sightings
from db.schema import EnrolledObject, Image, ImageStatus, PhotoStatus, ReferencePhoto
from tests.ai_support import (
    FakeDetector,
    FakeEmbedder,
    FakeOpenDetector,
    FakeSceneClassifier,
    make_engine,
)
from tests.db_support import TEST_PREFIX, skip_if_actual_data_has_problems
from tests.search_support import BOTTLE, PREFIX
from tests.search_support import make_object_with_references as make_object

CANDIDATE_BOX: Box = Box(10, 10, 40, 40)
BROAD_A: Box = Box(45, 5, 75, 30)
BROAD_B: Box = Box(8, 32, 38, 56)


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    storage.init_storage()
    database, _ = db.init_db()
    skip_if_actual_data_has_problems()
    if Image.select().where(Image.status == ImageStatus.PENDING.value).exists():
        pytest.skip("Actual database has pending images. The worker would claim one")
    if (
        ReferencePhoto.select()
        .where(ReferencePhoto.status == PhotoStatus.PENDING.value)
        .exists()
    ):
        pytest.skip("Actual database has pending reference photos")
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[list[str]]:
    files: list[str] = []
    before: set[int] = {i.id for i in Image.select(Image.id)}
    yield files
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    new: list[int] = [i.id for i in Image.select(Image.id) if i.id not in before]
    storage.delete_images(
        [p for p in db.delete_images(new) if not p.startswith(TEST_PREFIX)]
    )
    storage.delete_images(files)


class KeyRecorder:
    """Replaces `match_keys` of an engine. It answers with a sighting on the region at `answer_index`, if set."""

    def __init__(self, engine: AiEngine, strong: bool = True) -> None:
        self.answer_index: int | None = None
        self.strong: bool = strong
        self.calls: list[dict[str, Any]] = []
        engine.match_keys = self.match  # type: ignore[method-assign]
        engine.score_evidence = self.never  # type: ignore[method-assign]
        engine.match_regions = self.never  # type: ignore[method-assign]

    def match(
        self,
        sig_vectors: np.ndarray,
        dino_vectors: np.ndarray,
        regions: list[MatchRegion],
        objects: list[MatchObject],
    ) -> list[Sighting]:
        self.calls.append(
            {
                "sig": sig_vectors,
                "dino": dino_vectors,
                "regions": regions,
                "objects": objects,
            }
        )
        tracked: list[MatchObject] = [o for o in objects if o.tracked]
        if self.answer_index is None or not tracked:
            return []
        return [
            Sighting(
                tracked[0].object_id, self.answer_index, 0.8, 0.3, self.strong, 0.5
            )
        ]

    @staticmethod
    def never(*args: object) -> list[Sighting]:
        raise AssertionError("another matching mode must not run")


@dataclasses.dataclass
class Setup:
    worker: Worker
    engine: AiEngine
    detector: FakeDetector
    open_detector: FakeOpenDetector
    scene: FakeSceneClassifier
    ingest: IngestService
    recorder: KeyRecorder


def make_setup(verify_weak: bool = True) -> Setup:
    base: BackendConfig = load_config()
    config: BackendConfig = dataclasses.replace(
        base,
        worker=dataclasses.replace(base.worker, generic_pass_every=0),
        memory=dataclasses.replace(base.memory, verify_weak=verify_weak),
    )
    detector, open_detector, scene = (
        FakeDetector(),
        FakeOpenDetector(),
        FakeSceneClassifier(),
    )
    engine: AiEngine = make_engine(
        detector,
        FakeEmbedder(),
        scene_classifier=scene,
        open_detector=open_detector,
        matching_mode="keys",
    )
    worker: Worker = Worker(config, engine)
    return Setup(
        worker,
        engine,
        detector,
        open_detector,
        scene,
        IngestService(config, on_stored=worker.wake),
        KeyRecorder(engine),
    )


def sharp_frame(seed: int) -> bytes:
    pixels: np.ndarray = np.random.default_rng(seed).integers(
        0, 255, (60, 80, 3), dtype=np.uint8
    )
    buffer: BytesIO = BytesIO()
    PilImage.fromarray(pixels).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def frame_with_boxes(setup: Setup, seed: int) -> int:
    """Upload a frame with one candidate of the detector and two boxes of the broad detector. Return its id."""
    setup.detector.detections = [RawDetection(CANDIDATE_BOX, 0.6, "a bottle", "full")]
    setup.open_detector.detections = [
        OpenDetection(BROAD_A, "laptop", 0.9),
        OpenDetection(BROAD_B, "handcuff", 0.7),
    ]
    return setup.ingest.submit(sharp_frame(seed)).image_id


# ---- Reference vectors ----


def test_reference_vectors_have_a_dino_key_for_each_photo(cleanup: list[str]) -> None:
    engine = make_engine(matching_mode="keys")
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    references = ReferenceVectors(engine).get(obj.id)
    assert references is not None
    assert references.key_vectors.shape[0] == references.vectors.shape[0] == 2


# ---- The worker ----


def test_all_boxes_are_stored_and_matched_and_a_sighting_can_be_on_a_broad_box(
    cleanup: list[str],
) -> None:
    setup = make_setup()
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    setup.recorder.answer_index = 2  # BROAD_B
    image_id: int = frame_with_boxes(setup, 1)
    assert setup.worker.process_next() is True
    assert db.get_image(image_id).status == ImageStatus.DONE.value  # type: ignore[union-attr]

    call = setup.recorder.calls[-1]
    regions: list[MatchRegion] = call["regions"]
    assert len(regions) == 3 and call["sig"].shape[0] == call["dino"].shape[0] == 3
    assert regions[0].phrases == ("a bottle",) and regions[
        0
    ].det_score == pytest.approx(0.6)
    assert [r.phrases for r in regions[1:]] == [(), ()]
    target = next(o for o in call["objects"] if o.object_id == obj.id)
    assert target.key_references is not None and target.key_references.shape[0] == 2

    stored = db.get_regions(image_id)
    assert [(r.source, r.label) for r in stored] == [
        ("full", None),
        ("broad", "laptop"),
        ("broad", "handcuff"),
    ]
    record = sightings.get_last_seen(obj.id)
    assert record is not None and record.image_id == image_id
    assert (record.x0, record.y0, record.x1, record.y1) == (8, 32, 38, 56)
    # 1 SigLIP call for the 3 boxes of the frame, then 2 for the 2 reference photos.
    assert [len(c) for c in setup.scene.crop_calls] == [3, 1, 1]


def test_without_a_sighting_all_boxes_are_still_stored(cleanup: list[str]) -> None:
    setup = make_setup()
    make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    image_id: int = frame_with_boxes(setup, 2)
    setup.worker.process_next()
    assert len(db.get_regions(image_id)) == 3
    assert len(db.get_broad_regions(image_id)) == 2


def test_a_weak_sighting_on_a_broad_box_follows_the_hysteresis(
    cleanup: list[str],
) -> None:
    setup = make_setup(verify_weak=False)  # The fake check would promote the sighting
    setup.recorder.strong = False
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    setup.recorder.answer_index = 1  # BROAD_A
    frame_with_boxes(setup, 3)
    setup.worker.process_next()
    assert sightings.get_last_seen(obj.id) is None  # One weak frame is not enough
    second: int = frame_with_boxes(setup, 4)
    setup.worker.process_next()
    record = sightings.get_last_seen(obj.id)
    assert record is not None and record.image_id == second


def test_without_a_ready_object_the_frame_is_done_with_its_candidates() -> None:
    setup = make_setup()
    image_id: int = frame_with_boxes(setup, 5)
    setup.worker.process_next()
    assert db.get_image(image_id).status == ImageStatus.DONE.value  # type: ignore[union-attr]
    assert len(db.get_regions(image_id)) == 3
    assert setup.recorder.calls == []  # No object, so nothing to match


def test_an_error_of_the_broad_detector_leaves_the_candidates(
    cleanup: list[str],
) -> None:
    from ai.errors import InferenceError

    setup = make_setup()
    make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    image_id: int = frame_with_boxes(setup, 6)
    setup.open_detector.error = InferenceError("test")
    setup.worker.process_next()
    assert db.get_image(image_id).status == ImageStatus.DONE.value  # type: ignore[union-attr]
    assert len(setup.recorder.calls[-1]["regions"]) == 1  # The candidate only
    assert db.get_broad_regions(image_id) == []


# ---- The archive search ----


def test_the_archive_search_scores_the_stored_vectors_and_reads_no_frame(
    cleanup: list[str],
) -> None:
    setup = make_setup()
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    image_id: int = frame_with_boxes(setup, 7)
    setup.worker.process_next()  # No sighting: the recorder has no answer yet
    assert sightings.get_last_seen(obj.id) is None
    setup.recorder.answer_index = 1  # BROAD_A, a box that the memory did not match
    calls_before: int = len(setup.scene.crop_calls)
    answer = SearchService(load_config(), setup.engine).find(obj.id)
    assert answer.found and answer.image_id == image_id
    assert answer.box == (45, 5, 75, 30)
    call = setup.recorder.calls[-1]
    assert len(call["regions"]) == 3  # All boxes of the frame, the broad ones too
    # The search made no SigLIP vector of a frame box (only the reference crops, if they were not kept).
    assert all(len(c) == 1 for c in setup.scene.crop_calls[calls_before:])
