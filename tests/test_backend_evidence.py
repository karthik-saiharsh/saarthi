"""Tests of the mode `evidence` in the backend: the reference vectors, the memory and the archive search.

They use the actual database and storage. The scorer itself is tested in `tests/test_ai_evidence.py`, so here
the engine call `score_evidence` is replaced by a fixed answer. The tests check the flow around it: which
objects and regions the backend gives, which frames it reads, and what it records.
Every object has a name that starts with `__test__ `. Each test removes its rows and files.
"""

import dataclasses
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from PIL import Image as PilImage

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from ai.types import Box, Candidate, MatchObject, Sighting
from backend.memory import MemoryService
from backend.references import ReferenceVectors
from backend.search import SearchService
from backend.settings import BackendConfig, MemoryConfig, load_config
from db.api import gallery, sightings
from db.schema import EnrolledObject, Image, ImageStatus, ReferencePhoto, Region
from tests.ai_support import make_engine
from tests.db_support import (
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
)
from tests.search_support import (
    BOTTLE,
    PREFIX,
    direction,
    store_frame,
    stored_photo,
)
from tests.search_support import make_object_with_references as make_object

MEMORY: MemoryConfig = MemoryConfig(
    hysteresis_window_seconds=60,
    hysteresis_hits=2,
    verify_weak=False,
    verify_max_references=3,
)
CANDIDATE: Candidate = Candidate(
    Box(1, 1, 11, 11), 0.5, "full", ("a hand",), False, True
)
FRAME: PilImage.Image = PilImage.new("RGB", (200, 200), (90, 90, 90))
FAR: np.ndarray = direction(0, 0, 1)  # Cosine 0 with BOTTLE: below `candidate_min`


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
    EnrolledObject.delete().where(EnrolledObject.name.startswith(PREFIX)).execute()
    new: list[int] = [i.id for i in Image.select(Image.id) if i.id not in before]
    storage.delete_images(
        [p for p in db.delete_images(new) if not p.startswith(TEST_PREFIX)]
    )
    storage.delete_images(files)


class Recorder:
    """Replaces the evidence calls of an engine and records them."""

    def __init__(self, engine: AiEngine, answer: list[Sighting] | None = None) -> None:
        self.answer: list[Sighting] = answer or []
        self.scored: list[dict[str, Any]] = []
        self.crop_calls: list[list[Box]] = []
        self._embed = engine.embed_evidence_crops
        engine.score_evidence = self.score  # type: ignore[method-assign]
        engine.embed_evidence_crops = self.embed  # type: ignore[method-assign]
        engine.match_regions = self.no_tau  # type: ignore[method-assign]

    def embed(self, image: PilImage.Image, boxes: list[Box]) -> np.ndarray:
        self.crop_calls.append(list(boxes))
        return self._embed(image, boxes)

    def score(
        self,
        chosen: list[int],
        vectors: np.ndarray,
        crops: np.ndarray,
        objects: list[MatchObject],
    ) -> list[Sighting]:
        self.scored.append({"chosen": chosen, "crops": crops, "objects": objects})
        return list(self.answer)

    @staticmethod
    def no_tau(*args: object) -> list[Sighting]:
        raise AssertionError("the mode tau must not run")


# ---- Reference vectors ----


def test_reference_vectors_come_from_confirmed_photos_and_are_kept(
    cleanup: list[str],
) -> None:
    engine = make_engine(matching_mode="evidence")
    recorder = Recorder(engine)
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    references = ReferenceVectors(engine)

    first = references.get(obj.id)
    assert first is not None and first.vectors.shape[0] == 2
    assert sorted(b.width for call in recorder.crop_calls for b in call) == [60, 80]
    assert references.get(obj.id) is first  # Kept: no second run
    assert len(recorder.crop_calls) == 2

    # A new confirmed photo changes the set, so the vectors are made again.
    path, sha, width, height = stored_photo()
    cleanup.append(path)
    photo = gallery.add_reference_photo(obj.id, path, sha, width, height)
    ReferencePhoto.update(status="processing").where(
        ReferencePhoto.id == photo.id
    ).execute()  # The worker would do this
    gallery.store_photo_views(
        photo.id, (10, 10, 50, 50), 0.5, [("original", BOTTLE)], engine.embedding_model
    )
    assert references.get(obj.id) is first  # Located, not confirmed: no change
    gallery.confirm_photo(photo.id)
    again = references.get(obj.id)
    assert again is not None and again.vectors.shape[0] == 3


def test_a_photo_that_cannot_be_read_is_left_out(cleanup: list[str]) -> None:
    engine = make_engine(matching_mode="evidence")
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    storage.delete_images([cleanup[0]])
    references = ReferenceVectors(engine).get(obj.id)
    assert references is not None and references.vectors.shape[0] == 1
    storage.delete_images([cleanup[1]])
    assert ReferenceVectors(engine).get(obj.id) is None


# ---- Memory ----


def make_frame() -> tuple[Image, list[Region]]:
    image: Image = make_image()
    regions: list[Region] = make_regions(image, 1)
    set_status(image, ImageStatus.DONE)
    return db.get_image(image.id), regions  # type: ignore[return-value]


def test_the_memory_scores_with_evidence_and_records_a_strong_sighting(
    cleanup: list[str],
) -> None:
    engine = make_engine(matching_mode="evidence")
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    recorder = Recorder(engine, [Sighting(obj.id, 0, 0.26, 0.2, True, 7.5)])
    service = MemoryService(MEMORY, engine)
    image, regions = make_frame()
    # A low DINOv2 score (the watch on a wrist), and a detector label that names another thing ("a hand").
    vector = direction(0.26, 0.97)
    assert (
        service.remember(image, [CANDIDATE], regions, vector.reshape(1, -1), FRAME) == 1
    )

    call = recorder.scored[0]
    assert call["chosen"] == [0]
    target: MatchObject = next(o for o in call["objects"] if o.object_id == obj.id)
    assert target.references is not None and target.references.shape[0] == 2
    assert target.view_photos is not None and len(target.view_photos) == 2
    assert recorder.crop_calls[-1] == [CANDIDATE.box]
    record = sightings.get_last_seen(obj.id)
    assert record is not None and record.strong and record.image_id == image.id


def test_a_weak_evidence_sighting_follows_the_hysteresis(cleanup: list[str]) -> None:
    engine = make_engine(matching_mode="evidence")
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    Recorder(engine, [Sighting(obj.id, 0, 0.5, 0.2, False, 4.0)])
    service = MemoryService(MEMORY, engine)
    image, regions = make_frame()
    assert (
        service.remember(image, [CANDIDATE], regions, BOTTLE.reshape(1, -1), FRAME) == 1
    )
    assert sightings.get_last_seen(obj.id) is None  # One weak frame is not enough
    image, regions = make_frame()
    service.remember(image, [CANDIDATE], regions, BOTTLE.reshape(1, -1), FRAME)
    assert sightings.get_last_seen(obj.id) is not None


def test_without_a_picture_the_memory_uses_tau(cleanup: list[str]) -> None:
    engine = make_engine(matching_mode="evidence")
    obj = make_object(cleanup, [(60, BOTTLE)])
    recorder = Recorder(engine)
    called: list[bool] = []
    engine.match_regions = lambda *args: called.append(True) or []  # type: ignore[method-assign]
    image, regions = make_frame()
    MemoryService(MEMORY, engine).remember(
        image, [CANDIDATE], regions, BOTTLE.reshape(1, -1), None
    )
    assert called == [True] and recorder.scored == []
    assert sightings.get_last_seen(obj.id) is None


def test_an_object_without_readable_references_is_not_scored(
    cleanup: list[str],
) -> None:
    engine = make_engine(matching_mode="evidence")
    obj = make_object(cleanup, [(60, BOTTLE)])
    storage.delete_images(list(cleanup))
    recorder = Recorder(engine)
    image, regions = make_frame()
    assert (
        MemoryService(MEMORY, engine).remember(
            image, [CANDIDATE], regions, BOTTLE.reshape(1, -1), FRAME
        )
        == 0
    )
    assert all(
        o.object_id != obj.id for call in recorder.scored for o in call["objects"]
    )


# ---- Archive search ----


def make_search(engine: AiEngine, **changes: int) -> SearchService:
    base: BackendConfig = load_config()
    config: BackendConfig = dataclasses.replace(
        base, search=dataclasses.replace(base.search, **changes)
    )
    return SearchService(config, engine)


def test_the_archive_search_scores_frames_with_evidence(cleanup: list[str]) -> None:
    engine = make_engine(matching_mode="evidence")
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    frame = store_frame(minutes_ago=5)
    recorder = Recorder(engine, [Sighting(obj.id, 0, 0.9, 0.5, True, 8.0)])
    answer = make_search(engine).find(obj.id)
    assert answer.found and answer.image_id == frame.id and answer.strong
    assert answer.complete
    assert len(recorder.scored) == 1


def test_the_search_reads_only_frames_with_a_candidate(cleanup: list[str]) -> None:
    engine = make_engine(matching_mode="evidence")
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    store_frame(minutes_ago=5, vector=FAR)  # Not a candidate: never read
    store_frame(minutes_ago=6, vector=BOTTLE)
    recorder = Recorder(engine)  # No sighting anywhere
    answer = make_search(engine).find(obj.id)
    assert not answer.found and answer.complete
    assert len(recorder.scored) == 1  # Only the frame with a candidate was scored


def test_the_search_stops_after_the_evidence_limit(cleanup: list[str]) -> None:
    engine = make_engine(matching_mode="evidence")
    obj = make_object(cleanup, [(60, BOTTLE), (80, BOTTLE)])
    for minutes in (5, 6, 7):
        store_frame(minutes_ago=minutes, vector=BOTTLE)
    recorder = Recorder(engine)
    answer = make_search(engine, evidence_max_frames=2).find(obj.id)
    assert not answer.found and not answer.complete
    assert len(recorder.scored) == 2
