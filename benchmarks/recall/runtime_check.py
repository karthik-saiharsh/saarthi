"""Run the matching mode `keys` of the runtime on a frame set and compare it with the replay of `report.py`. Not part of the runtime.

It calls the `AiEngine` methods in the order of the worker and the memory in the mode `keys`
(`Worker._process_keys`, `MemoryService.match_keys`): the candidates of the detector, the boxes of the broad
detector, the SigLIP key and the DINOv2 key of every box, then `AiEngine.match_keys`. It needs no database:
the objects are enrolled from the reference photos of the set, as in `collect.py`.
It does not run the keypoint check (stage I5) or the memory rule (stage I6), so it compares one frame at a time.

It reads the frames read-only and writes `runtime.json` into OUT_DIR. Then it prints the result for each
truth instance and the false sightings, in the form of section 5 of `report.py`.

Usage:
    uv run python benchmarks/recall/runtime_check.py SET_JSON OUT_DIR [--limit N]
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    COVER_IGNORE,
    IOU_SAME,
    BoxTuple,
    FrameSet,
    cover,
    iou,
    load_set,
    resolve,
)

from ai.api import AiEngine, ImageError, open_image
from ai.settings import AiConfig, load_config
from ai.types import Box, Candidate, MatchObject, MatchRegion, Sighting

logger: logging.Logger = logging.getLogger("recall.runtime")


def runtime_config() -> AiConfig:
    """The config of the application in the mode `keys`, without the models that the check does not use."""
    config: AiConfig = load_config()
    return dataclasses.replace(
        config,
        matching=dataclasses.replace(config.matching, mode="keys"),
        speech=dataclasses.replace(config.speech, enabled=False),
        language=dataclasses.replace(config.language, enabled=False),
        verification=dataclasses.replace(config.verification, enabled=False),
    )


def enroll(
    engine: AiEngine, config: AiConfig, frame_set: FrameSet
) -> list[MatchObject]:
    """One object for each object of the set, with the key vectors of its reference photos."""
    objects: list[MatchObject] = []
    photo: int = 0
    for index, (name, spec) in enumerate(frame_set.objects.items(), start=1):
        sig: list[np.ndarray] = []
        dino: list[np.ndarray] = []
        photos: list[int] = []
        for path, box in spec.references:
            photo += 1
            image = open_image(resolve(path).read_bytes(), config.image.max_pixels)
            sig.append(engine.embed_evidence_crops(image, [Box(*box)])[0])
            dino.append(engine.embed_key_crops(image, [Box(*box)])[0])
            photos.append(photo)
        objects.append(
            MatchObject(
                object_id=index,
                tracked=True,
                phrases=spec.phrases,
                views=np.stack(dino),  # Not used by the mode `keys`
                tau=1.0,
                tau_strong=1.0,
                margin_min=0.0,
                references=np.stack(sig),
                reference_photos=np.array(photos),
                key_references=np.stack(dino),
            )
        )
        logger.info("Object %s: %d reference photos", name, len(photos))
    return objects


def run_frame(
    engine: AiEngine, image, phrases: list[str], objects: list[MatchObject]
) -> tuple[list[Candidate], int, list[Sighting]]:
    """The mode `keys` of the worker for one frame. Return the boxes, the number of detector candidates, the sightings."""
    candidates: list[Candidate] = engine.detect_candidates(
        image, phrases=phrases, include_generic=False
    )
    found = engine.detect_open(image) if engine.open_detection_enabled else []
    merged: list[Candidate] = candidates + engine.broad_candidates(image, found)
    if not merged:
        return merged, len(candidates), []
    boxes: list[Box] = [c.box for c in merged]
    sig: np.ndarray = engine.embed_evidence_crops(image, boxes)
    dino: np.ndarray = engine.embed_key_crops(image, boxes)
    regions: list[MatchRegion] = [
        MatchRegion(tuple(c.phrases), c.quality_ok, c.det_score, c.depicted)
        for c in merged
    ]
    return merged, len(candidates), engine.match_keys(sig, dino, regions, objects)


def summary(frame_set: FrameSet, results: dict[str, list[dict]]) -> None:
    """Print the result for each truth instance and the false sightings, as section 5 of `report.py`."""
    names: list[str] = list(frame_set.objects)
    counts: dict[str, int] = {}
    false: dict[str, list[int]] = {n: [0, 0] for n in names}  # home, cohort
    used: set[tuple[str, int]] = set()
    for name in names:
        cells: list[str] = []
        truth = frame_set.truth.get(name, {})
        for kind in ("true", "hard"):
            for key, truth_box in truth.get(kind, []):
                step: str = "notrun" if key not in results else "-"
                for position, s in enumerate(results.get(key, [])):
                    box: BoxTuple = tuple(s["box"])
                    if s["object"] == name and (
                        iou(box, truth_box) >= IOU_SAME
                        or cover(box, truth_box) >= COVER_IGNORE
                    ):
                        step = "STRONG" if s["strong"] else "weak"
                        used.add((key, position))
                cells.append(f"{key}{kind[0]}:{step}")
                if kind == "true":
                    counts[step] = counts.get(step, 0) + 1
        print(f"  {name:8s} " + " ".join(cells))
    for key, found in results.items():
        for position, s in enumerate(found):
            if (key, position) in used:
                continue
            # A sighting on a part of a truth box of the same object is not false (as `report.py` does).
            truth = frame_set.truth.get(s["object"], {})
            near: bool = any(
                frame == key and cover(tuple(s["box"]), box) >= COVER_IGNORE
                for kind in ("true", "hard")
                for frame, box in truth.get(kind, [])
            )
            if not near:
                false[s["object"]][int(key in frame_set.cohort)] += 1
    print(
        "  true instances: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    print(
        f"  false sightings: home {sum(f[0] for f in false.values())}, cohort {sum(f[1] for f in false.values())}"
    )
    for name, (home, cohort) in false.items():
        print(f"    {name:8s} {home} / {cohort}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    for noisy in ("ai", "httpx", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("set_file", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    assert args.limit is None or args.limit > 0, "--limit must be positive"
    frame_set: FrameSet = load_set(args.set_file)
    config: AiConfig = runtime_config()
    engine: AiEngine = AiEngine(config)
    assert engine.matching_mode == "keys", "the mode keys needs the scene model"
    objects: list[MatchObject] = enroll(engine, config, frame_set)
    names: list[str] = list(frame_set.objects)
    phrases: list[str] = list(
        dict.fromkeys(p for spec in frame_set.objects.values() for p in spec.phrases)
    )
    work: list[tuple[str, str]] = list(frame_set.frames.items()) + list(
        frame_set.cohort.items()
    )
    if args.limit is not None:
        work = work[: args.limit]
    results: dict[str, list[dict]] = {}
    started: float = time.perf_counter()
    for done, (key, path) in enumerate(work, start=1):
        try:
            image = open_image(resolve(path).read_bytes(), config.image.max_pixels)
        except ImageError:
            logger.warning("Frame %s cannot be read. It is left out", key)
            continue
        merged, n_candidates, found = run_frame(engine, image, phrases, objects)
        results[key] = [
            {
                "object": names[s.object_id - 1],
                "box": [
                    merged[s.region_index].box.x0,
                    merged[s.region_index].box.y0,
                    merged[s.region_index].box.x1,
                    merged[s.region_index].box.y1,
                ],
                "strong": s.strong,
                "sig": s.similarity,
                "dino": s.score,
                "broad": s.region_index >= n_candidates,
            }
            for s in found
        ]
        if done % 25 == 0:
            logger.info(
                "%d of %d frames, %.0f s",
                done,
                len(work),
                time.perf_counter() - started,
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "runtime.json").write_text(json.dumps(results, indent=1))
    logger.info("Done: %d frames, %.0f s", len(results), time.perf_counter() - started)
    print("\nRuntime, mode keys (one frame at a time)")
    summary(frame_set, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
