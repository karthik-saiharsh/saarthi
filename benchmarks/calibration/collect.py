"""Collect the raw detector output for the images of a dataset. The calibration reads it.

Usage: uv run python benchmarks/calibration/collect.py DATASET_DIR OUT_DIR [--phrases "a bottle,a water bottle"]

Each image goes through the same preparation as an upload (`storage.prepare_image`),
so the detector sees the same pixels as in the application. The detector runs with
low thresholds, so that no box is hidden. The calibration raises the thresholds
later, from the cached output. All three passes run (full frame, tiled, generic).

The output is `raw.json` in OUT_DIR. It holds the size, the blur score and the raw boxes of each image.
This script is not part of the runtime.
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

from ai.api import AiEngine
from ai.detector import CandidateDetector
from ai.embedder import RegionEmbedder
from ai.query import QueryEncoder
from ai.settings import load_config
from storage.imaging import prepare_image
from storage.settings import load_config as load_storage_config

logging.getLogger("ai").setLevel(logging.WARNING)

LOW_THRESHOLD: float = 0.15


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--phrases", default="a bottle,a water bottle")
    args: argparse.Namespace = parser.parse_args()
    phrases: list[str] = [p.strip() for p in args.phrases.split(",") if p.strip()]
    assert phrases, "phrases must not be empty"
    args.out.mkdir(parents=True, exist_ok=True)

    base = load_config()
    low = dataclasses.replace(
        base,
        scene=dataclasses.replace(base.scene, enabled=False),
        detection=dataclasses.replace(
            base.detection, box_threshold=LOW_THRESHOLD, text_threshold=LOW_THRESHOLD
        ),
    )
    detector: CandidateDetector = CandidateDetector(low)
    engine: AiEngine = AiEngine(
        low,
        detector=detector,
        embedder=RegionEmbedder(low),
        query_encoder=QueryEncoder(low),
        verifier=None,
    )
    storage_config = load_storage_config()

    result: dict = {"phrases": phrases, "images": {}}
    for path in sorted(args.dataset.glob("*.jp*g"), key=lambda p: int(p.stem)):
        started: float = time.perf_counter()
        prepared = prepare_image(path.read_bytes(), storage_config)
        image = engine.open_image(prepared.data)
        report = engine.assess_quality(image)
        raw = detector.detect(image, phrases, phrases, True, True)
        result["images"][path.stem] = {
            "width": image.width,
            "height": image.height,
            "blur_score": report.blur_score,
            "raw": [
                {
                    "box": [d.box.x0, d.box.y0, d.box.x1, d.box.y1],
                    "score": d.score,
                    "label": d.label,
                    "source": d.source,
                }
                for d in raw
            ],
        }
        print(
            f"{path.name}: {image.width}x{image.height} blur {report.blur_score:.0f}, "
            f"{len(raw)} raw box(es) in {time.perf_counter() - started:.0f} s",
            flush=True,
        )
        (args.out / "raw.json").write_text(json.dumps(result))
    print("done")


if __name__ == "__main__":
    main()
