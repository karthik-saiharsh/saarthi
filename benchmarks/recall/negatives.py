"""Collect the negatives of the reference photos and add them to a signal store. Not part of the runtime.

A reference photo shows the object and also its place: the table, the bed, other things. Every box of the photo
that is not the object is a negative from the home of the patient, and it costs nothing at enrollment.
This script finds these boxes with the boxes of `collect.py` (the candidates of today and the YOLOE boxes),
leaves out each box that covers a half or more of the object box (a part of the object, or a box around it),
and writes the vectors of the others. A negative belongs only to the object of its photo: another registered
object in the photo is a negative of this object, but not of itself.

It reads the photos read-only and writes `negatives.npz` and `negatives.json` into STORE_DIR.

Usage:
    uv run python benchmarks/recall/negatives.py SET_JSON STORE_DIR
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect import Collector, Polygon, as_tuple, best_polygon, read
from common import COVER_IGNORE, VECTOR_KINDS, BoxTuple, cover, load_set

logger: logging.Logger = logging.getLogger("recall.negatives")


def collect(collector: Collector) -> tuple[list[dict], dict[str, np.ndarray]]:
    """The negative boxes of all reference photos, with the photo numbers of `Collector.enroll`."""
    rows: list[dict] = []
    vectors: dict[str, list[np.ndarray]] = {k: [] for k in VECTOR_KINDS}
    photo: int = 0
    for name in collector.names:
        for path, object_box in collector.set.objects[name].references:
            photo += 1
            image = read(collector.config, path)
            candidates = collector.engine.detect_candidates(
                image, phrases=collector.phrases, include_generic=False
            )
            broad = collector.broad.detect(image)
            found: list[tuple[BoxTuple, str]] = [
                (as_tuple(c.box), "gdino") for c in candidates
            ] + [(b, "yoloe") for b, _, _, _ in broad]
            kept: list[tuple[BoxTuple, str]] = [
                (box, source)
                for box, source in found
                if cover(box, object_box) < COVER_IGNORE
            ]
            logger.info(
                "Object %s, photo %d: %d boxes, %d negatives",
                name,
                photo,
                len(found),
                len(kept),
            )
            if not kept:
                continue
            polygons: list[Polygon] = [best_polygon(box, broad) for box, _ in kept]
            made = collector.signals(image, [box for box, _ in kept], polygons)
            for kind in VECTOR_KINDS:
                vectors[kind].append(made[kind])
            rows.extend(
                {"object": name, "photo": photo, "box": list(box), "source": source}
                for box, source in kept
            )
    stacked: dict[str, np.ndarray] = {
        k: np.concatenate(v).astype(np.float32) for k, v in vectors.items() if v
    }
    return rows, stacked


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    for noisy in ("ai", "httpx", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("set_file", type=Path)
    parser.add_argument("store", type=Path)
    args = parser.parse_args()
    assert (args.store / "vectors.npz").is_file(), f"no store in {args.store}"
    collector: Collector = Collector(load_set(args.set_file), args.store)
    rows, vectors = collect(collector)
    assert rows, "no negative box in the reference photos"
    np.savez(args.store / "negatives.npz", **vectors)
    (args.store / "negatives.json").write_text(json.dumps(rows, indent=1))
    logger.info("Wrote %d negatives", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
