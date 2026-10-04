"""Calibration of the verification (stage I5): matching keypoints of true and false candidates.

Usage: uv run python benchmarks/calibration/analyse_verification.py DATASET_DIR CACHE_DIR [--min-similarity 0.35]

For each candidate (a labelled bottle, or a false candidate whose similarity to the object is
at least `--min-similarity`), the script counts the keypoints that match with the reference
photos. The reference photos are the other positive photos (never the photo of the candidate),
the three that are most similar to the candidate, as the application chooses them.
The count of a candidate is the highest count over its references.
This script is not part of the runtime.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from common import candidates_for, iou, load_image, load_raw

from ai.api import AiEngine
from ai.settings import load_config
from ai.types import Box
from ai.verifier import KeypointVerifier

MAX_REFERENCES: int = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    args = parser.parse_args()

    raw = load_raw(args.cache)
    labels = json.loads((Path(__file__).parent / "labels.json").read_text())["boxes"]
    truth = {n: [Box(*b) for b in boxes] for n, boxes in labels.items()}
    positives = [n for n, boxes in truth.items() if boxes]
    base = load_config()
    engine = AiEngine(
        dataclasses.replace(
            base,
            verification=dataclasses.replace(base.verification, enabled=False),
            scene=dataclasses.replace(base.scene, enabled=False),
        )
    )
    verifier = KeypointVerifier(base)
    images = {n: load_image(args.dataset, n) for n in raw["images"]}
    views_of = {
        n: np.stack([v for _, v in engine.embed_views(images[n], truth[n][0])])
        for n in positives
    }

    rows = []
    for name, image in images.items():
        candidates = candidates_for(
            image, raw["images"][name], raw["phrases"], base, ("full", "generic")
        )
        vectors = engine.embed_regions(image, [c.box for c in candidates])
        for c, vector in zip(candidates, vectors):
            best_iou = max((iou(c.box, b) for b in truth[name]), default=0.0)
            kind = (
                "TRUE " if best_iou >= 0.5 else "false" if best_iou < 0.1 else "part "
            )
            if kind == "part ":
                continue
            refs = [p for p in positives if p != name]
            similarity = {p: float((views_of[p] @ vector).max()) for p in refs}
            best_sim = max(similarity.values())
            if kind == "false" and best_sim < args.min_similarity:
                continue
            order = sorted(refs, key=lambda p: -similarity[p])[:MAX_REFERENCES]
            candidate_crop = engine.verification_crop(image, c.box)
            counts = [
                verifier.count_matches(
                    engine.verification_crop(images[p], truth[p][0]), candidate_crop
                )
                for p in order
            ]
            rows.append((kind, name, best_sim, max(counts), counts, c.quality_ok))
            print(
                f"{kind} photo {name:>2} sim {best_sim:.3f} matches {counts} best {max(counts):>3} gates {'ok' if c.quality_ok else 'FAIL'}",
                flush=True,
            )

    true_counts = sorted(r[3] for r in rows if r[0] == "TRUE ")
    false_counts = sorted(r[3] for r in rows if r[0] == "false")
    print(f"\nTRUE bottles ({len(true_counts)}): matches {true_counts}")
    print(
        f"false candidates with a similarity of at least {args.min_similarity} ({len(false_counts)}): matches {false_counts}"
    )
    gap = (min(true_counts) if true_counts else 0) - (
        max(false_counts) if false_counts else 0
    )
    print(
        f"lowest true count - highest false count = {gap}  (positive: a limit between them exists)"
    )


if __name__ == "__main__":
    main()
