"""Calibration of the matching, with a leave-out test. It needs the embedder.

Usage: uv run python benchmarks/calibration/analyse_matching.py DATASET_DIR CACHE_DIR

Protocol: the photos with a labelled bottle are the positives. For each subset of 3 or 4
positive photos, the script enrolls the object from them (the views of the labelled box,
as the enrollment does), calculates `tau` with the formula of the application, and tests:

- recall: the bottle of each positive photo that is NOT in the subset has a similarity of at least `tau`,
- false accepts: a candidate that is not the bottle (any photo) has a similarity of at least `tau`.

The similarity of a candidate is the highest cosine to a view of the object, as in `ai/matching.py`.
The candidates are those of the detector with the default thresholds (the passes `full` and `generic`).
A small data set: read the numbers as a sign, not as a proof. This script is not part of the runtime.
"""

import dataclasses
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from common import candidates_for, iou, load_image, load_raw, with_quality

from ai.api import AiEngine
from ai.calibration import calibrate
from ai.quality import blur_score
from ai.settings import load_config
from ai.types import Box

HIT_IOU: float = 0.5
FALSE_IOU: float = 0.1
FACTORS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
FIXED: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def main() -> None:
    dataset: Path = Path(sys.argv[1])
    raw: dict = load_raw(Path(sys.argv[2]))
    labels: dict[str, list[list[int]]] = json.loads(
        (Path(__file__).parent / "labels.json").read_text()
    )["boxes"]
    base = load_config()
    engine: AiEngine = AiEngine(
        dataclasses.replace(
            base,
            verification=dataclasses.replace(base.verification, enabled=False),
            scene=dataclasses.replace(base.scene, enabled=False),
        )
    )
    truth = {n: [Box(*b) for b in boxes] for n, boxes in labels.items()}
    positives: list[str] = [n for n, boxes in truth.items() if boxes]
    images = {n: load_image(dataset, n) for n in raw["images"]}

    # Views of each positive photo (as the enrollment makes them), and the vector of each candidate.
    views_of: dict[str, np.ndarray] = {
        n: np.stack([v for _, v in engine.embed_views(images[n], truth[n][0])])
        for n in positives
    }
    records: list[dict] = []
    for name, image in images.items():
        # The gate without the blur check. The blur limit is applied later, so it can be swept.
        candidates = candidates_for(
            image,
            raw["images"][name],
            raw["phrases"],
            with_quality(base, crop_blur_min=0.0),
            ("full", "generic"),
        )
        vectors = engine.embed_regions(image, [c.box for c in candidates])
        for c, vector in zip(candidates, vectors):
            best_iou = max((iou(c.box, b) for b in truth[name]), default=0.0)
            kind = (
                "hit"
                if best_iou >= HIT_IOU
                else "false" if best_iou < FALSE_IOU else "part"
            )
            crop = image.crop((c.box.x0, c.box.y0, c.box.x1, c.box.y1))
            records.append(
                {
                    "photo": name,
                    "kind": kind,
                    "vector": vector,
                    "good_nb": c.quality_ok,
                    "blur": blur_score(crop),
                    "box": c.box,
                    "det": c.det_score,
                }
            )
    print(
        f"Photos: {len(images)}. Positive photos: {positives}. Candidates: {len(records)} "
        f"({sum(r['kind']=='hit' for r in records)} hits, {sum(r['kind']=='false' for r in records)} false, "
        f"{sum(r['kind']=='part' for r in records)} partial)\n"
    )

    def passes(record: dict, blur_min: float) -> bool:
        """The quality gates of the application, with a chosen limit for the blur of the crop."""
        return bool(record["good_nb"] and record["blur"] >= blur_min)

    def sim(record: dict, views: np.ndarray) -> float:
        return float((views @ record["vector"]).max())

    # ---- C1: the similarity of every candidate against the views of ALL positive photos but its own.
    print(
        "C1. Similarity of the labelled bottle in each positive photo to the object made from the OTHER positive photos\n"
    )
    print(
        "   photo  gates  crop_blur   similarity   (the best hit candidate of the photo)"
    )
    for name in positives:
        others = np.vstack([views_of[p] for p in positives if p != name])
        hits = [r for r in records if r["photo"] == name and r["kind"] == "hit"]
        best = max(hits, key=lambda r: sim(r, others))
        print(
            f"   {name:>5}  {'ok' if passes(best, base.quality.crop_blur_min) else 'FAIL':<5}  {best['blur']:>8.1f}   {sim(best, others):>9.3f}"
        )
    print(
        "\n   the 12 false candidates that look most like the object (made from all positive photos):"
    )
    all_views = np.vstack(list(views_of.values()))
    falses = sorted(
        (r for r in records if r["kind"] == "false"), key=lambda r: -sim(r, all_views)
    )
    print("   photo  similarity  gates  crop_blur  box")
    for r in falses[:12]:
        b = r["box"]
        print(
            f"   {r['photo']:>5}  {sim(r, all_views):>9.3f}  {'ok' if passes(r, base.quality.crop_blur_min) else 'FAIL':<5}  {r['blur']:>8.1f}  {b.width}x{b.height}"
        )

    # ---- C2: the leave-out test.
    def evaluate(
        tau_of: callable,
        blur_min: float = base.quality.crop_blur_min,
        offset: float = 0.0,
    ) -> dict:
        recalls, recalls_gated, fa, fa_gated, taus = [], [], [], [], []
        strong_true, strong_false = [], []
        for k in (3, 4):
            for subset in itertools.combinations(positives, k):
                views = np.vstack([views_of[p] for p in subset])
                tau = tau_of(subset)
                taus.append(tau)
                for held in (p for p in positives if p not in subset):
                    hits = [
                        r for r in records if r["photo"] == held and r["kind"] == "hit"
                    ]
                    best = max((sim(r, views) for r in hits), default=-1.0)
                    best_gated = max(
                        (sim(r, views) for r in hits if passes(r, blur_min)),
                        default=-1.0,
                    )
                    recalls.append(best >= tau)
                    recalls_gated.append(best_gated >= tau)
                    strong_true.append(best_gated >= tau + offset)
                fa.append(
                    sum(sim(r, views) >= tau for r in records if r["kind"] == "false")
                )
                fa_gated.append(
                    sum(
                        sim(r, views) >= tau
                        for r in records
                        if r["kind"] == "false" and passes(r, blur_min)
                    )
                )
                strong_false.append(
                    sum(
                        sim(r, views) >= tau + offset
                        for r in records
                        if r["kind"] == "false" and passes(r, blur_min)
                    )
                )
        return {
            "tau": float(np.mean(taus)),
            "tau_min": min(taus),
            "tau_max": max(taus),
            "recall": float(np.mean(recalls)),
            "recall_gated": float(np.mean(recalls_gated)),
            "fa": float(np.mean(fa)),
            "fa_max": max(fa),
            "fa_gated": float(np.mean(fa_gated)),
            "strong_true": float(np.mean(strong_true)),
            "strong_false": float(np.mean(strong_false)),
        }

    def formula(factor: float) -> callable:
        # The formula alone: the floor is switched off here, and swept in C3.
        config = dataclasses.replace(
            base.enrollment, tau_std_factor=factor, tau_min=0.0
        )

        def tau_of(subset: tuple[str, ...]) -> float:
            views = np.vstack([views_of[p] for p in subset])
            ids = [i for i, p in enumerate(subset) for _ in views_of[p]]
            return calibrate(
                views, ids, np.empty((0, views.shape[1]), dtype=np.float32), config
            ).tau

        return tau_of

    print(
        "\nC2. Leave-out test. The threshold `tau` comes from the formula: mean - factor * std of the similarities"
    )
    print("    between views of different photos. (The application uses factor 0.5.)")
    print(
        "    subsets of 3 and 4 positive photos; recall = held-out bottles that reach tau;"
    )
    print(
        "    false accepts = false candidates (all photos) that reach tau, per subset.\n"
    )
    print(
        "    factor   mean tau (min..max)    recall   recall_with_gates   false_accepts (max)   false_accepts_with_gates"
    )
    for factor in FACTORS:
        r = evaluate(formula(factor))
        mark = (
            "  <- the present value" if factor == base.enrollment.tau_std_factor else ""
        )
        print(
            f"    {factor:>5.1f}   {r['tau']:.3f} ({r['tau_min']:.3f}..{r['tau_max']:.3f})   {r['recall']:>6.2f}   {r['recall_gated']:>10.2f}   "
            f"{r['fa']:>10.2f} ({r['fa_max']})   {r['fa_gated']:>10.2f}{mark}"
        )
    print("\n    for comparison, a fixed threshold for every object:\n")
    print(
        "    tau      recall   recall_with_gates   false_accepts (max)   false_accepts_with_gates"
    )
    for tau in FIXED:
        r = evaluate(lambda subset, tau=tau: tau)
        print(
            f"    {tau:>5.2f}   {r['recall']:>6.2f}   {r['recall_gated']:>10.2f}   {r['fa']:>10.2f} ({r['fa_max']})   {r['fa_gated']:>10.2f}"
        )

    base_tau = formula(base.enrollment.tau_std_factor)
    print(
        "\nC3. A floor under `tau`: tau = max(formula, floor). The formula alone is unstable (see C2)."
    )
    print(
        "    A strong sighting needs tau + the strong offset (the application uses 0.05)."
    )
    print(
        "    A false strong sighting changes the last-seen record at once, so the numbers on the right matter most.\n"
    )
    print(
        "    floor   offset   recall   recall_with_gates   false_accepts_with_gates   FALSE STRONG (with gates)   true strong"
    )
    for floor in (0.0, 0.40, 0.45, 0.50):
        for offset in (0.05, 0.10, 0.15):
            r = evaluate(
                lambda subset, floor=floor: max(base_tau(subset), floor), offset=offset
            )
            mark = (
                "  <- the present values"
                if floor == base.enrollment.tau_min
                and offset == base.enrollment.tau_strong_offset
                else ""
            )
            print(
                f"    {floor:>5.2f}   {offset:>5.2f}   {r['recall']:>6.2f}   {r['recall_gated']:>10.2f}   {r['fa_gated']:>14.2f}   {r['strong_false']:>20.2f}   {r['strong_true']:>12.2f}{mark}"
            )

    print(
        "\nC4. The blur limit of the crop (`quality.crop_blur_min`), with tau = max(formula, 0.45) and a strong offset of 0.05\n"
    )
    print(
        "    crop_blur_min   recall_with_gates   false_accepts_with_gates   false strong"
    )
    for blur_min in (0.0, 5.0, 10.0, 20.0, 50.0):
        r = evaluate(
            lambda subset: max(base_tau(subset), 0.45), blur_min=blur_min, offset=0.05
        )
        mark = (
            "  <- the present value" if blur_min == base.quality.crop_blur_min else ""
        )
        print(
            f"    {blur_min:>11.0f}   {r['recall_gated']:>12.2f}   {r['fa_gated']:>20.2f}   {r['strong_false']:>10.2f}{mark}"
        )


if __name__ == "__main__":
    main()
