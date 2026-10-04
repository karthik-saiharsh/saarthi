"""Calibration of the detector and of the quality gates, from the cache of `collect.py`.

Usage: uv run python benchmarks/calibration/analyse_detection.py DATASET_DIR CACHE_DIR

It needs no model. It replays the cached detector output with other settings and
compares the candidates with the labels in `labels.json`. A candidate is a hit if it
has an IoU of at least 0.5 with a labelled box, and a false box if it has an IoU of
less than 0.1 with all labelled boxes. Other candidates (parts of the object) count as neither.
This script is not part of the runtime.
"""

import json
import sys
from pathlib import Path

from common import candidates_for, iou, load_image, load_raw, with_detection

from ai.geometry import touches_edge
from ai.quality import blur_score
from ai.settings import load_config
from ai.types import Box

HIT_IOU: float = 0.5
FALSE_IOU: float = 0.1
THRESHOLDS: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
PASSES: dict[str, tuple[str, ...]] = {
    "full only (the default frame)": ("full",),
    "full + generic": ("full", "generic"),
    "full + tile + generic": ("full", "tile", "generic"),
}


def main() -> None:
    dataset: Path = Path(sys.argv[1])
    cache: Path = Path(sys.argv[2])
    raw: dict = load_raw(cache)
    labels: dict[str, list[list[int]]] = json.loads(
        (Path(__file__).parent / "labels.json").read_text()
    )["boxes"]
    base = load_config()
    images = {name: load_image(dataset, name) for name in raw["images"]}
    truth: dict[str, list[Box]] = {
        n: [Box(*b) for b in boxes] for n, boxes in labels.items()
    }
    total_truth: int = sum(len(v) for v in truth.values())

    print(
        f"Images: {len(images)}. Labelled bottles: {total_truth}. Default box threshold: {base.detection.box_threshold}\n"
    )
    print(
        "A. Detector: how many labelled bottles get a box (hit), and how many false boxes there are"
    )
    print("   (box threshold sweep; the text threshold of the cache is 0.15)\n")
    for label, sources in PASSES.items():
        print(f"   {label}")
        print(
            "   box_thr  hits  hits_that_pass_the_quality_gates  false_boxes  false_boxes_per_image"
        )
        for threshold in THRESHOLDS:
            config = with_detection(base, box_threshold=threshold)
            hits = good_hits = false = 0
            for name, image in images.items():
                candidates = candidates_for(
                    image, raw["images"][name], raw["phrases"], config, sources
                )
                for box in truth[name]:
                    best = max(candidates, key=lambda c: iou(c.box, box), default=None)
                    if best is not None and iou(best.box, box) >= HIT_IOU:
                        hits += 1
                        good_hits += best.quality_ok
                false += sum(
                    1
                    for c in candidates
                    if all(iou(c.box, b) < FALSE_IOU for b in truth[name])
                )
            print(
                f"   {threshold:>5.2f}   {hits:>2}/{total_truth}   {good_hits:>2}/{total_truth}   {false:>11}   {false / len(images):>8.1f}"
            )
        print()

    print(
        "B. The quality gates on the labelled bottles (default thresholds, passes: full + generic)\n"
    )
    config = base
    q = config.quality
    print(
        f"   gates: min_crop_side {q.min_crop_side}, crop_blur_min {q.crop_blur_min}, edge rule on, depiction rule on"
    )
    print("   photo  box_size  crop_blur  reasons")
    for name, image in images.items():
        candidates = candidates_for(
            image, raw["images"][name], raw["phrases"], config, ("full", "generic")
        )
        for box in truth[name]:
            best = max(candidates, key=lambda c: iou(c.box, box), default=None)
            if best is None or iou(best.box, box) < HIT_IOU:
                print(f"   {name:>5}  (no box with IoU >= {HIT_IOU})")
                continue
            b = best.box
            crop_blur = blur_score(image.crop((b.x0, b.y0, b.x1, b.y1)))
            reasons = []
            if min(b.width, b.height) < q.min_crop_side:
                reasons.append("small")
            if touches_edge(b, image.width, image.height):
                reasons.append("EDGE")
            if crop_blur < q.crop_blur_min:
                reasons.append("BLUR")
            if not best.quality_ok and not reasons:
                reasons.append("DEPICTION")
            print(
                f"   {name:>5}  {b.width}x{b.height:<5}  {crop_blur:>8.1f}  {', '.join(reasons) or 'passes'}"
            )

    print("\n   crop blur of the false boxes (the lower, the more blurred):")
    values = []
    for name, image in images.items():
        for c in candidates_for(
            image, raw["images"][name], raw["phrases"], config, ("full", "generic")
        ):
            if all(iou(c.box, b) < FALSE_IOU for b in truth[name]):
                values.append(
                    blur_score(image.crop((c.box.x0, c.box.y0, c.box.x1, c.box.y1)))
                )
    values.sort()
    if values:
        print(
            f"   {len(values)} false boxes. min {values[0]:.0f}, 25% {values[len(values)//4]:.0f}, median {values[len(values)//2]:.0f}, 75% {values[3*len(values)//4]:.0f}, max {values[-1]:.0f}"
        )
        print(
            f"   below the present minimum of {q.crop_blur_min:.0f}: {sum(v < q.crop_blur_min for v in values)} of {len(values)}"
        )

    print(
        "\nC. Which pass and which score finds each labelled bottle (default thresholds otherwise)\n"
    )
    print(
        "   photo  best IoU box: score and pass          (a hit needs the score to be at least the box threshold)"
    )
    for name, image in images.items():
        for box in truth[name]:
            row = []
            for source in ("full", "generic", "tile"):
                found = candidates_for(
                    image,
                    raw["images"][name],
                    raw["phrases"],
                    with_detection(base, box_threshold=0.15),
                    (source,),
                )
                best = max(found, key=lambda c: iou(c.box, box), default=None)
                row.append(
                    f"{source}: {best.det_score:.2f}"
                    if best is not None and iou(best.box, box) >= HIT_IOU
                    else f"{source}: none"
                )
            print(f"   {name:>5}  " + "   ".join(row))

    print(
        "\nD. NMS (`nms_iou`): boxes per image and hits (box threshold 0.25, passes full + generic)\n"
    )
    print("   nms_iou  boxes_per_image  hits  false_boxes")
    for nms in (0.3, 0.5, 0.6, 0.7, 0.8):
        config = with_detection(base, nms_iou=nms)
        boxes = hits = false = 0
        for name, image in images.items():
            candidates = candidates_for(
                image, raw["images"][name], raw["phrases"], config, ("full", "generic")
            )
            boxes += len(candidates)
            for box in truth[name]:
                best = max(candidates, key=lambda c: iou(c.box, box), default=None)
                hits += best is not None and iou(best.box, box) >= HIT_IOU
            false += sum(
                1
                for c in candidates
                if all(iou(c.box, b) < FALSE_IOU for b in truth[name])
            )
        print(
            f"   {nms:>5.1f}   {boxes / len(images):>10.1f}        {hits}/{total_truth}   {false:>6}"
        )

    print(
        "\nE. The blur of the whole frame (`quality.blur_min`) against the frames that show the bottle\n"
    )
    print("   photo  blur_score  has_bottle")
    for name in sorted(images, key=int):
        print(
            f"   {name:>5}  {raw['images'][name]['blur_score']:>9.0f}   {'yes' if truth[name] else ''}"
        )


if __name__ == "__main__":
    main()
