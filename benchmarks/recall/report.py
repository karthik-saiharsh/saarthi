"""Read the signal store of `collect.py` and print the recall reports. No model runs, so a change is tested in seconds.

Sections:
1. Boxes: which detector found each true object.
2. Today: the step where each true object was lost, in the modes `evidence` and `tau`, and the false sightings.
3. Signals: the false-accept rate of each signal at the bar that keeps all true objects (needs the truth).
4. Cohort: the bars of each object from a false-alarm budget (needs no truth).
5. Replay: the two-key rule of the plan with given bars, at frame level, against today.

Usage:
    uv run python benchmarks/recall/report.py SET_JSON STORE_DIR [--boxes union|dedup|gdino] [--dino cls|pool_mask]
        [--sig-strong 0.77] [--dino-strong 0.42] [--sig-weak 0.72] [--word-vote 0.3]
        [--dino-weak 0.40] [--negatives none|sig|dino|both] [--neg-margin 0.0]
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    COVER_IGNORE,
    IOU_SAME,
    FrameSet,
    cover,
    iou,
    load_set,
    load_store,
)

logger: logging.Logger = logging.getLogger("recall.report")

# The agreement of a box with an object is the mean of its 2 best reference photos (the rule `top_photos` of today).
TOP_PHOTOS: int = 2
# The budgets of section 4: the share of cohort boxes that may pass a bar.
BUDGETS: tuple[float, ...] = (1e-2, 1e-3)
SIGNALS: tuple[str, ...] = ("pooled", "sig", "cls", "pool_mask", "pool_attn", "word")


@dataclass
class Data:
    """The store with the labels of each box for each object."""

    set: FrameSet
    boxes: list[dict]
    frames: list[dict]
    meta: dict
    frame: np.ndarray  # the frame key of each box
    cohort: np.ndarray  # True: a box of a cohort frame
    source: np.ndarray  # "gdino" or "yoloe"
    quality_ok: np.ndarray
    score: dict[str, dict[str, np.ndarray]]  # object -> signal -> one value per box
    role: dict[str, np.ndarray]  # object -> "true", "hard", "ignore" or "neg" per box
    instance: dict[
        str, np.ndarray
    ]  # object -> the index of the truth instance per box, -1 for none
    pos_best: dict[
        str, dict[str, np.ndarray]
    ]  # object -> key -> the best reference cosine per box
    neg_best: dict[
        str, dict[str, np.ndarray]
    ]  # object -> key -> the best negative cosine per box (`negatives.py`)


def top_mean(similarity: np.ndarray, top: int) -> np.ndarray:
    """Rows: boxes, columns: reference photos. The mean of the `top` best photos for each box."""
    ordered: np.ndarray = np.sort(similarity, axis=1)[:, ::-1]
    return ordered[:, :top].mean(axis=1)


def load(set_path: Path, store: Path) -> Data:
    frame_set: FrameSet = load_set(set_path)
    boxes, vectors, references, reference_vectors, frames, meta = load_store(store)
    assert boxes, "the store has no box"
    frame: np.ndarray = np.array([b["frame"] for b in boxes])
    cohort: np.ndarray = np.array([key in frame_set.cohort for key in frame])
    ref_object: np.ndarray = np.array([r["object"] for r in references])
    score: dict[str, dict[str, np.ndarray]] = {}
    role: dict[str, np.ndarray] = {}
    instance: dict[str, np.ndarray] = {}
    pos_best: dict[str, dict[str, np.ndarray]] = {}
    neg_best: dict[str, dict[str, np.ndarray]] = {}
    negatives: dict[str, np.ndarray] = {}
    neg_object: np.ndarray = np.array([])
    if (store / "negatives.npz").is_file():
        negatives = dict(np.load(store / "negatives.npz"))
        neg_object = np.array(
            [r["object"] for r in json.loads((store / "negatives.json").read_text())]
        )
    for name in meta["objects"]:
        mine: np.ndarray = ref_object == name
        pos_best[name] = {
            k: (vectors[k] @ reference_vectors[k][mine].T).max(axis=1)
            for k in ("sig", "cls", "pool_mask")
        }
        if (neg_object == name).any():
            neg_best[name] = {
                k: (vectors[k] @ negatives[k][neg_object == name].T).max(axis=1)
                for k in ("sig", "cls", "pool_mask")
            }
        score[name] = {
            k: top_mean(vectors[k] @ reference_vectors[k][mine].T, TOP_PHOTOS)
            for k in vectors
        }
        score[name]["word"] = np.array([b["word"][name] for b in boxes])
        roles: np.ndarray = np.array(["neg"] * len(boxes), dtype=object)
        index: np.ndarray = -np.ones(len(boxes), dtype=int)
        number: int = 0
        for kind in ("true", "hard"):
            for key, truth_box in frame_set.truth.get(name, {}).get(kind, []):
                for i in np.where(frame == key)[0]:
                    box = tuple(boxes[i]["box"])
                    if iou(box, truth_box) >= IOU_SAME:
                        roles[i], index[i] = kind, number
                    elif roles[i] == "neg" and cover(box, truth_box) >= COVER_IGNORE:
                        roles[i] = "ignore"
                number += 1
        role[name], instance[name] = roles, index
    return Data(
        frame_set,
        boxes,
        frames,
        meta,
        frame,
        cohort,
        np.array([b["source"] for b in boxes]),
        np.array([b["quality_ok"] for b in boxes]),
        score,
        role,
        instance,
        pos_best,
        neg_best,
    )


def instances(data: Data, name: str, kind: str) -> list[tuple[int, str, tuple]]:
    """(index, frame key, box) of each truth instance of one kind, with the numbering of `load`."""
    truth = data.set.truth.get(name, {})
    offset: int = 0 if kind == "true" else len(truth.get("true", []))
    return [(offset + i, key, box) for i, (key, box) in enumerate(truth.get(kind, []))]


def full_recall_bar(values: np.ndarray, data: Data, name: str) -> float:
    """The lowest value of the best box of each true instance. Every true instance with a box reaches it."""
    best: list[float] = [
        float(values[data.instance[name] == i].max())
        for i, _, _ in instances(data, name, "true")
        if (data.instance[name] == i).any()
    ]
    return min(best) if best else float("inf")


# ---- 1. Boxes ----


def section_boxes(data: Data) -> None:
    print(
        "\n1. Boxes. For each truth instance: G = today's detector (Grounding DINO with the gallery phrases), Y = YOLOE"
    )
    totals: dict[str, list[int]] = {"G": [0, 0], "Y": [0, 0], "union": [0, 0]}
    for name in data.meta["objects"]:
        cells: list[str] = []
        for kind in ("true", "hard"):
            for i, key, _ in instances(data, name, kind):
                hit: np.ndarray = data.instance[name] == i
                g: bool = bool((hit & (data.source == "gdino")).any())
                y: bool = bool((hit & (data.source == "yoloe")).any())
                cells.append(f"{key}{kind[0]}:{'G' if g else '-'}{'Y' if y else '-'}")
                if kind == "true":
                    for tag, found in (("G", g), ("Y", y), ("union", g or y)):
                        totals[tag][0] += found
                        totals[tag][1] += 1
        print(f"  {name:8s} " + " ".join(cells))
    print(
        "  true instances found: "
        + ", ".join(f"{k} {a}/{b}" for k, (a, b) in totals.items())
    )


# ---- 2. Today ----


def section_today(data: Data) -> dict[str, dict[int, str]]:
    """Print the funnel of today. Return the outcome of each true instance for each mode."""
    print(
        "\n2. Today's pipeline (all objects enrolled together). The step where each truth instance ends"
    )
    print(
        "   calibration: "
        + ", ".join(
            f"{n} tau {c['tau']:.3f}" for n, c in data.meta["calibration"].items()
        )
    )
    rows_of: dict[str, list[int]] = {}
    for record in data.frames:
        rows_of[record["key"]] = [
            record["first_row"] + i
            for i in range(record["rows"])
            if data.boxes[record["first_row"] + i]["source"] == "gdino"
        ]
    by_key: dict[str, dict] = {r["key"]: r for r in data.frames}
    outcome: dict[str, dict[int, str]] = {"evidence": {}, "tau": {}}
    for mode in ("evidence", "tau"):
        print(f"  mode {mode}")
        counts: dict[str, int] = {}
        for name in data.meta["objects"]:
            cells: list[str] = []
            for kind in ("true", "hard"):
                for i, key, truth_box in instances(data, name, kind):
                    step: str = (
                        today_step(
                            data, by_key[key], rows_of[key], name, i, truth_box, mode
                        )
                        if key in by_key
                        else "notrun"
                    )
                    cells.append(f"{key}{kind[0]}:{step}")
                    if kind == "true":
                        outcome[mode][(name, i)] = step
                        counts[step] = counts.get(step, 0) + 1
            print(f"    {name:8s} " + " ".join(cells))
        print(
            "    true instances: "
            + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        )
        home, cohort = false_today(data, by_key, rows_of, mode)
        print(
            f"    false sightings: home {home}, cohort {cohort} (frames {sum(not data.set.cohort.get(r['key']) for r in data.frames)} home, {sum(bool(data.set.cohort.get(r['key'])) for r in data.frames)} cohort)"
        )
    return outcome


def on_instance(
    data: Data, row: int, name: str, instance: int, truth_box: tuple
) -> bool:
    """True if a sighting on this box shows the right place: the box is the instance, or a part of it."""
    if data.instance[name][row] == instance:
        return True
    return (
        data.role[name][row] == "ignore"
        and cover(tuple(data.boxes[row]["box"]), truth_box) >= COVER_IGNORE
    )


def today_step(
    data: Data,
    record: dict,
    rows: list[int],
    name: str,
    instance: int,
    truth_box: tuple,
    mode: str,
) -> str:
    """STRONG, weak, or the step that lost the instance: nobox, quality, notchosen (evidence only), below."""
    hits: list[int] = [r for r in rows if data.instance[name][r] == instance]
    result: dict | None = record[mode]
    for s in (result or {}).get("sightings", []):
        row: int = rows[s["candidate"]]
        if s["object"] == name and on_instance(data, row, name, instance, truth_box):
            return "STRONG" if s["strong"] else "weak"
    if not hits:
        return "nobox"
    if not any(data.quality_ok[r] for r in hits):
        return "quality"
    if mode == "evidence" and not any(rows.index(r) in result["chosen"] for r in hits):
        return "notchosen"
    return "below"


def false_today(data: Data, by_key: dict, rows_of: dict, mode: str) -> tuple[int, int]:
    home: int = 0
    cohort: int = 0
    for key, record in by_key.items():
        for s in (record[mode] or {}).get("sightings", []):
            row: int = rows_of[key][s["candidate"]]
            if data.role[s["object"]][row] == "neg":
                if key in data.set.cohort:
                    cohort += 1
                else:
                    home += 1
    return home, cohort


# ---- 3. Signals ----


def section_signals(data: Data) -> None:
    print(
        "\n3. Signals. False accepts at the bar that keeps all true instances (home boxes / cohort boxes)"
    )
    print("   " + " ".join(f"{s:>16s}" for s in ("object",) + SIGNALS))
    totals: dict[str, list[int]] = {s: [0, 0] for s in SIGNALS}
    for name in data.meta["objects"]:
        negative: np.ndarray = data.role[name] == "neg"
        cells: list[str] = []
        for signal in SIGNALS:
            values: np.ndarray = data.score[name][signal]
            bar: float = full_recall_bar(values, data, name)
            home: int = int((values[negative & ~data.cohort] >= bar).sum())
            cohort: int = int((values[negative & data.cohort] >= bar).sum())
            totals[signal][0] += home
            totals[signal][1] += cohort
            cells.append(f"{home}/{cohort}")
        print("   " + f"{name:>16s}" + " ".join(f"{c:>16s}" for c in cells))
    print(
        "   "
        + f"{'all':>16s}"
        + " ".join(f"{f'{a}/{b}':>16s}" for a, b in totals.values())
    )
    negatives: int = int(
        sum(
            ((data.role[n] == "neg") & ~data.cohort).sum() for n in data.meta["objects"]
        )
    )
    cohort_boxes: int = int(data.cohort.sum())
    print(
        f"   negatives: {negatives} home boxes (over all objects), {cohort_boxes} cohort boxes for each object"
    )


# ---- 4. Cohort ----


def cohort_bar(data: Data, name: str, signal: str, budget: float) -> float:
    """The value that only `budget` of the cohort boxes reach. It needs no truth."""
    values: np.ndarray = data.score[name][signal][data.cohort]
    assert len(values), "the store has no cohort box"
    return float(np.quantile(values, 1.0 - budget))


def section_cohort(data: Data) -> None:
    print(
        f"\n4. Cohort ({int(data.cohort.sum())} boxes of frames with no registered object). The bar for each false-alarm budget"
    )
    for signal in ("sig", "cls", "pool_mask"):
        print(f"   {signal}")
        for name in data.meta["objects"]:
            values: np.ndarray = data.score[name][signal][data.cohort]
            bars: str = "  ".join(
                f"budget {b:g}: {cohort_bar(data, name, signal, b):.3f}"
                for b in BUDGETS
            )
            true_bar: float = full_recall_bar(data.score[name][signal], data, name)
            print(
                f"     {name:8s} max {values.max():.3f}  {bars}  | lowest true {true_bar:.3f}"
            )


# ---- 5. Replay ----


@dataclass(frozen=True)
class Bars:
    sig_strong: float
    dino_strong: float
    sig_weak: float
    word_vote: float
    dino: str
    boxes: str
    dino_weak: float | None = (
        None  # None: the 1 % cohort bar of each object. A number: one fixed bar, no cohort
    )
    negatives: str = (
        "none"  # the keys with the negatives rule: "none", "sig", "dino" or "both"
    )
    neg_margin: float = 0.0


def duplicates(data: Data) -> np.ndarray:
    """True for a YOLOE box with an IoU of 0.5 or more with a Grounding DINO box of its frame (the runtime keeps one)."""
    result: np.ndarray = np.zeros(len(data.boxes), bool)
    for key in dict.fromkeys(data.frame):
        rows: np.ndarray = np.where(data.frame == key)[0]
        gdino = [tuple(data.boxes[r]["box"]) for r in rows if data.source[r] == "gdino"]
        for r in rows:
            if data.source[r] == "yoloe":
                box = tuple(data.boxes[r]["box"])
                result[r] = any(iou(box, g) >= IOU_SAME for g in gdino)
    return result


def replay(data: Data, bars: Bars) -> dict[str, list[tuple[str, int, int, bool]]]:
    """The two-key rule at frame level. Return for each frame the sightings: (object, row, strong)."""
    use: np.ndarray = (
        np.ones(len(data.boxes), bool)
        if bars.boxes == "union"
        else data.source == "gdino"
    )
    if bars.boxes == "dedup":
        use = ~duplicates(data)
    names: list[str] = data.meta["objects"]
    level: dict[str, np.ndarray] = {}  # 2 strong, 1 weak, 0 nothing
    for name in names:
        s: np.ndarray = data.score[name]["sig"]
        d: np.ndarray = data.score[name][bars.dino]
        dino_p99: float = (
            cohort_bar(data, name, bars.dino, BUDGETS[0])
            if bars.dino_weak is None
            else bars.dino_weak
        )
        strong: np.ndarray = (s >= bars.sig_strong) & (d >= bars.dino_strong)
        weak: np.ndarray = (s >= bars.sig_weak) & (
            (d >= dino_p99) | (data.score[name]["word"] >= bars.word_vote)
        )
        result: np.ndarray = np.where(strong, 2, np.where(weak, 1, 0))
        # The negatives rule: a box that is closer to a negative of the reference photos than to the object is refused.
        keys: list[str] = {"none": [], "sig": ["sig"], "dino": [bars.dino]}.get(
            bars.negatives, ["sig", bars.dino]
        )
        for key in keys:
            if name in data.neg_best and key in data.neg_best[name]:
                closer: np.ndarray = (
                    data.pos_best[name][key] - data.neg_best[name][key]
                    < bars.neg_margin
                )
                result = np.where(closer, 0, result)
        # Blur is a cap, not a veto: a box that fails the crop quality can be weak at most.
        result = np.where(~data.quality_ok & (result == 2), 1, result)
        level[name] = np.where(use, result, 0)
    sightings: dict[str, list] = {}
    for key in dict.fromkeys(data.frame):
        rows: np.ndarray = np.where(data.frame == key)[0]
        found: list = []
        for name in names:
            passing = [r for r in rows if level[name][r] > 0]
            # One box is one object: the object with the highest SigLIP agreement keeps it.
            passing = [
                r
                for r in passing
                if all(
                    data.score[name]["sig"][r] >= data.score[o]["sig"][r]
                    or level[o][r] == 0
                    for o in names
                    if o != name
                )
            ]
            if passing:
                best: int = max(
                    passing, key=lambda r: (level[name][r], data.score[name]["sig"][r])
                )
                found.append((name, int(best), level[name][best] == 2))
        sightings[key] = found
    return sightings


def section_replay(data: Data, bars: Bars, today: dict[str, dict]) -> None:
    print(f"\n5. Replay of the two-key rule: {bars}")
    sightings = replay(data, bars)
    counts: dict[str, int] = {}
    for name in data.meta["objects"]:
        cells: list[str] = []
        for kind in ("true", "hard"):
            for i, key, truth_box in instances(data, name, kind):
                step: str = "-"
                if key not in sightings:
                    step = "notrun"
                elif not (data.instance[name] == i).any():
                    step = "nobox"
                for obj, row, strong in sightings.get(key, []):
                    if obj == name and on_instance(data, row, name, i, truth_box):
                        step = "STRONG" if strong else "weak"
                cells.append(f"{key}{kind[0]}:{step}")
                if kind == "true":
                    counts[step] = counts.get(step, 0) + 1
        print(f"  {name:8s} " + " ".join(cells))
    home: int = 0
    cohort: int = 0
    # object -> [strong home, strong cohort, weak home, weak cohort]
    split: dict[str, list[int]] = {n: [0, 0, 0, 0] for n in data.meta["objects"]}
    for key, found in sightings.items():
        for obj, row, strong in found:
            if data.role[obj][row] == "neg":
                in_cohort: bool = key in data.set.cohort
                cohort += in_cohort
                home += not in_cohort
                split[obj][(0 if strong else 2) + in_cohort] += 1
    print(
        "  true instances: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    print(f"  false sightings: home {home}, cohort {cohort}")
    for name, (sh, sc, wh, wc) in split.items():
        print(f"    {name:8s} strong {sh} / {sc}, weak {wh} / {wc}")
    print(
        "\nSummary (true instances: strong + weak = found, of all; false sightings home / cohort)"
    )
    for mode in ("evidence", "tau"):
        steps: list[str] = list(today[mode].values())
        print(
            f"  today {mode:8s}: {steps.count('STRONG')} + {steps.count('weak')} = {steps.count('STRONG') + steps.count('weak')} of {len(steps)}"
        )
    print(
        f"  two keys      : {counts.get('STRONG', 0)} + {counts.get('weak', 0)} = {counts.get('STRONG', 0) + counts.get('weak', 0)} of {sum(counts.values())}, false {home} / {cohort}"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("set_file", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("--boxes", choices=["union", "dedup", "gdino"], default="union")
    parser.add_argument("--dino", choices=["cls", "pool_mask"], default="cls")
    parser.add_argument("--sig-strong", type=float, default=0.77)
    parser.add_argument("--dino-strong", type=float, default=0.42)
    parser.add_argument("--sig-weak", type=float, default=0.72)
    parser.add_argument("--word-vote", type=float, default=0.3)
    parser.add_argument(
        "--dino-weak",
        type=float,
        default=None,
        help="one fixed DINOv2 bar of the weak rule (no cohort). Default: the 1 %% cohort bar",
    )
    parser.add_argument(
        "--negatives", choices=["none", "sig", "dino", "both"], default="none"
    )
    parser.add_argument("--neg-margin", type=float, default=0.0)
    args = parser.parse_args()
    data: Data = load(args.set_file, args.store)
    section_boxes(data)
    today = section_today(data)
    section_signals(data)
    if data.cohort.any():
        section_cohort(data)
        section_replay(
            data,
            Bars(
                args.sig_strong,
                args.dino_strong,
                args.sig_weak,
                args.word_vote,
                args.dino,
                args.boxes,
                args.dino_weak,
                args.negatives,
                args.neg_margin,
            ),
            today,
        )
    else:
        logger.warning("The store has no cohort frame. Sections 4 and 5 need it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
