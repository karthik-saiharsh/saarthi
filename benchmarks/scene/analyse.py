"""Tally the rooms of the results of `compare.py` against `labels.json`.

Usage: uv run python benchmarks/scene/analyse.py OUT_DIR

It reads `scene_lists.json` and `scene_detector.json` from OUT_DIR. It reports:
- the rooms that the fixed lists give for one photo, and when the scores of the neighbouring photos are averaged,
- the room claims of the detector method for several score cutoffs.
The neighbours are the photos with the next numbers. The numbers may not be the order in time. This is an illustration.
This script is not part of the runtime.
"""

import json
import sys
from pathlib import Path

ROOM_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("a bathroom", ("a toilet", "a bathtub")),
    ("a kitchen", ("a refrigerator", "a stove")),
    ("a bedroom", ("a bed",)),
    ("a living room", ("a sofa", "a television")),
    ("a study room", ("a desk", "a laptop")),
]


def main() -> None:
    out: Path = Path(sys.argv[1])
    accepted = json.loads((Path(__file__).parent / "labels.json").read_text())["rooms"]

    lists = json.loads((out / "scene_lists.json").read_text())
    names = sorted(lists, key=int)
    rooms = list(lists[names[0]]["room_all"])

    def best(scores: dict[str, float]) -> str:
        return max(scores, key=scores.get)

    def averaged(index: int, width: int) -> dict[str, float]:
        low, high = max(0, index - width), min(len(names) - 1, index + width)
        return {
            room: sum(lists[names[j]]["room_all"][room] for j in range(low, high + 1))
            / (high - low + 1)
            for room in rooms
        }

    print(
        "Fixed lists: the room of each photo (the score of the best room in brackets)"
    )
    print("  photo  one photo                 +-1 neighbours")
    for i, name in enumerate(names):
        one, near = best(lists[name]["room_all"]), best(averaged(i, 1))
        print(
            f"  {name:>5}  {one:<18}{lists[name]['room_all'][one]:>5.2f}  {'WRONG' if one not in accepted[name] else '':<6}"
            f" {near:<18} {'WRONG' if near not in accepted[name] else ''}"
        )
    for width in (0, 1, 2):
        right = sum(
            best(averaged(i, width)) in accepted[n] for i, n in enumerate(names)
        )
        print(f"  window +-{width}: {right}/{len(names)} rooms right")

    detector_file = out / "scene_detector.json"
    if detector_file.exists():
        detector = json.loads(detector_file.read_text())
        print("\nDetector method: the room claims for several score cutoffs")
        for cutoff in (0.30, 0.35, 0.40, 0.45, 0.50):
            right = wrong = none = 0
            for name, entry in detector.items():
                evidence = {k: v for k, v in entry["evidence"].items() if v >= cutoff}
                room = next(
                    (r for r, needs in ROOM_RULES if any(n in evidence for n in needs)),
                    None,
                )
                if room is None:
                    none += 1
                elif room in accepted[name]:
                    right += 1
                else:
                    wrong += 1
            print(
                f"  cutoff {cutoff:.2f}: right {right}, WRONG {wrong}, no claim {none}"
            )


def rule_grid(out: Path) -> None:
    """Try the room rule of the backend with several settings (`room_min_frames`, agree fraction, `room_min_score`).

    The rule: the frames are the photo and its neighbours. The scores of the rooms are averaged. The best room is named
    only if enough frames have it as the best room, and its average score is high enough. Else the answer says nothing.
    """
    accepted = json.loads((Path(__file__).parent / "labels.json").read_text())["rooms"]
    lists = json.loads((out / "scene_lists.json").read_text())
    names = sorted(lists, key=int)
    rooms = list(lists[names[0]]["room_all"])
    print("\nThe room rule: right claims / WRONG claims / no claim, for 14 photos")
    for side in (1, 2):
        print(
            f"  neighbours: {side} photo(s) on each side (a photo near the end has fewer)"
        )
        print("  min_frames  agree  min_score   right  WRONG  none")
        for min_frames in (2, 3):
            for agree in (0.5, 0.6, 0.67, 0.75, 1.0):
                for min_score in (0.3, 0.4, 0.5, 0.6):
                    right = wrong = none = 0
                    for i, name in enumerate(names):
                        low, high = max(0, i - side), min(len(names) - 1, i + side)
                        frames = [
                            lists[names[j]]["room_all"] for j in range(low, high + 1)
                        ]
                        average = {
                            r: sum(f[r] for f in frames) / len(frames) for r in rooms
                        }
                        best = max(average, key=average.get)
                        agreeing = sum(max(f, key=f.get) == best for f in frames)
                        claims = (
                            len(frames) >= min_frames
                            and agreeing / len(frames) >= agree
                            and average[best] >= min_score
                        )
                        if not claims:
                            none += 1
                        elif best in accepted[name]:
                            right += 1
                        else:
                            wrong += 1
                    print(
                        f"  {min_frames:>10}  {agree:>5.2f}  {min_score:>8.1f}   {right:>5}  {wrong:>5}  {none:>4}"
                    )


if __name__ == "__main__":
    main()
    rule_grid(Path(sys.argv[1]))
