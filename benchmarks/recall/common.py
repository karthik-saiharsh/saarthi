"""Shared helpers of the recall benchmark: the set file, the store and the box geometry.

The set file names private frames. Keep it out of the repository. See README.md for its format.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT: Path = Path(__file__).resolve().parents[2]
logger: logging.Logger = logging.getLogger("recall")

# The vector kinds of the store. Each one has one row for each box and one row for each reference photo.
VECTOR_KINDS: tuple[str, ...] = (
    "pooled",  # Today: DINOv2 patches of the whole frame, pooled in the box (`AiEngine.embed_regions`)
    "sig",  # SigLIP image vector of the crop with `matching.crop_padding`
    "cls",  # DINOv2 CLS token of the tight crop at 224 px
    "pool_mask",  # DINOv2 patches of the tight crop, pooled inside the YOLOE mask
    "pool_attn",  # DINOv2 patches of the tight crop, pooled with the CLS attention (a simple form of Object-DINO)
)
# A box with an IoU of this value or more with a truth box shows the same thing.
IOU_SAME: float = 0.5
# A false box that covers this part of a truth box (or the reverse) is a part of the object. It is not a negative.
COVER_IGNORE: float = 0.5


BoxTuple = tuple[int, int, int, int]


@dataclass(frozen=True)
class ObjectSpec:
    """An object of the set: its detector phrases and its reference photos."""

    phrases: tuple[str, ...]
    references: tuple[tuple[str, BoxTuple], ...]  # (frame path, box)


@dataclass(frozen=True)
class FrameSet:
    """The content of a set file."""

    objects: dict[str, ObjectSpec]
    frames: dict[str, str]  # key -> path. Frames of the home, with truth
    cohort: dict[str, str]  # key -> path. Frames that show no registered object
    truth: dict[
        str, dict[str, list[tuple[str, BoxTuple]]]
    ]  # object -> "true"/"hard" -> (frame key, box)


def resolve(path: str) -> Path:
    """Return an absolute path. A relative path starts at the project folder."""
    assert isinstance(path, str) and path, "path must be a non-empty string"
    candidate: Path = Path(path)
    return candidate if candidate.is_absolute() else PROJECT / candidate


def _box(values: list) -> BoxTuple:
    assert len(values) == 4, "a box has 4 numbers"
    x0, y0, x1, y1 = (round(v) for v in values)
    assert x0 < x1 and y0 < y1, f"empty box {values}"
    return (x0, y0, x1, y1)


def load_set(path: Path) -> FrameSet:
    """Read and check a set file."""
    assert path.is_file(), f"no set file at {path}"
    raw: dict = json.loads(path.read_text())
    objects: dict[str, ObjectSpec] = {}
    for name, spec in raw["objects"].items():
        phrases: tuple[str, ...] = tuple(spec["phrases"])
        refs: tuple[tuple[str, BoxTuple], ...] = tuple(
            (r["image"], _box(r["box"])) for r in spec["references"]
        )
        assert phrases, f"object {name} needs a phrase"
        assert len(refs) >= 2, f"object {name} needs 2 reference photos or more"
        objects[name] = ObjectSpec(phrases, refs)
    frames: dict[str, str] = dict(raw["frames"])
    cohort: dict[str, str] = {}
    cohort_dir: str | None = raw.get("cohort_dir")
    if cohort_dir:
        folder: Path = resolve(cohort_dir)
        assert folder.is_dir(), f"no cohort folder at {folder}"
        files: list[Path] = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        cohort = {f"C{i}": str(p) for i, p in enumerate(files)}
    truth: dict[str, dict[str, list[tuple[str, BoxTuple]]]] = {}
    for name, kinds in raw.get("truth", {}).items():
        assert name in objects, f"truth for an unknown object {name}"
        truth[name] = {
            kind: [(frame, _box(box)) for frame, box in kinds.get(kind, [])]
            for kind in ("true", "hard")
        }
        for kind in truth[name].values():
            for frame, _ in kind:
                assert frame in frames, f"truth names an unknown frame {frame}"
    assert not set(frames) & set(cohort), "frame keys and cohort keys must differ"
    return FrameSet(objects, frames, cohort, truth)


def iou(a: BoxTuple, b: BoxTuple) -> float:
    """Intersection over union of two boxes."""
    w: int = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h: int = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter: int = w * h
    union: int = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cover(a: BoxTuple, b: BoxTuple) -> float:
    """The intersection as a part of the smaller box."""
    w: int = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h: int = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller: int = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller if smaller > 0 else 0.0


def save_store(
    out_dir: Path,
    boxes: list[dict],
    vectors: dict[str, list[np.ndarray]],
    references: list[dict],
    reference_vectors: dict[str, np.ndarray],
    frames: list[dict],
    meta: dict,
) -> None:
    """Write the store. A partial store is written in the same way, so a long run can stop and keep its work."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stacked: dict[str, np.ndarray] = {
        k: (np.concatenate(v).astype(np.float32) if v else np.zeros((0, 1), np.float32))
        for k, v in vectors.items()
    }
    for kind, array in stacked.items():
        assert len(array) == len(
            boxes
        ), f"{kind}: {len(array)} rows for {len(boxes)} boxes"
    np.savez(
        out_dir / "vectors.npz",
        **stacked,
        **{f"ref_{k}": v for k, v in reference_vectors.items()},
    )
    (out_dir / "boxes.json").write_text(json.dumps(boxes))
    (out_dir / "references.json").write_text(json.dumps(references, indent=1))
    (out_dir / "frames.json").write_text(json.dumps(frames, indent=1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def load_store(
    out_dir: Path,
) -> tuple[
    list[dict],
    dict[str, np.ndarray],
    list[dict],
    dict[str, np.ndarray],
    list[dict],
    dict,
]:
    """Read the store of `collect.py`."""
    assert (out_dir / "vectors.npz").is_file(), f"no store in {out_dir}"
    data = np.load(out_dir / "vectors.npz")
    vectors: dict[str, np.ndarray] = {k: data[k] for k in VECTOR_KINDS}
    reference_vectors: dict[str, np.ndarray] = {
        k: data[f"ref_{k}"] for k in VECTOR_KINDS
    }
    boxes: list[dict] = json.loads((out_dir / "boxes.json").read_text())
    references: list[dict] = json.loads((out_dir / "references.json").read_text())
    frames: list[dict] = json.loads((out_dir / "frames.json").read_text())
    meta: dict = json.loads((out_dir / "meta.json").read_text())
    return boxes, vectors, references, reference_vectors, frames, meta
