"""Measure the signals of an evidence-based match on stored data. This script is not part of the runtime.

It reads a copy of the data read-only: the SQLite file, the region index and the image files.
For each region and each enrolled object it computes these signals:

- dino_best    the highest DINOv2 cosine over the views of the object (the signal of today)
- dino_agree   the mean of the best DINOv2 cosine of the 2 best reference photos
- dino_margin  dino_best minus the best dino_best of any other object
- siglip_vis   the mean of the SigLIP cosine of the 2 best reference crops
- meaning      the SigLIP share of the object's words against the other objects and common things
- keypoints    the LightGlue match count with the best reference crop (only for candidates)

Usage:
    uv run python benchmarks/matching/evidence.py collect OUT_DIR [--keypoints]
    uv run python benchmarks/matching/evidence.py report OUT_DIR TRUTH_JSON

TRUTH_JSON maps each object name to the list of frame ids that show it, for example
{"Watch": [1, 2, 15], "Headset": [5, 9, 10]}. An object that is not in the file is not scored.
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

PROJECT: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from ai.api import open_image
from ai.scene import SceneClassifier
from ai.settings import AiConfig, load_config
from ai.verifier import KeypointVerifier

logger: logging.Logger = logging.getLogger("evidence")

# Common things that a box can show instead of an enrolled object. The meaning signal compares with them.
CONTRAST_PHRASES: tuple[str, ...] = (
    "a hand",
    "an arm",
    "a person",
    "a face",
    "a table",
    "a wall",
    "a floor",
    "a chair",
    "a bed",
    "a phone",
    "a bottle",
    "a cup",
    "a bag",
    "a cable",
    "a laptop",
    "a paper",
    "a bracelet",
    "an object",
)
# The padding of a crop for SigLIP, as in the open search.
SIGLIP_PADDING: float = 0.05
# A region is a candidate for the keypoint check when its DINOv2 score is at least this value.
KEYPOINT_CANDIDATE_MIN: float = 0.35
TOP_PHOTOS: int = 2


def connect(db_path: Path) -> sqlite3.Connection:
    assert db_path.is_file(), f"no database at {db_path}"
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def read_image(config: AiConfig, file_path: str) -> Image.Image:
    data: bytes = (PROJECT / "storage" / "uploads" / file_path).read_bytes()
    return open_image(data, config.image.max_pixels)


def padded(
    image: Image.Image, box: tuple[int, int, int, int], pad: float
) -> Image.Image:
    x0, y0, x1, y1 = box
    px: int = round((x1 - x0) * pad)
    py: int = round((y1 - y0) * pad)
    return image.crop(
        (
            max(0, x0 - px),
            max(0, y0 - py),
            min(image.width, x1 + px),
            min(image.height, y1 + py),
        )
    )


def top_photo_mean(
    scores: np.ndarray, photo_ids: np.ndarray, top: int
) -> tuple[float, int]:
    """Return the mean of the best score of the `top` best photos, and the number of photos used."""
    best: list[float] = sorted(
        (float(scores[photo_ids == p].max()) for p in np.unique(photo_ids)),
        reverse=True,
    )
    used: list[float] = best[:top]
    return float(np.mean(used)), len(used)


def collect(out_dir: Path, with_keypoints: bool) -> None:
    config: AiConfig = load_config()
    db: sqlite3.Connection = connect(PROJECT / "db" / "data" / "saarthi.db")
    index = faiss.read_index(str(PROJECT / "db" / "data" / "regions.faiss"))
    scene: SceneClassifier = SceneClassifier(config)
    verifier: KeypointVerifier | None = (
        KeypointVerifier(config) if with_keypoints else None
    )

    objects: list[dict] = []
    for oid, name, phrase, synonyms, tau in db.execute(
        "select id, name, category_phrase, synonyms, tau from enrolled_object where status = 'ready' order by id"
    ):
        rows = db.execute(
            "select photo_id, vector from object_view where enrolled_object_id = ?",
            (oid,),
        ).fetchall()
        photos = db.execute(
            "select id, file_path, x0, y0, x1, y1 from reference_photo where enrolled_object_id = ? and status = 'confirmed' order by id",
            (oid,),
        ).fetchall()
        sig_crops: list[Image.Image] = []
        kp_crops: list[Image.Image] = []
        for _, path, x0, y0, x1, y1 in photos:
            image = read_image(config, path)
            sig_crops.append(padded(image, (x0, y0, x1, y1), SIGLIP_PADDING))
            kp_crops.append(
                padded(image, (x0, y0, x1, y1), config.verification.crop_padding)
            )
        objects.append(
            {
                "id": oid,
                "name": name,
                "tau": tau,
                "phrases": [phrase, *json.loads(synonyms)],
                "dino": np.stack([np.frombuffer(v, dtype=np.float32) for _, v in rows]),
                "dino_photo": np.array([p for p, _ in rows]),
                "siglip": scene.embed_crops(sig_crops),
                "siglip_photo": np.array([p[0] for p in photos]),
                "kp_crops": kp_crops,
            }
        )
        logger.info(
            "Object %d: %d views, %d reference crops", oid, len(rows), len(photos)
        )

    # One SigLIP vector for each phrase. A class scores with its best phrase.
    classes: list[tuple[str, list[str]]] = [(o["name"], o["phrases"]) for o in objects]
    classes += [(p, [p]) for p in CONTRAST_PHRASES]
    phrase_list: list[str] = [p for _, ps in classes for p in ps]
    phrase_vectors: np.ndarray = scene.embed_phrases(phrase_list)
    owner: np.ndarray = np.array([i for i, (_, ps) in enumerate(classes) for _ in ps])
    scale: float = float(scene._model.logit_scale.exp().item())
    bias: float = float(scene._model.logit_bias.item())

    records: list[dict] = []
    frames = db.execute(
        "select id, file_path from image where status = 'done' order by id"
    ).fetchall()
    for image_id, path in frames:
        regions = db.execute(
            "select id, x0, y0, x1, y1, phrases, quality_ok, in_hand from region where image_id = ? order by id",
            (image_id,),
        ).fetchall()
        if not regions:
            continue
        image = read_image(config, path)
        crops = [
            padded(image, (r[1], r[2], r[3], r[4]), SIGLIP_PADDING) for r in regions
        ]
        crop_vectors: np.ndarray = scene.embed_crops(crops)
        for (rid, x0, y0, x1, y1, phrases, quality_ok, in_hand), sv in zip(
            regions, crop_vectors
        ):
            dv: np.ndarray = index.reconstruct(rid)
            logits: np.ndarray = (phrase_vectors @ sv) * scale + bias
            class_logits: np.ndarray = np.array(
                [logits[owner == c].max() for c in range(len(classes))]
            )
            shares: np.ndarray = np.exp(class_logits - class_logits.max())
            shares /= shares.sum()
            best: dict[int, float] = {
                o["id"]: float((o["dino"] @ dv).max()) for o in objects
            }
            for i, o in enumerate(objects):
                agree, _ = top_photo_mean(o["dino"] @ dv, o["dino_photo"], TOP_PHOTOS)
                vis, _ = top_photo_mean(o["siglip"] @ sv, o["siglip_photo"], TOP_PHOTOS)
                rival: float = max(
                    (v for k, v in best.items() if k != o["id"]), default=0.0
                )
                keypoints: int | None = None
                if verifier is not None and best[o["id"]] >= KEYPOINT_CANDIDATE_MIN:
                    candidate = padded(
                        image, (x0, y0, x1, y1), config.verification.crop_padding
                    )
                    keypoints = max(
                        verifier.count_matches(ref, candidate) for ref in o["kp_crops"]
                    )
                records.append(
                    {
                        "frame": image_id,
                        "region": rid,
                        "object": o["name"],
                        "tau": o["tau"],
                        "phrases": json.loads(phrases),
                        "object_phrases": o["phrases"],
                        "quality_ok": bool(quality_ok),
                        "in_hand": bool(in_hand),
                        "size": [x1 - x0, y1 - y0],
                        "dino_best": best[o["id"]],
                        "dino_agree": agree,
                        "dino_margin": best[o["id"]] - rival,
                        "siglip_vis": vis,
                        "meaning": float(shares[i]),
                        "keypoints": keypoints,
                    }
                )
        logger.info("Frame %d: %d regions", image_id, len(regions))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signals.json").write_text(json.dumps(records, indent=1))
    logger.info("Wrote %d records", len(records))


def report(out_dir: Path, truth_path: Path) -> None:
    records: list[dict] = json.loads((out_dir / "signals.json").read_text())
    truth: dict[str, list[int]] = json.loads(truth_path.read_text())
    keys = (
        "dino_best",
        "dino_agree",
        "dino_margin",
        "siglip_vis",
        "meaning",
        "keypoints",
    )
    for name, frames in truth.items():
        rows = [r for r in records if r["object"] == name]
        print(f"\n=== {name}: true frames {frames}")
        print("frame region  ok  " + "  ".join(f"{k:>11}" for k in keys) + "  phrases")
        for frame in sorted({r["frame"] for r in rows}):
            mark = "*" if frame in frames else " "
            best = sorted(
                (r for r in rows if r["frame"] == frame), key=lambda r: -r["siglip_vis"]
            )[:3]
            for r in best:
                values = "  ".join(
                    f"{'-' if r[k] is None else (r[k] if k == 'keypoints' else round(r[k], 3)):>11}"
                    for k in keys
                )
                print(
                    f"{mark}{frame:4d} {r['region']:6d}  {int(r['quality_ok'])}  {values}  {r['phrases']}"
                )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["collect", "report"])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("truth", type=Path, nargs="?")
    parser.add_argument("--keypoints", action="store_true")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.out_dir, args.keypoints)
    else:
        assert args.truth is not None, "report needs TRUTH_JSON"
        report(args.out_dir, args.truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
