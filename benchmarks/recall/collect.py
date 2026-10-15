"""Run the models one time on a frame set and write the signal store. Not part of the runtime.

For each frame it does two things:

1. **Today's pipeline**, through `AiEngine` with the calls of the worker and the memory: the detector with the
   phrases of the gallery, the region vectors, the mode `evidence` and the mode `tau`. All objects of the set are
   enrolled and calibrated together, as the carer would do it. No generic pass, no keypoint check.
2. **The new signals** (plan Revision 3) for every box of the union of the Grounding DINO candidates and the
   YOLOE boxes: the SigLIP crop vector, the DINOv2 CLS token of the tight crop, the DINOv2 patches pooled inside
   the YOLOE mask and with the CLS attention, and the detector score of each object word.

It reads the frames read-only and writes only to OUT_DIR.

Usage:
    uv run python benchmarks/recall/collect.py SET_JSON OUT_DIR [--limit N]
"""

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    IOU_SAME,
    VECTOR_KINDS,
    BoxTuple,
    FrameSet,
    ObjectSpec,
    iou,
    load_set,
    resolve,
    save_store,
)
from transformers import AutoModel

from ai.api import AiEngine, ImageError, open_image
from ai.open_detector import (
    MODELS_DIR,
    ensure_weights,
    import_ultralytics,
)
from ai.postprocess import apply_crop_quality
from ai.runtime import load_kwargs
from ai.settings import AiConfig, load_config
from ai.types import Box, Candidate, MatchObject, MatchRegion, Sighting

logger: logging.Logger = logging.getLogger("recall.collect")

# DINOv2 sees a crop at this size. 224 / 14 gives a grid of 16 x 16 patches.
CROP_SIZE: int = 224
GRID: int = 16
# The attention of the CLS token is the mean of the heads of these last layers.
ATTENTION_LAYERS: int = 4
# A mask that covers less than this part of its box is not used.
MASK_MIN_FILL: float = 0.05
# Write the partial store after this number of frames.
CHECKPOINT_EVERY: int = 25
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

Polygon = np.ndarray | None


class CropSignals:
    """DINOv2 vectors of tight crops: the CLS token, the patches pooled in the mask, the patches pooled with attention."""

    def __init__(self, config: AiConfig) -> None:
        models = config.models
        kwargs: dict = load_kwargs(models, models.embedder_revision)
        # The attention weights are needed, and only the eager implementation returns them.
        self._model = AutoModel.from_pretrained(
            models.embedder_id, attn_implementation="eager", **kwargs
        ).eval()
        self._mean: torch.Tensor = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std: torch.Tensor = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def embed(
        self, image: Image.Image, boxes: list[BoxTuple], polygons: list[Polygon]
    ) -> dict[str, np.ndarray]:
        assert len(boxes) == len(polygons), "one polygon (or None) per box"
        if not boxes:
            return {
                k: np.zeros((0, 384), np.float32)
                for k in ("cls", "pool_mask", "pool_attn")
            }
        pixels: list[torch.Tensor] = []
        weights: list[np.ndarray] = []
        for box, polygon in zip(boxes, polygons):
            crop: Image.Image = image.crop(box).resize(
                (CROP_SIZE, CROP_SIZE), Image.BICUBIC
            )
            array: np.ndarray = np.asarray(crop, dtype=np.float32) / 255.0
            pixels.append(torch.from_numpy(array).permute(2, 0, 1))
            weights.append(_mask_grid(box, polygon))
        batch: torch.Tensor = (torch.stack(pixels) - self._mean) / self._std
        with torch.inference_mode():
            out = self._model(pixel_values=batch, output_attentions=True)
        hidden: torch.Tensor = out.last_hidden_state
        cls: torch.Tensor = hidden[:, 0]
        patches: torch.Tensor = hidden[:, 1:]
        attention: torch.Tensor = torch.stack(
            [
                out.attentions[-i][:, :, 0, 1:].mean(1)
                for i in range(1, ATTENTION_LAYERS + 1)
            ]
        ).mean(0)
        attention = attention / attention.sum(1, keepdim=True)
        mask: torch.Tensor = torch.from_numpy(np.stack(weights))
        mask = mask / mask.sum(1, keepdim=True).clamp(min=1e-6)
        unit = torch.nn.functional.normalize
        return {
            "cls": unit(cls, dim=-1).numpy(),
            "pool_mask": unit((patches * mask[..., None]).sum(1), dim=-1).numpy(),
            "pool_attn": unit((patches * attention[..., None]).sum(1), dim=-1).numpy(),
        }


def _mask_grid(box: BoxTuple, polygon: Polygon) -> np.ndarray:
    """Return the share of each patch of the 16 x 16 grid that is inside the mask. Without a mask, all patches count."""
    x0, y0, x1, y1 = box
    if polygon is None:
        return np.ones(GRID * GRID, np.float32)
    canvas: Image.Image = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(canvas).polygon(
        [(float(px - x0), float(py - y0)) for px, py in polygon], fill=255
    )
    grid: np.ndarray = (
        np.asarray(canvas.resize((GRID, GRID), Image.BILINEAR), dtype=np.float32)
        / 255.0
    )
    if grid.mean() < MASK_MIN_FILL:
        return np.ones(GRID * GRID, np.float32)
    return grid.reshape(-1)


class BroadBoxes:
    """YOLOE prompt-free boxes with their masks, with the settings of `[open_detection]`."""

    def __init__(self, config: AiConfig) -> None:
        models = config.models
        weights: Path = ensure_weights(
            MODELS_DIR,
            models.open_detector_file,
            models.open_detector_url,
            models.open_detector_sha256,
            allow_download=not models.local_files_only,
        )
        YOLOE, settings, yolo_logger = import_ultralytics()
        settings.update({"sync": False})
        yolo_logger.setLevel(logging.WARNING)
        self._model = YOLOE(str(weights))
        self._cfg = config.open_detection

    def detect(self, image: Image.Image) -> list[tuple[BoxTuple, float, str, Polygon]]:
        result = self._model.predict(
            image,
            conf=self._cfg.confidence,
            imgsz=self._cfg.input_size,
            verbose=False,
            max_det=self._cfg.max_boxes,
        )[0]
        polygons = (
            result.masks.xy if result.masks is not None else [None] * len(result.boxes)
        )
        found: list[tuple[BoxTuple, float, str, Polygon]] = []
        for item, polygon in zip(result.boxes, polygons):
            x0, y0, x1, y1 = (round(v) for v in item.xyxy[0].tolist())
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.width, x1), min(image.height, y1)
            if min(x1 - x0, y1 - y0) < self._cfg.min_box_side:
                continue
            usable: Polygon = (
                polygon if polygon is not None and len(polygon) >= 3 else None
            )
            found.append(
                (
                    (x0, y0, x1, y1),
                    float(item.conf),
                    str(result.names[int(item.cls)]),
                    usable,
                )
            )
        found.sort(key=lambda f: -f[1])
        return found[: self._cfg.max_boxes]


def benchmark_config() -> AiConfig:
    """The config of the application, without the models that the benchmark does not use."""
    config: AiConfig = load_config()
    return dataclasses.replace(
        config,
        speech=dataclasses.replace(config.speech, enabled=False),
        language=dataclasses.replace(config.language, enabled=False),
        verification=dataclasses.replace(config.verification, enabled=False),
        open_detection=dataclasses.replace(config.open_detection, enabled=False),
    )


def read(config: AiConfig, path: str) -> Image.Image:
    return open_image(resolve(path).read_bytes(), config.image.max_pixels)


def to_box(box: BoxTuple) -> Box:
    return Box(*box)


def as_tuple(box: Box) -> BoxTuple:
    return (box.x0, box.y0, box.x1, box.y1)


def best_polygon(
    box: BoxTuple, broad: list[tuple[BoxTuple, float, str, Polygon]]
) -> Polygon:
    """The mask of the YOLOE box that fits `box` best, if it fits well enough."""
    best: tuple[float, Polygon] = (0.0, None)
    for other, _, _, polygon in broad:
        overlap: float = iou(box, other)
        if overlap > best[0]:
            best = (overlap, polygon)
    return best[1] if best[0] >= IOU_SAME else None


class Collector:
    def __init__(self, frame_set: FrameSet, out_dir: Path) -> None:
        self.set: FrameSet = frame_set
        self.out_dir: Path = out_dir
        self.config: AiConfig = benchmark_config()
        self.engine: AiEngine = AiEngine(self.config)
        self.broad: BroadBoxes = BroadBoxes(self.config)
        self.crops: CropSignals = CropSignals(self.config)
        self.names: list[str] = list(frame_set.objects)
        self.phrases: list[str] = []
        for spec in frame_set.objects.values():
            for phrase in spec.phrases:
                if phrase not in self.phrases:
                    self.phrases.append(phrase)
        self.boxes: list[dict] = []
        self.vectors: dict[str, list[np.ndarray]] = {k: [] for k in VECTOR_KINDS}
        self.frames: list[dict] = []
        self.references: list[dict] = []
        self.reference_vectors: dict[str, np.ndarray] = {}
        self.objects: list[MatchObject] = []
        self.calibration: dict[str, dict] = {}

    def signals(
        self, image: Image.Image, boxes: list[BoxTuple], polygons: list[Polygon]
    ) -> dict[str, np.ndarray]:
        """All vector kinds of the store for the boxes of one image."""
        made: dict[str, np.ndarray] = self.crops.embed(image, boxes, polygons)
        made["pooled"] = self.engine.embed_regions(image, [to_box(b) for b in boxes])
        made["sig"] = self.engine.embed_evidence_crops(
            image, [to_box(b) for b in boxes]
        )
        return made

    def enroll(self) -> None:
        """Make the views and the reference vectors, then calibrate each object against the others (stages E3, E4)."""
        views: dict[str, list[np.ndarray]] = {n: [] for n in self.names}
        photo_ids: dict[str, list[int]] = {n: [] for n in self.names}
        collected: dict[str, list[np.ndarray]] = {k: [] for k in VECTOR_KINDS}
        photo: int = 0
        for name in self.names:
            spec: ObjectSpec = self.set.objects[name]
            for path, box in spec.references:
                photo += 1
                image: Image.Image = read(self.config, path)
                for _, vector in self.engine.embed_views(image, to_box(box)):
                    views[name].append(vector)
                    photo_ids[name].append(photo)
                polygon: Polygon = best_polygon(box, self.broad.detect(image))
                made = self.signals(image, [box], [polygon])
                for kind in VECTOR_KINDS:
                    collected[kind].append(made[kind])
                self.references.append(
                    {
                        "object": name,
                        "photo": photo,
                        "box": list(box),
                        "has_mask": polygon is not None,
                    }
                )
        self.reference_vectors = {k: np.concatenate(v) for k, v in collected.items()}
        for name in self.names:
            own: np.ndarray = np.stack(views[name])
            others: np.ndarray = np.stack(
                [v for n in self.names if n != name for v in views[n]]
            )
            result = self.engine.calibrate_object(own, photo_ids[name], others)
            rows: np.ndarray = np.array([r["object"] == name for r in self.references])
            self.objects.append(
                MatchObject(
                    object_id=self.names.index(name) + 1,
                    tracked=True,
                    phrases=self.set.objects[name].phrases,
                    views=own,
                    tau=result.tau,
                    tau_strong=result.tau_strong,
                    margin_min=result.margin_min,
                    view_photos=np.array(photo_ids[name]),
                    references=self.reference_vectors["sig"][rows],
                    reference_photos=np.array(
                        [r["photo"] for r in self.references if r["object"] == name]
                    ),
                )
            )
            self.calibration[name] = {
                "tau": result.tau,
                "tau_strong": result.tau_strong,
                "note": result.note,
            }
            logger.info("Object %s: %d views, tau %.3f", name, len(own), result.tau)

    def sightings(self, found: list[Sighting]) -> list[dict]:
        return [
            {
                "object": self.names[s.object_id - 1],
                "candidate": s.region_index,
                "strong": s.strong,
                "score": s.score,
                "similarity": s.similarity,
                "margin": s.margin,
            }
            for s in found
        ]

    def frame(self, key: str, path: str, kind: str) -> None:
        image: Image.Image = read(self.config, path)
        started: float = time.perf_counter()
        candidates: list[Candidate] = self.engine.detect_candidates(
            image, phrases=self.phrases, include_generic=False
        )
        detected: float = time.perf_counter()
        record: dict = {
            "key": key,
            "kind": kind,
            "first_row": len(self.boxes),
            "evidence": None,
            "tau": None,
        }
        if candidates:
            vectors: np.ndarray = self.engine.embed_regions(
                image, [c.box for c in candidates]
            )
            regions: list[MatchRegion] = [
                MatchRegion(tuple(c.phrases), c.quality_ok) for c in candidates
            ]
            chosen: list[int] = self.engine.evidence_regions(
                vectors, regions, self.objects
            )
            crops: np.ndarray = self.engine.embed_evidence_crops(
                image, [candidates[i].box for i in chosen]
            )
            record["evidence"] = {
                "chosen": chosen,
                "sightings": self.sightings(
                    self.engine.score_evidence(chosen, vectors, crops, self.objects)
                ),
            }
            record["tau"] = {
                "sightings": self.sightings(
                    self.engine.match_regions(vectors, regions, self.objects)
                )
            }
        today: float = time.perf_counter()
        broad = self.broad.detect(image)
        rows: list[dict] = []
        polygons: list[Polygon] = []
        for index, c in enumerate(candidates):
            box: BoxTuple = as_tuple(c.box)
            polygon = best_polygon(box, broad)
            rows.append(
                {
                    "source": "gdino",
                    "candidate": index,
                    "box": list(box),
                    "score": c.det_score,
                    "labels": list(c.phrases),
                    "quality_ok": c.quality_ok,
                    "in_hand": c.in_hand,
                }
            )
            polygons.append(polygon)
        broad_candidates: list[Candidate] = [
            Candidate(to_box(b), conf, "full", (), False, True)
            for b, conf, _, _ in broad
        ]
        checked: list[Candidate] = apply_crop_quality(
            broad_candidates,
            image,
            self.config.quality.min_crop_side,
            self.config.quality.crop_blur_min,
        )
        for (box, conf, label, polygon), c in zip(broad, checked):
            rows.append(
                {
                    "source": "yoloe",
                    "candidate": None,
                    "box": list(box),
                    "score": conf,
                    "labels": [label],
                    "quality_ok": c.quality_ok,
                    "in_hand": False,
                }
            )
            polygons.append(polygon)
        for row, polygon in zip(rows, polygons):
            row["frame"] = key
            row["has_mask"] = polygon is not None
            # The detector score of each object word on this box. Evidence only, never a gate.
            row["word"] = {
                name: max(
                    [
                        c.det_score
                        for c in candidates
                        if iou(as_tuple(c.box), tuple(row["box"])) >= IOU_SAME
                        and set(c.phrases) & set(self.set.objects[name].phrases)
                    ]
                    or [0.0]
                )
                for name in self.names
            }
        if rows:
            made = self.signals(image, [tuple(r["box"]) for r in rows], polygons)
            for k in VECTOR_KINDS:
                self.vectors[k].append(made[k])
        self.boxes.extend(rows)
        record["rows"] = len(rows)
        record["seconds"] = {
            "detector": round(detected - started, 3),
            "today_matching": round(today - detected, 3),
            "new_signals": round(time.perf_counter() - today, 3),
        }
        self.frames.append(record)

    def save(self) -> None:
        meta: dict = {
            "objects": self.names,
            "phrases": self.phrases,
            "calibration": self.calibration,
            "embedder": self.engine.embedding_model,
            "matching": dataclasses.asdict(self.config.matching),
            "frames_done": len(self.frames),
        }
        save_store(
            self.out_dir,
            self.boxes,
            self.vectors,
            self.references,
            self.reference_vectors,
            self.frames,
            meta,
        )

    def run(self, limit: int | None) -> None:
        self.enroll()
        work: list[tuple[str, str, str]] = [
            (k, p, "home") for k, p in self.set.frames.items()
        ]
        work += [(k, p, "cohort") for k, p in self.set.cohort.items()]
        if limit is not None:
            work = work[:limit]
        started: float = time.perf_counter()
        for done, (key, path, kind) in enumerate(work, start=1):
            try:
                self.frame(key, path, kind)
            except ImageError:
                logger.warning("Frame %s cannot be read. It is left out", key)
                continue
            if done % CHECKPOINT_EVERY == 0:
                self.save()
                logger.info(
                    "%d of %d frames, %.0f s",
                    done,
                    len(work),
                    time.perf_counter() - started,
                )
        self.save()
        logger.info(
            "Done: %d frames, %d boxes, %.0f s",
            len(self.frames),
            len(self.boxes),
            time.perf_counter() - started,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    for noisy in ("ai", "httpx", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("set_file", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N frames (a quick check)",
    )
    args = parser.parse_args()
    assert args.limit is None or args.limit > 0, "--limit must be positive"
    Collector(load_set(args.set_file), args.out_dir).run(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
