"""Shared helpers of the calibration scripts. This module is not part of the runtime."""

import dataclasses
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ai.postprocess import apply_crop_quality, select_candidates
from ai.settings import AiConfig
from ai.types import Box, Candidate, RawDetection
from storage.imaging import prepare_image
from storage.settings import load_config as load_storage_config


def load_raw(out_dir: Path) -> dict:
    """Read the cache that `collect.py` wrote."""
    return json.loads((out_dir / "raw.json").read_text())


def raw_detections(entry: dict, min_score: float = 0.0) -> list[RawDetection]:
    """Make the raw detections of one image. Keep those with a score of at least `min_score`."""
    return [
        RawDetection(Box(*d["box"]), d["score"], d["label"], d["source"])
        for d in entry["raw"]
        if d["score"] >= min_score
    ]


def load_image(dataset: Path, name: str) -> Image.Image:
    """Open an image as the application does: through the preparation of an upload."""
    prepared = prepare_image(
        (dataset / f"{name}.jpeg").read_bytes(), load_storage_config()
    )
    from io import BytesIO

    return Image.open(BytesIO(prepared.data)).convert("RGB")


def with_detection(config: AiConfig, **changes: object) -> AiConfig:
    """Return a copy of the config with other values in `[detection]`."""
    return dataclasses.replace(
        config, detection=dataclasses.replace(config.detection, **changes)
    )


def with_quality(config: AiConfig, **changes: object) -> AiConfig:
    """Return a copy of the config with other values in `[quality]`."""
    return dataclasses.replace(
        config, quality=dataclasses.replace(config.quality, **changes)
    )


def candidates_for(
    image: Image.Image,
    entry: dict,
    vocabulary: list[str],
    config: AiConfig,
    sources: tuple[str, ...] = ("full", "tile", "generic"),
) -> list[Candidate]:
    """Make the candidates of one image as the engine does, with the thresholds of `config`.

    The cache holds the output of a loose run. A box with a score below
    `box_threshold` is dropped here. `sources` selects the passes.
    """
    raw: list[RawDetection] = [
        d
        for d in raw_detections(entry, config.detection.box_threshold)
        if d.source in sources
    ]
    candidates: list[Candidate] = select_candidates(
        raw, image.width, image.height, vocabulary, config.detection
    )
    quality = config.quality
    return apply_crop_quality(
        candidates, image, quality.min_crop_side, quality.crop_blur_min
    )


def iou(a: Box, b: Box) -> float:
    """Return the intersection over union of two boxes."""
    width: int = min(a.x1, b.x1) - max(a.x0, b.x0)
    height: int = min(a.y1, b.y1) - max(a.y0, b.y0)
    inter: int = width * height if width > 0 and height > 0 else 0
    return inter / (a.area + b.area - inter)


def draw_candidates(
    image: Image.Image, candidates: list[Candidate], title: str, max_side: int = 700
) -> Image.Image:
    """Draw numbered boxes on a copy of the image, for the labelling by eye."""
    scale: float = min(1.0, max_side / max(image.size))
    picture: Image.Image = image.resize(
        (round(image.width * scale), round(image.height * scale))
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(picture)
    for index, c in enumerate(candidates):
        box = [round(v * scale) for v in (c.box.x0, c.box.y0, c.box.x1, c.box.y1)]
        colour = (255, 40, 40) if c.quality_ok else (255, 170, 0)
        draw.rectangle(box, outline=colour, width=2)
        draw.text((box[0] + 3, box[1] + 2), str(index), fill=colour)
    draw.text((4, 4), title, fill=(0, 90, 255))
    return picture
