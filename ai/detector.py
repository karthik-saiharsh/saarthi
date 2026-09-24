"""Candidate detector: Grounding DINO with three kinds of pass.

- full: the whole frame, with the vocabulary phrases.
- tile: a 2 x 2 grid, for the small phrases only (a switch in the config).
- generic: the phrase "an object", for objects that the words do not describe.

The detector returns raw boxes. `ai.postprocess` turns them into candidates.
"""

import logging
import math
import threading
import time
from collections.abc import Sequence

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ai.errors import InferenceError, ModelLoadError
from ai.geometry import Tile, clip_box, map_tile_box, tile_grid
from ai.postprocess import plan_prompts
from ai.runtime import load_kwargs, resolve_device, short_model_name
from ai.settings import AiConfig
from ai.types import Box, RawDetection, Source

logger: logging.Logger = logging.getLogger(__name__)

# The longest side of the detector input. This is the default of the model.
LONGEST_EDGE: int = 1333


class CandidateDetector:
    """Finds boxes for text phrases with Grounding DINO. Load it once."""

    def __init__(self, config: AiConfig) -> None:
        assert config is not None, "config is required"
        self._config: AiConfig = config
        self._device: str = resolve_device(config.models.device)
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        kwargs = load_kwargs(config.models, config.models.detector_revision)
        try:
            self._processor = AutoProcessor.from_pretrained(
                config.models.detector_id, **kwargs
            )
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                config.models.detector_id, **kwargs
            )
        except (OSError, ValueError) as error:
            logger.error(
                "Cannot load the detector %s: %s", config.models.detector_id, error
            )
            raise ModelLoadError(f"Cannot load {config.models.detector_id}") from error
        self._model.eval().to(self._device)
        self.name: str = short_model_name(
            config.models.detector_id, config.models.detector_revision
        )
        logger.info(
            "Detector %s ready on %s in %.1f s",
            self.name,
            self._device,
            time.perf_counter() - started,
        )

    def count_tokens(self, phrase: str) -> int:
        """Return the number of text tokens of a phrase."""
        return len(
            self._processor.tokenizer(phrase, add_special_tokens=False)["input_ids"]
        )

    def detect(
        self,
        image: Image.Image,
        phrases: Sequence[str],
        small_phrases: Sequence[str] = (),
        include_generic: bool = False,
        tiling: bool | None = None,
        context_phrases: bool = True,
    ) -> list[RawDetection]:
        """Return the raw boxes of all passes. `ai.postprocess` cleans them up.

        `tiling` switches the tiled pass for this call. None uses the config.
        `context_phrases` adds the hand and depiction phrases to each prompt.
        Switch it off to look for one object only (see `AiEngine.locate_object`).
        """
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert phrases, "phrases must not be empty"
        cfg = self._config.detection
        width, height = image.size
        started: float = time.perf_counter()
        extras: list[str] = (
            [cfg.hand_phrase, *cfg.depiction_phrases] if context_phrases else []
        )
        detections: list[RawDetection] = []
        passes: int = 0

        for prompt in plan_prompts(
            list(phrases), extras, cfg.chunk_size, self.count_tokens
        ):
            detections += self._run(image, prompt, "full")
            passes += 1

        small: list[str] = [p for p in phrases if p in set(small_phrases)]
        use_tiling: bool = cfg.tiling_enabled if tiling is None else tiling
        if use_tiling and small and min(width, height) >= 2:
            tile_extras: list[str] = [cfg.hand_phrase] if context_phrases else []
            tiles: list[Tile] = tile_grid(width, height, cfg.tile_overlap)
            for prompt in plan_prompts(
                small, tile_extras, cfg.chunk_size, self.count_tokens
            ):
                for tile in tiles:
                    crop: Image.Image = image.crop((tile.x0, tile.y0, tile.x1, tile.y1))
                    detections += self._run(crop, prompt, "tile", tile)
                    passes += 1

        if include_generic:
            detections += self._run(image, [cfg.generic_phrase], "generic")
            passes += 1

        logger.info(
            "Detector: %d pass(es), %d raw box(es) in %.2f s",
            passes,
            len(detections),
            time.perf_counter() - started,
        )
        return detections

    def _run(
        self,
        image: Image.Image,
        prompt: list[str],
        source: Source,
        tile: Tile | None = None,
    ) -> list[RawDetection]:
        """Run one detector pass. Return boxes in the coordinates of the full frame."""
        cfg = self._config.detection
        width, height = image.size
        try:
            inputs = self._processor(
                images=image,
                text=[prompt],
                return_tensors="pt",
                size={
                    "shortest_edge": cfg.input_short_edge,
                    "longest_edge": LONGEST_EDGE,
                },
            ).to(self._device)
            with self._lock, torch.inference_mode():
                outputs = self._model(**inputs)
            result: dict = self._processor.post_process_grounded_object_detection(
                outputs,
                threshold=cfg.box_threshold,
                text_threshold=cfg.text_threshold,
                target_sizes=[(height, width)],
            )[0]
        except (RuntimeError, ValueError, KeyError) as error:
            logger.error("Detector pass failed: %s: %s", type(error).__name__, error)
            raise InferenceError("Detector pass failed") from error

        found: list[RawDetection] = []
        labels: list[str] = list(result["text_labels"])
        for coordinates, score, label in zip(
            result["boxes"].tolist(), result["scores"].tolist(), labels
        ):
            box: Box | None = _to_box(coordinates, width, height)
            if box is None:
                continue
            if tile is not None:
                box = map_tile_box(box, tile)
            found.append(RawDetection(box, float(score), str(label), source))
        return found


def _to_box(coordinates: list[float], width: int, height: int) -> Box | None:
    """Round a float box outwards to pixels and clip it. Return None if nothing is left."""
    x0, y0, x1, y1 = (
        math.floor(coordinates[0]),
        math.floor(coordinates[1]),
        math.ceil(coordinates[2]),
        math.ceil(coordinates[3]),
    )
    if x0 >= x1 or y0 >= y1:
        return None
    return clip_box(Box(x0, y0, x1, y1), width, height)
