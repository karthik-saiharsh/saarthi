"""Region embedder: DINOv2 patch features pooled inside each box.

One forward pass over the whole frame gives a grid of patch vectors. The
vector of a region is the mean of the patches that lie inside its box.
A small box is embedded again from a padded crop, with the same pooling,
so all vectors stay in one space.
"""

import logging
import threading
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ai.errors import InferenceError, ModelLoadError
from ai.runtime import load_kwargs, resolve_device, short_model_name
from ai.settings import PATCH_SIZE, AiConfig
from ai.types import Box

logger: logging.Logger = logging.getLogger(__name__)

# Crops that go through the model together.
CROP_BATCH_SIZE: int = 16


def frame_size(
    width: int, height: int, short_edge: int, long_edge_max: int
) -> tuple[int, int]:
    """Return the model input size (width, height) for a frame.

    The short edge is scaled to `short_edge`. The long edge is cut to
    `long_edge_max`. Both sides are multiples of the patch size.
    """
    assert width > 0 and height > 0, "frame size must be positive"
    scale: float = short_edge / min(width, height)
    if max(width, height) * scale > long_edge_max:
        scale = long_edge_max / max(width, height)

    def snap(value: float) -> int:
        return max(PATCH_SIZE, int(value / PATCH_SIZE) * PATCH_SIZE)

    return snap(width * scale), snap(height * scale)


def is_small_box(
    box: Box, width: int, height: int, model_size: tuple[int, int], min_side: int
) -> bool:
    """Return True if the short side of the box is below `min_side` in model pixels.

    The frame is scaled to `model_size` before the model sees it. The patch grid
    follows the scaled size, so the cutoff must too. A 64 px box on a 12 MP
    frame covers less than one patch of the model input.
    """
    assert width > 0 and height > 0, "frame size must be positive"
    assert min_side > 0, "min_side must be positive"
    scaled_width: float = box.width * model_size[0] / width
    scaled_height: float = box.height * model_size[1] / height
    return min(scaled_width, scaled_height) < min_side


def pool_box(tokens: torch.Tensor, box: Box, width: int, height: int) -> torch.Tensor:
    """Return the mean of the patch vectors that lie inside the box.

    `tokens` has the shape (rows, columns, dimension) and covers the frame
    of `width` x `height` pixels. A patch counts if its centre is in the
    box. If no centre is in the box, the patch under the box centre is used.
    """
    rows, columns, _ = tokens.shape
    column_centres: torch.Tensor = (torch.arange(columns) + 0.5) * (width / columns)
    row_centres: torch.Tensor = (torch.arange(rows) + 0.5) * (height / rows)
    column_mask: torch.Tensor = (column_centres >= box.x0) & (column_centres < box.x1)
    row_mask: torch.Tensor = (row_centres >= box.y0) & (row_centres < box.y1)
    if column_mask.any() and row_mask.any():
        return (
            tokens[row_mask][:, column_mask].reshape(-1, tokens.shape[-1]).mean(dim=0)
        )
    column: int = min(columns - 1, int((box.x0 + box.x1) / 2 / width * columns))
    row: int = min(rows - 1, int((box.y0 + box.y1) / 2 / height * rows))
    return tokens[row, column]


def padded_square(box: Box, padding: float, width: int, height: int) -> Box:
    """Return a square region around the box, with padding, moved into the frame."""
    side: int = max(box.width, box.height)
    side = min(round(side * (1.0 + 2.0 * padding)), width, height)
    centre_x: float = (box.x0 + box.x1) / 2
    centre_y: float = (box.y0 + box.y1) / 2
    x0: int = int(min(max(centre_x - side / 2, 0), width - side))
    y0: int = int(min(max(centre_y - side / 2, 0), height - side))
    return Box(x0, y0, x0 + side, y0 + side)


class RegionEmbedder:
    """Makes unit-length region vectors with DINOv2. Load it once."""

    def __init__(self, config: AiConfig) -> None:
        assert config is not None, "config is required"
        self._config: AiConfig = config
        self._device: str = resolve_device(config.models.device)
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        kwargs = load_kwargs(config.models, config.models.embedder_revision)
        try:
            processor = AutoImageProcessor.from_pretrained(
                config.models.embedder_id, **kwargs
            )
            self._model = AutoModel.from_pretrained(config.models.embedder_id, **kwargs)
        except (OSError, ValueError) as error:
            logger.error(
                "Cannot load the embedder %s: %s", config.models.embedder_id, error
            )
            raise ModelLoadError(f"Cannot load {config.models.embedder_id}") from error
        self._model.eval().to(self._device)
        self._mean: torch.Tensor = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        self._std: torch.Tensor = torch.tensor(processor.image_std).view(1, 3, 1, 1)
        self.dimension: int = int(self._model.config.hidden_size)
        self.name: str = short_model_name(
            config.models.embedder_id, config.models.embedder_revision
        )
        logger.info(
            "Embedder %s ready on %s in %.1f s, dimension %d",
            self.name,
            self._device,
            time.perf_counter() - started,
            self.dimension,
        )

    def _pixels(self, image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
        resized: Image.Image = image.convert("RGB").resize(
            size, Image.Resampling.BICUBIC
        )
        array: np.ndarray = np.asarray(resized, dtype=np.float32) / 255.0
        tensor: torch.Tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return (tensor - self._mean) / self._std

    def _patch_grid(self, pixels: torch.Tensor) -> torch.Tensor:
        """Run the model. Return patch vectors with the shape (batch, rows, columns, dimension)."""
        _, _, height, width = pixels.shape
        rows, columns = height // PATCH_SIZE, width // PATCH_SIZE
        with self._lock, torch.inference_mode():
            hidden: torch.Tensor = self._model(
                pixel_values=pixels.to(self._device)
            ).last_hidden_state
        patches: torch.Tensor = hidden[:, -rows * columns :, :]  # Skip the CLS token
        return patches.reshape(pixels.shape[0], rows, columns, -1).float().cpu()

    def embed(self, image: Image.Image, boxes: list[Box]) -> np.ndarray:
        """Return one unit-length float32 vector for each box, in the same order."""
        assert isinstance(image, Image.Image), "image must be a PIL image"
        assert isinstance(boxes, list), "boxes must be a list"
        width, height = image.size
        assert all(
            0 <= b.x0 and 0 <= b.y0 and b.x1 <= width and b.y1 <= height for b in boxes
        ), "every box must lie inside the image"
        if not boxes:
            return np.empty((0, self.dimension), dtype=np.float32)

        cfg = self._config.embedding
        started: float = time.perf_counter()
        try:
            vectors: list[torch.Tensor | None] = [None] * len(boxes)
            size: tuple[int, int] = frame_size(
                width, height, cfg.frame_short_edge, cfg.frame_long_edge_max
            )
            small: list[int] = [
                i
                for i, b in enumerate(boxes)
                if is_small_box(b, width, height, size, cfg.small_box_min_side)
            ]
            small_set: set[int] = set(small)
            large: list[int] = [i for i in range(len(boxes)) if i not in small_set]

            if large:
                grid: torch.Tensor = self._patch_grid(self._pixels(image, size))[0]
                for i in large:
                    vectors[i] = pool_box(grid, boxes[i], width, height)

            for start in range(0, len(small), CROP_BATCH_SIZE):
                batch: list[int] = small[start : start + CROP_BATCH_SIZE]
                crops: list[torch.Tensor] = []
                for i in batch:
                    square: Box = padded_square(
                        boxes[i], cfg.crop_padding, width, height
                    )
                    crop: Image.Image = image.crop(
                        (square.x0, square.y0, square.x1, square.y1)
                    )
                    crops.append(self._pixels(crop, (cfg.crop_size, cfg.crop_size)))
                grids: torch.Tensor = self._patch_grid(torch.cat(crops))
                for position, i in enumerate(batch):
                    vectors[i] = (
                        grids[position].reshape(-1, grids.shape[-1]).mean(dim=0)
                    )

            stacked: torch.Tensor = torch.stack([v for v in vectors if v is not None])
            result: np.ndarray = torch.nn.functional.normalize(stacked, dim=1).numpy()
        except (RuntimeError, ValueError) as error:
            logger.error("Embedding failed: %s: %s", type(error).__name__, error)
            raise InferenceError("Embedding failed") from error
        logger.info(
            "Embedded %d region(s) (%d from crops) in %.2f s",
            len(boxes),
            len(small),
            time.perf_counter() - started,
        )
        return result.astype(np.float32)
