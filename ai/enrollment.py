"""Helpers for the views of a reference photo (stage E3). Pure functions."""

from PIL import Image, ImageEnhance

from ai.types import Box

# A box must keep at least this many pixels on each side.
MIN_BOX_SIDE: int = 2


def scale_box(box: Box, factor: float, width: int, height: int) -> Box:
    """Make a box bigger (`factor` > 0) or smaller (`factor` < 0) around its centre.

    The result is cut to the image and keeps at least `MIN_BOX_SIDE` pixels on each side.
    """
    assert -1.0 < factor < 1.0, "factor must be between -1 and 1"
    assert width >= MIN_BOX_SIDE and height >= MIN_BOX_SIDE, "image is too small"
    centre_x: float = (box.x0 + box.x1) / 2
    centre_y: float = (box.y0 + box.y1) / 2
    half_width: float = max(box.width * (1.0 + factor) / 2, MIN_BOX_SIDE / 2)
    half_height: float = max(box.height * (1.0 + factor) / 2, MIN_BOX_SIDE / 2)
    x0: int = max(0, round(centre_x - half_width))
    y0: int = max(0, round(centre_y - half_height))
    x1: int = min(width, round(centre_x + half_width))
    y1: int = min(height, round(centre_y + half_height))
    x1 = max(x1, min(width, x0 + MIN_BOX_SIDE))
    y1 = max(y1, min(height, y0 + MIN_BOX_SIDE))
    return Box(min(x0, x1 - MIN_BOX_SIDE), min(y0, y1 - MIN_BOX_SIDE), x1, y1)


def brighten(image: Image.Image, step: float) -> Image.Image:
    """Return a copy of the image that is brighter by `step` (0.15 means 15 % brighter)."""
    assert isinstance(image, Image.Image), "image must be a PIL image"
    assert step > 0, "step must be positive"
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(1.0 + step)
