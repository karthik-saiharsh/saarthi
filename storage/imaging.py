"""Prepare an uploaded image for storage. No disk access.

The result is a JPEG with the orientation applied and without any metadata
(EXIF, GPS, XMP, ICC profile, comments, thumbnails). It is built from the raw
pixels of the decoded image, so nothing else can be copied into the file.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from storage.errors import ImageTooLargeError, InvalidCropError, UnsupportedImageError
from storage.settings import StorageConfig

logger: logging.Logger = logging.getLogger(__name__)

# Formats are accepted by their content, never by the file name or the content type.
ACCEPTED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})
# Colour modes that convert to RGB without a loss of meaning. 16-bit images are refused.
ACCEPTED_MODES: frozenset[str] = frozenset(
    {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}
)
WHITE: tuple[int, int, int, int] = (255, 255, 255, 255)


@dataclass(frozen=True)
class PreparedImage:
    """An image that is ready to be written to disk."""

    data: bytes = field(repr=False)  # The JPEG file
    sha256: str  # Hash of `data`
    width: int  # Pixels of the saved image, after the orientation was applied
    height: int


def _open(data: bytes, config: StorageConfig) -> Image.Image:
    """Open and fully decode the image. Raise an error if it is not usable."""
    if not data:
        raise UnsupportedImageError("The file is empty")
    if len(data) > config.max_bytes:
        raise ImageTooLargeError(f"The file is larger than {config.max_bytes} bytes")
    try:
        image: Image.Image = Image.open(BytesIO(data))
    except Image.DecompressionBombError as error:
        raise ImageTooLargeError("The image has too many pixels") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise UnsupportedImageError(
            "The file is not an image that we can read"
        ) from error

    if image.format not in ACCEPTED_FORMATS:
        raise UnsupportedImageError(f"The image format {image.format} is not accepted")
    if image.width * image.height > config.max_pixels:
        raise ImageTooLargeError(f"The image has more than {config.max_pixels} pixels")
    if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) > 1:
        raise UnsupportedImageError("Animated images are not accepted")
    if image.mode not in ACCEPTED_MODES:
        raise UnsupportedImageError(f"The colour mode {image.mode} is not accepted")
    try:
        image.load()
    except Image.DecompressionBombError as error:
        raise ImageTooLargeError("The image has too many pixels") from error
    except (OSError, ValueError, SyntaxError, EOFError) as error:
        # A truncated or damaged file.
        raise UnsupportedImageError("The image file is damaged") from error
    return image


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert to RGB. A transparent area becomes white."""
    has_alpha: bool = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba: Image.Image = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, WHITE), rgba).convert(
            "RGB"
        )
    return image.convert("RGB")


def prepare_image(data: bytes, config: StorageConfig) -> PreparedImage:
    """Check an upload and make the JPEG that storage writes.

    The same input always gives the same bytes, so an exact duplicate upload
    has the same hash.

    Raise `UnsupportedImageError` for a file that is not a JPEG, PNG or WebP
    image, or is animated or damaged. Raise `ImageTooLargeError` if it has too
    many bytes or pixels.
    """
    assert isinstance(data, bytes), "data must be bytes"
    assert isinstance(config, StorageConfig), "config must be a StorageConfig"

    image: Image.Image = _open(data, config)
    source_format: str = str(image.format)
    rgb: Image.Image = _to_rgb(ImageOps.exif_transpose(image))
    # A new image from raw pixels has no `info`, so `save` cannot copy metadata.
    clean: Image.Image = Image.frombytes("RGB", rgb.size, rgb.tobytes())

    buffer: BytesIO = BytesIO()
    clean.save(buffer, format="JPEG", quality=config.jpeg_quality, optimize=True)
    output: bytes = buffer.getvalue()
    logger.info(
        "Prepared a %s image of %dx%d: %d bytes in, %d bytes out",
        source_format,
        clean.width,
        clean.height,
        len(data),
        len(output),
    )
    return PreparedImage(
        data=output,
        sha256=hashlib.sha256(output).hexdigest(),
        width=clean.width,
        height=clean.height,
    )


def crop_jpeg(
    data: bytes, box: tuple[int, int, int, int], max_side: int, config: StorageConfig
) -> bytes:
    """Return a JPEG of a part of a stored image, for example for the confirmation screen.

    `box` is (x0, y0, x1, y1) in pixels of the stored image. The longest side of
    the result is at most `max_side`. A small crop is not made larger. The new
    file has no metadata. Raise `InvalidCropError` if the box is not inside the image.
    """
    assert isinstance(data, bytes) and data, "data must be non-empty bytes"
    assert len(box) == 4 and all(
        isinstance(v, int) for v in box
    ), "box needs 4 integers"
    assert isinstance(max_side, int) and max_side > 0, "max_side must be positive"
    assert isinstance(config, StorageConfig), "config must be a StorageConfig"
    try:
        image: Image.Image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise UnsupportedImageError(
            "The stored file is not an image that we can read"
        ) from error
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise InvalidCropError("The box is not inside the image")

    crop: Image.Image = image.convert("RGB").crop(box)
    if max(crop.size) > max_side:
        crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    clean: Image.Image = Image.frombytes("RGB", crop.size, crop.tobytes())
    buffer: BytesIO = BytesIO()
    clean.save(buffer, format="JPEG", quality=config.jpeg_quality, optimize=True)
    return buffer.getvalue()
