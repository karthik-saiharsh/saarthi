"""Broad detector: finds objects that nobody registered, without a list of classes (YOLOE, prompt-free).

The detector gets no class names. It names each box from its own list of about 4600 names. A name can be
wrong: a box that shows a notebook may be called "passbook". So the name is a hint, and the backend
checks it before the voice says it. The detector is fast (about 0.5 s a frame on a CPU), so it can run on
every frame next to the registered-object detector (Grounding DINO).

The package `ultralytics` has the licence AGPL-3.0. The owner accepted it for this academic project.
The package can contact the internet (a check at import, anonymous usage data, an install of other
packages). This module switches that off before the import. The weights are downloaded one time from the
URL in `config.toml`, and the SHA-256 is checked before they are loaded. The pictures never leave the device.
"""

import hashlib
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from ai.errors import InferenceError, ModelLoadError
from ai.runtime import resolve_device
from ai.settings import AiConfig
from ai.types import Box, OpenDetection

logger: logging.Logger = logging.getLogger(__name__)

MODELS_DIR: Path = Path(__file__).resolve().parent / "models"
DOWNLOAD_TIMEOUT_SECONDS: int = 60
CHUNK_BYTES: int = 1024 * 1024
# The longest label that the module keeps.
MAX_LABEL_CHARS: int = 80


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(
    directory: Path, name: str, url: str, sha256: str, allow_download: bool
) -> Path:
    """Return the path of the weights file. Download it if needed. Check the SHA-256 each time.

    The file name is plain (the config checks it), so the file stays inside `directory`.
    """
    assert isinstance(directory, Path), "directory must be a Path"
    assert (
        name and "/" not in name and ".." not in name
    ), "name must be a plain file name"
    path: Path = directory / name
    if path.is_file():
        if _sha256_of(path) == sha256:
            return path
        raise ModelLoadError(f"The file {name} has the wrong SHA-256. Delete it.")
    if not allow_download:
        raise ModelLoadError(
            f"The file {name} is missing in {directory} and downloads are switched off"
        )
    directory.mkdir(parents=True, exist_ok=True)
    partial: Path = path.with_suffix(path.suffix + ".part")
    logger.info("Downloading the model file %s", name)
    try:
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            partial.open("wb") as file,
        ):
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                file.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        partial.unlink(missing_ok=True)
        raise ModelLoadError(f"Cannot download {name}: {error}") from error
    if _sha256_of(partial) != sha256:
        partial.unlink(missing_ok=True)
        raise ModelLoadError(f"The downloaded file {name} has the wrong SHA-256")
    partial.replace(path)
    return path


def import_ultralytics() -> tuple[Any, Any, Any]:
    """Import the parts of `ultralytics` that the detector uses. Return `(YOLOE, settings, LOGGER)`.

    The switches below must be set before the first import of the package.
    The import also replaces `PIL.Image.open` for the whole process. The new function tries to install a HEIF
    plugin and raises another error for a file that is not an image, so the upload step would not refuse such
    a file in the right way. This function puts the original back.
    """
    os.environ["YOLO_OFFLINE"] = "1"  # No online check, no download
    os.environ["YOLO_AUTOINSTALL"] = "0"  # No pip install at run time
    os.environ["YOLO_VERBOSE"] = "False"
    os.environ["YOLO_CONFIG_DIR"] = str(MODELS_DIR / "ultralytics-settings")
    original_open = Image.open
    try:
        # The import is late on purpose: it is slow, and the tests with fake models do not need it.
        from ultralytics import YOLOE, settings
        from ultralytics.utils import LOGGER
    finally:
        Image.open = original_open
    return YOLOE, settings, LOGGER


class OpenDetector:
    """Names the objects in a frame without a list of classes. Load it once."""

    def __init__(self, config: AiConfig) -> None:
        assert config is not None, "config is required"
        self._config: AiConfig = config
        self._device: str = (
            "cpu" if resolve_device(config.models.device) == "cpu" else "cuda:0"
        )
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        models = config.models
        weights: Path = ensure_weights(
            MODELS_DIR,
            models.open_detector_file,
            models.open_detector_url,
            models.open_detector_sha256,
            allow_download=not models.local_files_only,
        )
        try:
            YOLOE, settings, LOGGER = import_ultralytics()
            settings.update({"sync": False})  # No anonymous usage data
            LOGGER.setLevel(logging.WARNING)
            self._model = YOLOE(str(weights))
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
            logger.error("Cannot load the broad detector %s: %s", weights.name, error)
            raise ModelLoadError(f"Cannot load {weights.name}") from error
        self.name: str = f"{weights.stem}@{models.open_detector_sha256[:8]}"
        logger.info(
            "Broad detector %s ready on %s in %.1f s (%d names)",
            self.name,
            self._device,
            time.perf_counter() - started,
            len(self._model.names),
        )

    def detect(self, image: Image.Image) -> list[OpenDetection]:
        """Return the named boxes of a frame, the most confident first. The list can be empty."""
        assert isinstance(image, Image.Image), "image must be a PIL image"
        cfg = self._config.open_detection
        try:
            with self._lock:
                result = self._model.predict(
                    image.convert("RGB"),
                    conf=cfg.confidence,
                    imgsz=cfg.input_size,
                    device=self._device,
                    verbose=False,
                )[0]
            names: dict[int, str] = result.names
            found: list[OpenDetection] = []
            for item in result.boxes:
                x0, y0, x1, y1 = (round(v) for v in item.xyxy[0].tolist())
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(image.width, x1), min(image.height, y1)
                if min(x1 - x0, y1 - y0) < cfg.min_box_side:
                    continue
                label: str = " ".join(str(names[int(item.cls)]).split())[
                    :MAX_LABEL_CHARS
                ]
                found.append(
                    OpenDetection(Box(x0, y0, x1, y1), label, float(item.conf))
                )
        except (RuntimeError, ValueError, KeyError, IndexError) as error:
            logger.error("Broad detection failed: %s: %s", type(error).__name__, error)
            raise InferenceError("Broad detection failed") from error
        found.sort(key=lambda d: -d.confidence)
        found = found[: cfg.max_boxes]
        logger.info("Broad detector found %d box(es)", len(found))
        return found
