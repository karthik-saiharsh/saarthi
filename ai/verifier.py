"""Keypoint verifier (stage I5): do two pictures show the same object?

LightGlue with SuperPoint finds keypoints in both pictures and matches them.
Many matches mean that both pictures show the same textured thing. A few
matches mean that they do not. The model never decides alone: the engine
compares the count with `verification.min_matches`.

LICENCE: the weights are for academic or non-profit non-commercial research
use only. See the comment in `config.toml` and `docs/backend/Verification.md`.
"""

import logging
import threading
import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ai.errors import InferenceError, ModelLoadError
from ai.runtime import load_kwargs, resolve_device, short_model_name
from ai.settings import AiConfig

logger: logging.Logger = logging.getLogger(__name__)


class KeypointVerifier:
    """Counts the matching keypoints of two pictures. Load it once."""

    def __init__(self, config: AiConfig) -> None:
        assert config is not None, "config is required"
        self._config: AiConfig = config
        self._device: str = resolve_device(config.models.device)
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        model_id: str = config.models.verifier_id
        revision: str = config.models.verifier_revision
        kwargs = load_kwargs(config.models, revision)
        try:
            # The processor is not a weights file: it gets no `use_safetensors`.
            self._processor = AutoImageProcessor.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=config.models.local_files_only,
                trust_remote_code=False,
            )
            self._model = AutoModel.from_pretrained(model_id, **kwargs)
        except (OSError, ValueError) as error:
            logger.error("Cannot load the verifier %s: %s", model_id, error)
            raise ModelLoadError(f"Cannot load {model_id}") from error
        self._model.eval().to(self._device)
        self.name: str = short_model_name(model_id, revision)
        logger.info(
            "Verifier %s ready on %s in %.1f s",
            self.name,
            self._device,
            time.perf_counter() - started,
        )

    def count_matches(self, first: Image.Image, second: Image.Image) -> int:
        """Return the number of keypoints that match between two pictures."""
        assert isinstance(first, Image.Image), "first must be a PIL image"
        assert isinstance(second, Image.Image), "second must be a PIL image"
        pair: list[Image.Image] = [first.convert("RGB"), second.convert("RGB")]
        try:
            inputs = self._processor(pair, return_tensors="pt").to(self._device)
            with self._lock, torch.inference_mode():
                outputs = self._model(**inputs)
            result: dict = self._processor.post_process_keypoint_matching(
                outputs,
                [[(p.height, p.width) for p in pair]],
                threshold=self._config.verification.match_threshold,
            )[0]
            count: int = int(result["keypoints0"].shape[0])
        except (RuntimeError, ValueError, KeyError) as error:
            logger.error("Verification failed: %s: %s", type(error).__name__, error)
            raise InferenceError("Verification failed") from error
        logger.debug("The verifier found %d matching keypoint(s)", count)
        return count
