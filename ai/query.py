"""Query encoder: turns the words of a user into unit-length vectors.

The backend compares the vector of a question ("the blue thing I drink from")
with the vectors of the descriptions of the enrolled objects. The model is a
small sentence encoder (mean pooling over the last hidden state). It only
reads text. It never sees an image.
"""

import logging
import threading
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ai.errors import InferenceError, ModelLoadError
from ai.runtime import load_kwargs, resolve_device, short_model_name
from ai.settings import AiConfig

logger: logging.Logger = logging.getLogger(__name__)


class QueryEncoder:
    """Makes unit-length text vectors with a sentence encoder. Load it once."""

    def __init__(self, config: AiConfig) -> None:
        assert config is not None, "config is required"
        self._config: AiConfig = config
        self._device: str = resolve_device(config.models.device)
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        model_id: str = config.models.query_encoder_id
        revision: str = config.models.query_encoder_revision
        kwargs = load_kwargs(config.models, revision)
        try:
            # The tokenizer is not a weights file: it gets no `use_safetensors`.
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=config.models.local_files_only,
                trust_remote_code=False,
            )
            self._model = AutoModel.from_pretrained(model_id, **kwargs)
        except (OSError, ValueError) as error:
            logger.error("Cannot load the query encoder %s: %s", model_id, error)
            raise ModelLoadError(f"Cannot load {model_id}") from error
        self._model.eval().to(self._device)
        self.name: str = short_model_name(model_id, revision)
        self.dimension: int = int(self._model.config.hidden_size)
        logger.info(
            "Query encoder %s ready on %s in %.1f s, dimension %d",
            self.name,
            self._device,
            time.perf_counter() - started,
            self.dimension,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one unit-length float32 vector for each text, in the same order."""
        assert isinstance(texts, list), "texts must be a list"
        assert all(isinstance(t, str) for t in texts), "every text must be a string"
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            batch = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._config.query.max_tokens,
                return_tensors="pt",
            ).to(self._device)
            with self._lock, torch.inference_mode():
                hidden: torch.Tensor = self._model(**batch).last_hidden_state
            # Mean over the real tokens only. The padding must not count.
            mask: torch.Tensor = batch["attention_mask"].unsqueeze(-1).float()
            pooled: torch.Tensor = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            result: np.ndarray = (
                torch.nn.functional.normalize(pooled, dim=1).float().cpu().numpy()
            )
        except (RuntimeError, ValueError) as error:
            logger.error("Encoding failed: %s: %s", type(error).__name__, error)
            raise InferenceError("Encoding failed") from error
        logger.debug("Encoded %d text(s)", len(texts))
        return result.astype(np.float32)
