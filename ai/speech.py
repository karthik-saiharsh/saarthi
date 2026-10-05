"""Speech recognizer (stage Q0): the speech of a user becomes text.

A Whisper model reads samples of audio and writes English text. The audio stays in memory.
It is never written to a file and never logged. The text is a private question: the log shows its
length only. The model runs on the device. Nothing goes to another service.
"""

import logging
import threading
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ai.errors import InferenceError, ModelLoadError
from ai.runtime import load_kwargs, resolve_device, short_model_name
from ai.settings import AiConfig

logger: logging.Logger = logging.getLogger(__name__)

# The model reads audio of this rate only.
SAMPLE_RATE: int = 16000


class SpeechRecognizer:
    """Turns audio samples into text. Load it once."""

    def __init__(self, config: AiConfig) -> None:
        assert config is not None, "config is required"
        self._config: AiConfig = config
        self._device: str = resolve_device(config.models.device)
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        model_id: str = config.models.speech_id
        revision: str = config.models.speech_revision
        try:
            self._processor = WhisperProcessor.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=config.models.local_files_only,
                trust_remote_code=False,
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(
                model_id, **load_kwargs(config.models, revision)
            )
        except (OSError, ValueError) as error:
            logger.error("Cannot load the speech model %s: %s", model_id, error)
            raise ModelLoadError(f"Cannot load {model_id}") from error
        self._model.eval().to(self._device)
        self.name: str = short_model_name(model_id, revision)
        logger.info(
            "Speech model %s ready on %s in %.1f s",
            self.name,
            self._device,
            time.perf_counter() - started,
        )

    def transcribe(self, samples: np.ndarray) -> str:
        """Return the words in the samples. The text is empty if the model hears no speech.

        `samples` are floats between -1 and 1, one channel, at 16 kHz.
        """
        assert isinstance(samples, np.ndarray), "samples must be a numpy array"
        assert samples.ndim == 1 and samples.dtype == np.float32, "need float32 mono"
        max_samples: int = self._config.speech.max_seconds * SAMPLE_RATE
        assert 0 < samples.size <= max_samples, "samples out of range"
        try:
            features = self._processor(
                samples, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(self._device)
            with self._lock, torch.inference_mode():
                tokens = self._model.generate(
                    features,
                    language="en",
                    task="transcribe",
                    max_new_tokens=self._config.speech.max_new_tokens,
                )
            text: str = self._processor.batch_decode(tokens, skip_special_tokens=True)[
                0
            ]
        except (RuntimeError, ValueError) as error:
            logger.error(
                "Speech recognition failed: %s: %s", type(error).__name__, error
            )
            raise InferenceError("Speech recognition failed") from error
        words: str = " ".join(text.split())
        logger.info("Speech recognized: %d characters", len(words))
        return words
