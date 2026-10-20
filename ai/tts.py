"""Speech synthesizer (stage Q3 on the server): a sentence becomes speech, for the speaker of the wearable.

The page does not use it: the browser speaks with a local voice of the device. The wearable has no voice of its own.
The model is Piper with one voice (`[models] voice_*` in `config.toml`). It runs on the CPU with ONNX Runtime.
The text is the answer to a private question, so it is never logged and the audio is never written to a file.
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

from ai.errors import InferenceError, ModelLoadError
from ai.open_detector import MODELS_DIR, ensure_weights
from ai.settings import AiConfig

logger: logging.Logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """Turns English text into 16-bit mono PCM samples. Load it once. It is safe to call from several threads."""

    def __init__(self, config: AiConfig, directory: Path = MODELS_DIR) -> None:
        assert config is not None, "config is required"
        assert isinstance(directory, Path), "directory must be a Path"
        models = config.models
        self._length_scale: float = config.tts.length_scale
        self._max_chars: int = config.tts.max_chars
        self._lock: threading.Lock = threading.Lock()
        started: float = time.perf_counter()
        allow: bool = not models.local_files_only
        voice_path: Path = ensure_weights(
            directory, models.voice_file, models.voice_url, models.voice_sha256, allow
        )
        config_path: Path = ensure_weights(
            directory,
            models.voice_file + ".json",
            models.voice_config_url,
            models.voice_config_sha256,
            allow,
        )
        try:
            # The download folder of Piper is for the Chinese phonemizer only. It points into ai/models, so
            # nothing can be written to the folder where the application starts.
            self._voice: PiperVoice = PiperVoice.load(
                voice_path, config_path, download_dir=directory
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            logger.error("Cannot load the voice %s: %s", models.voice_file, error)
            raise ModelLoadError(
                f"Cannot load the voice {models.voice_file}"
            ) from error
        self.sample_rate: int = int(self._voice.config.sample_rate)
        self.name: str = f"{voice_path.stem}@{models.voice_sha256[:8]}"
        logger.info(
            "Voice %s ready in %.1f s (%d Hz)",
            self.name,
            time.perf_counter() - started,
            self.sample_rate,
        )

    def synthesize(self, text: str) -> np.ndarray:
        """Return the speech of the text as int16 samples (mono, `sample_rate`). A text above `tts.max_chars` is cut."""
        assert isinstance(text, str), "text must be a string"
        words: str = " ".join(text.split())[: self._max_chars]
        if not words:
            return np.empty(0, dtype=np.int16)
        started: float = time.perf_counter()
        settings: SynthesisConfig = SynthesisConfig(length_scale=self._length_scale)
        try:
            with self._lock:
                parts: list[np.ndarray] = [
                    chunk.audio_int16_array.reshape(-1)
                    for chunk in self._voice.synthesize(words, settings)
                ]
        except (RuntimeError, ValueError) as error:
            logger.error("Speech synthesis failed: %s", type(error).__name__)
            raise InferenceError("Speech synthesis failed") from error
        samples: np.ndarray = (
            np.concatenate(parts).astype(np.int16)
            if parts
            else np.empty(0, dtype=np.int16)
        )
        logger.info(
            "Synthesized %.1f s of speech in %.2f s",
            samples.size / self.sample_rate,
            time.perf_counter() - started,
        )
        return samples
