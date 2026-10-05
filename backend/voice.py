"""The voice input (stage Q0): a recording of the user becomes text.

The browser sends a WAV file: 16-bit samples, one channel, 16 kHz. The service checks the header and the
size, turns the samples into numbers and gives them to the speech recognizer. The audio stays in memory.
It is never written to a file and never logged. The text is a private question: the log shows its length only.
"""

import io
import logging
import wave

import numpy as np

from ai.api import AiEngine
from backend.errors import InvalidAudioError, VoiceUnavailableError
from backend.settings import BackendConfig, VoiceConfig

logger: logging.Logger = logging.getLogger(__name__)

# Fixed messages. They contain no detail of the recording.
MESSAGE_FORMAT: str = "The recording is not a usable WAV file."
MESSAGE_TOO_SHORT: str = "The recording is too short."
MESSAGE_TOO_LONG: str = "The recording is too long."
MESSAGE_OFF: str = "Listening is switched off."

# Size of the WAV header, and a small extra space for other chunks.
WAV_OVERHEAD_BYTES: int = 1024
BYTES_PER_SAMPLE: int = 2


class VoiceService:
    """Turns a recording into text. It is safe to call from several threads."""

    def __init__(self, config: BackendConfig, engine: AiEngine) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: VoiceConfig = config.voice
        self._engine: AiEngine = engine

    @property
    def enabled(self) -> bool:
        return self._engine.speech_enabled

    @property
    def max_seconds(self) -> int:
        return self._engine.speech_max_seconds

    @property
    def max_bytes(self) -> int:
        """The largest WAV file that `transcribe` accepts."""
        return (
            self.max_seconds * self._engine.speech_sample_rate * BYTES_PER_SAMPLE
            + WAV_OVERHEAD_BYTES
        )

    def transcribe(self, wav: bytes) -> str:
        """Return the words in the recording. The text is empty for a silent recording."""
        assert isinstance(wav, bytes), "wav must be bytes"
        if not self.enabled:
            raise VoiceUnavailableError(MESSAGE_OFF)
        if len(wav) > self.max_bytes:
            raise InvalidAudioError(MESSAGE_TOO_LONG)
        samples: np.ndarray = self._decode(wav)
        rate: int = self._engine.speech_sample_rate
        if samples.size < self._config.min_seconds * rate:
            raise InvalidAudioError(MESSAGE_TOO_SHORT)
        if samples.size > self.max_seconds * rate:
            raise InvalidAudioError(MESSAGE_TOO_LONG)
        level: float = float(np.sqrt(np.mean(np.square(samples))))
        if level < self._config.silence_rms:
            logger.info("The recording is silent")
            return ""
        return self._engine.transcribe(samples)

    def _decode(self, wav: bytes) -> np.ndarray:
        """Return the samples as floats between -1 and 1. Refuse any other format."""
        try:
            with wave.open(io.BytesIO(wav), "rb") as reader:
                if (
                    reader.getnchannels() != 1
                    or reader.getsampwidth() != BYTES_PER_SAMPLE
                    or reader.getframerate() != self._engine.speech_sample_rate
                    or reader.getcomptype() != "NONE"
                ):
                    raise InvalidAudioError(MESSAGE_FORMAT)
                # Read no more than the limit, so a wrong header cannot make a big array.
                limit: int = self.max_seconds * self._engine.speech_sample_rate + 1
                raw: bytes = reader.readframes(min(reader.getnframes(), limit))
        except (wave.Error, EOFError) as error:
            raise InvalidAudioError(MESSAGE_FORMAT) from error
        pcm: np.ndarray = np.frombuffer(raw, dtype="<i2")
        return (pcm.astype(np.float32) / 32768.0).astype(np.float32)
