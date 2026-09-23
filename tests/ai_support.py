"""Fake ai models and an engine for tests that need the flow but not the real models."""

import numpy as np

from ai.api import AiEngine
from ai.settings import load_config
from ai.types import RawDetection


class FakeDetector:
    """Returns the detections that the test sets. It records the switches of each call."""

    name = "fake-detector@00000000"

    def __init__(self) -> None:
        self.generic_flags: list[bool] = []
        self.detections: list[RawDetection] = []
        self.error: Exception | None = None

    def detect(
        self, image, phrases, small_phrases=(), include_generic=False, tiling=None
    ):
        self.generic_flags.append(include_generic)
        if self.error is not None:
            error, self.error = self.error, None  # Fail one time only
            raise error
        return list(self.detections)


class FakeEmbedder:
    """Returns one unit vector for each box. The size is `dimension`."""

    name = "fake-embedder@00000000"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension: int = dimension
        self.error: Exception | None = None

    def embed(self, image, boxes):
        if self.error is not None:
            raise self.error
        vectors: np.ndarray = np.zeros((len(boxes), self.dimension), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def make_engine(
    detector: FakeDetector | None = None, embedder: FakeEmbedder | None = None
) -> AiEngine:
    """An `AiEngine` with fake models. The quality gate and post-processing are real."""
    return AiEngine(
        load_config(),
        detector=detector or FakeDetector(),  # type: ignore[arg-type]
        embedder=embedder or FakeEmbedder(),  # type: ignore[arg-type]
    )
