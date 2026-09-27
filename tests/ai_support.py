"""Fake ai models and an engine for tests that need the flow but not the real models."""

import dataclasses

import numpy as np

from ai.api import AiEngine
from ai.settings import load_config
from ai.types import RawDetection


class FakeDetector:
    """Returns the detections that the test sets. It records the switches of each call."""

    name = "fake-detector@00000000"

    def __init__(self) -> None:
        self.generic_flags: list[bool] = []
        self.phrase_calls: list[list[str]] = []
        self.context_flags: list[bool] = []
        self.detections: list[RawDetection] = []
        self.error: Exception | None = None

    def detect(
        self,
        image,
        phrases,
        small_phrases=(),
        include_generic=False,
        tiling=None,
        context_phrases=True,
    ):
        self.generic_flags.append(include_generic)
        self.phrase_calls.append(list(phrases))
        self.context_flags.append(context_phrases)
        if self.error is not None:
            error, self.error = self.error, None  # Fail one time only
            raise error
        return list(self.detections)


class FakeEmbedder:
    """Returns one unit vector for each box. The size is `dimension`."""

    name = "fake-embedder@00000000"

    def __init__(self, dimension: int = 384, one_hot_by_box: bool = False) -> None:
        self.dimension: int = dimension
        self.error: Exception | None = None
        # True: each box gets its own vector (one-hot at a place that depends on the box).
        self.one_hot_by_box: bool = one_hot_by_box
        self.calls: list[tuple[float, list]] = (
            []
        )  # The mean brightness of the image, the boxes

    def embed(self, image, boxes):
        if self.error is not None:
            raise self.error
        self.calls.append((float(np.asarray(image.convert("L")).mean()), list(boxes)))
        vectors: np.ndarray = np.zeros((len(boxes), self.dimension), dtype=np.float32)
        for row, box in enumerate(boxes):
            column: int = (
                (box.x0 + 7 * box.y0) % self.dimension if self.one_hot_by_box else 0
            )
            vectors[row, column] = 1.0
        return vectors


def make_engine(
    detector: FakeDetector | None = None,
    embedder: FakeEmbedder | None = None,
    blur_min: float | None = None,
) -> AiEngine:
    """An `AiEngine` with fake models. The quality gate and post-processing are real.

    `blur_min` replaces the limit of the config (the shipped config has the gate off).
    """
    config = load_config()
    if blur_min is not None:
        config = dataclasses.replace(
            config, quality=dataclasses.replace(config.quality, blur_min=blur_min)
        )
    return AiEngine(
        config,
        detector=detector or FakeDetector(),  # type: ignore[arg-type]
        embedder=embedder or FakeEmbedder(),  # type: ignore[arg-type]
    )
