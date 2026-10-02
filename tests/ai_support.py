"""Fake ai models and an engine for tests that need the flow but not the real models."""

import dataclasses
import re
import zlib

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


class FakeQueryEncoder:
    """Hashes the words of a text into a vector, so texts that share words are close.

    Deterministic across runs. It is a stand-in for a real sentence encoder.
    """

    name = "fake-query-encoder@00000000"

    def __init__(self, dimension: int = 64) -> None:
        self.dimension: int = dimension
        self.error: Exception | None = None
        self.calls: list[list[str]] = []

    def encode(self, texts):
        if self.error is not None:
            raise self.error
        self.calls.append(list(texts))
        vectors: np.ndarray = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9']+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % self.dimension] += 1.0
            norm: float = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class FakeVerifier:
    """Returns a number of matching keypoints that the test sets. It records the calls.

    `answers` is a queue: each call takes the first number. When it is empty, the call gives `answer`.
    """

    name = "fake-verifier@00000000"

    def __init__(self, answer: int = 100) -> None:
        self.answer: int = answer
        self.answers: list[int] = []
        self.error: Exception | None = None
        self.calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def count_matches(self, first, second) -> int:
        if self.error is not None:
            raise self.error
        self.calls.append((first.size, second.size))
        return self.answers.pop(0) if self.answers else self.answer


def make_engine(
    detector: FakeDetector | None = None,
    embedder: FakeEmbedder | None = None,
    blur_min: float | None = None,
    query_encoder: FakeQueryEncoder | None = None,
    verifier: FakeVerifier | None = None,
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
        query_encoder=query_encoder or FakeQueryEncoder(),  # type: ignore[arg-type]
        verifier=verifier or FakeVerifier(),  # type: ignore[arg-type]
    )
