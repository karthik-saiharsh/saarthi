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


class FakeSceneClassifier:
    """Gives the scores that the test sets. It records the size of each picture that it gets."""

    name = "fake-scene@00000000"

    def __init__(
        self,
        rooms: dict[str, float] | None = None,
        surfaces: list[tuple[str, float]] | None = None,
    ) -> None:
        scene = load_config().scene
        even: float = 1.0 / len(scene.rooms)
        self.rooms: dict[str, float] = rooms or {room: even for room in scene.rooms}
        chosen = surfaces or [
            (surface, 1.0 / len(scene.surfaces)) for surface in scene.surfaces
        ]
        self.surfaces: list[tuple[str, float]] = sorted(
            chosen, key=lambda item: -item[1]
        )
        self.error: Exception | None = None
        # Rooms that the next calls give, one entry for each call. When it is empty, the call gives `rooms`.
        self.room_queue: list[dict[str, float]] = []
        self.room_calls: list[tuple[int, int]] = []
        self.surface_calls: list[tuple[int, int]] = []
        self.crop_calls: list[list[tuple[int, int]]] = []
        self.phrase_calls: list[list[str]] = []
        # The colour of the top-left pixel of a fake crop gives its phrase.
        self.crop_labels: dict[tuple[int, int, int], str] = {}

    def room_scores(self, image):
        if self.error is not None:
            raise self.error
        self.room_calls.append(image.size)
        if self.room_queue:
            return dict(self.room_queue.pop(0))
        return dict(self.rooms)

    def surface_scores(self, image):
        if self.error is not None:
            raise self.error
        self.surface_calls.append(image.size)
        return list(self.surfaces)

    # ---- Vectors for the open search ----

    dimension = 8

    def _vector(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        vector = rng.normal(size=self.dimension).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def embed_crops(self, crops):
        if self.error is not None:
            raise self.error
        self.crop_calls.append([c.size for c in crops])
        # A crop that is a flat colour gets a vector from that colour, so a test can set which phrase fits it.
        return (
            np.stack([self._vector(self.crop_label(c)) for c in crops])
            if crops
            else np.empty((0, self.dimension), dtype=np.float32)
        )

    def crop_label(self, crop) -> str:
        """The phrase that a fake crop shows. A test paints the crop with a colour and maps the colour here."""
        return self.crop_labels.get(crop.convert("RGB").getpixel((0, 0)), "unknown")

    def embed_phrases(self, phrases):
        if self.error is not None:
            raise self.error
        self.phrase_calls.append(list(phrases))
        return np.stack([self._vector(p) for p in phrases])

    def pair_probabilities(self, crops, phrases):
        cosine = crops @ phrases.T
        return (1.0 / (1.0 + np.exp(-(cosine * 20.0 - 10.0)))).astype(np.float32)


class FakeSpeechRecognizer:
    """Gives the text that the test sets. It records the length of each recording."""

    name = "fake-speech@00000000"

    def __init__(self, text: str = "where is my bottle") -> None:
        self.text: str = text
        self.error: Exception | None = None
        self.calls: list[int] = []

    def transcribe(self, samples):
        if self.error is not None:
            raise self.error
        self.calls.append(int(samples.size))
        return self.text


class FakeOpenDetector:
    """Gives the named boxes that the test sets. It records the size of each frame."""

    name = "fake-open@00000000"

    def __init__(self, detections: list | None = None) -> None:
        self.detections: list = detections or []
        self.error: Exception | None = None
        self.calls: list[tuple[int, int]] = []

    def detect(self, image):
        if self.error is not None:
            raise self.error
        self.calls.append(image.size)
        return list(self.detections)


def make_engine(
    detector: FakeDetector | None = None,
    embedder: FakeEmbedder | None = None,
    blur_min: float | None = None,
    query_encoder: FakeQueryEncoder | None = None,
    verifier: FakeVerifier | None = None,
    scene_classifier: FakeSceneClassifier | None = None,
    speech_recognizer: FakeSpeechRecognizer | None = None,
    open_detector: FakeOpenDetector | None = None,
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
        scene_classifier=scene_classifier or FakeSceneClassifier(),  # type: ignore[arg-type]
        speech_recognizer=speech_recognizer or FakeSpeechRecognizer(),  # type: ignore[arg-type]
        open_detector=open_detector or FakeOpenDetector(),  # type: ignore[arg-type]
    )
