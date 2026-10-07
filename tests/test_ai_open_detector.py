"""The broad detector: the weights file, the checks, and the way that boxes are cleaned. No real model."""

import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ai.errors import InferenceError, ModelLoadError
from ai.open_detector import OpenDetector, ensure_weights, import_ultralytics
from ai.settings import load_config
from tests.ai_support import FakeOpenDetector, make_engine

CONFIG = load_config()
DATA = b"weights of a test"
DIGEST = hashlib.sha256(DATA).hexdigest()


# ---- The weights file ----


def test_a_good_file_is_used_without_a_download(tmp_path: Path) -> None:
    (tmp_path / "w.pt").write_bytes(DATA)
    assert (
        ensure_weights(tmp_path, "w.pt", "https://x", DIGEST, False)
        == tmp_path / "w.pt"
    )


def test_a_file_with_a_wrong_hash_is_refused(tmp_path: Path) -> None:
    (tmp_path / "w.pt").write_bytes(b"changed")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        ensure_weights(tmp_path, "w.pt", "https://x", DIGEST, True)


def test_a_missing_file_is_not_downloaded_when_downloads_are_off(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelLoadError, match="switched off"):
        ensure_weights(tmp_path, "w.pt", "https://x", DIGEST, False)
    assert not list(tmp_path.iterdir())


def test_a_download_is_checked_and_kept(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(DATA)
    target = tmp_path / "models"
    path = ensure_weights(target, "w.pt", source.as_uri(), DIGEST, True)
    assert path.read_bytes() == DATA and not list(target.glob("*.part"))


def test_a_download_with_a_wrong_hash_leaves_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"other data")
    target = tmp_path / "models"
    with pytest.raises(ModelLoadError, match="SHA-256"):
        ensure_weights(target, "w.pt", source.as_uri(), DIGEST, True)
    assert not list(target.iterdir())


def test_a_failed_download_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="download"):
        ensure_weights(tmp_path, "w.pt", (tmp_path / "none.pt").as_uri(), DIGEST, True)


# ---- The cleaning of the boxes ----


class FakeBox:
    def __init__(self, xyxy: list[float], cls: int, conf: float) -> None:
        self.xyxy = [SimpleNamespace(tolist=lambda: xyxy)]
        self.cls = cls
        self.conf = conf


def make_detector(boxes: list[FakeBox], names: dict[int, str]) -> OpenDetector:
    detector = object.__new__(OpenDetector)  # No model is loaded
    detector._config = CONFIG
    detector._device = "cpu"
    detector._lock = threading.Lock()
    result = SimpleNamespace(boxes=boxes, names=names)
    detector._model = SimpleNamespace(predict=lambda *a, **k: [result])
    return detector


NAMES = {0: "laptop", 1: "  coffee   cup ", 2: "x" * 300}
IMAGE = Image.new("RGB", (400, 300))


def test_boxes_are_clipped_cleaned_and_sorted() -> None:
    boxes = [
        FakeBox([10.4, 20.2, 110.6, 120.0], 1, 0.4),
        FakeBox([-5, -5, 405, 310], 0, 0.9),
        FakeBox([50, 50, 60, 200], 0, 0.8),  # The short side is 10 px: dropped
    ]
    found = make_detector(boxes, NAMES).detect(IMAGE)
    assert [d.label for d in found] == ["laptop", "coffee cup"]
    assert (found[0].box.x0, found[0].box.y0, found[0].box.x1, found[0].box.y1) == (
        0,
        0,
        400,
        300,
    )
    assert (found[1].box.x0, found[1].box.x1) == (10, 111)
    assert found[0].confidence == 0.9


def test_a_long_label_is_cut() -> None:
    found = make_detector([FakeBox([0, 0, 100, 100], 2, 0.5)], NAMES).detect(IMAGE)
    assert len(found[0].label) == 80


def test_only_the_most_confident_boxes_are_kept() -> None:
    boxes = [FakeBox([0, 0, 100, 100], 0, 0.3 + i / 1000) for i in range(30)]
    found = make_detector(boxes, NAMES).detect(IMAGE)
    assert len(found) == CONFIG.open_detection.max_boxes
    assert found[0].confidence >= found[-1].confidence


def test_no_box_gives_an_empty_list() -> None:
    assert make_detector([], NAMES).detect(IMAGE) == []


def test_a_model_error_becomes_an_inference_error() -> None:
    detector = make_detector([], NAMES)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret detail")

    detector._model = SimpleNamespace(predict=fail)
    with pytest.raises(InferenceError) as caught:
        detector.detect(IMAGE)
    assert "secret" not in str(caught.value)


# ---- The engine ----


def test_the_engine_gives_the_boxes_of_the_detector() -> None:
    fake = FakeOpenDetector(["a box"])
    engine = make_engine(open_detector=fake)
    assert engine.open_detection_enabled
    assert engine.detect_open(IMAGE) == ["a box"]
    assert fake.calls == [(400, 300)]


def test_a_switched_off_detector_is_reported() -> None:
    engine = make_engine()
    engine._open_detector = None
    assert not engine.open_detection_enabled
    with pytest.raises(AssertionError, match="switched off"):
        engine.detect_open(IMAGE)


def test_the_import_of_ultralytics_does_not_replace_the_image_opener() -> None:
    # The package replaces `PIL.Image.open`. The upload step needs the original error for a file that is not an image.
    import io

    import PIL.Image

    original = PIL.Image.open
    import_ultralytics()
    assert PIL.Image.open is original
    with pytest.raises(PIL.Image.UnidentifiedImageError):
        PIL.Image.open(io.BytesIO(b"this is not an image"))
