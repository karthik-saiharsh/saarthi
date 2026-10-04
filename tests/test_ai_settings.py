import re
from pathlib import Path

import pytest

from ai.errors import AiConfigError
from ai.settings import DEFAULT_CONFIG_PATH, PATCH_SIZE, AiConfig, load_config

DEFAULT_TEXT: str = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def write(tmp_path: Path, text: str) -> Path:
    path: Path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def replace(text: str, old: str, new: str) -> str:
    assert old in text, f"test setup: {old!r} not in the default config"
    return text.replace(old, new, 1)


def test_default_config_loads_and_is_consistent() -> None:
    config: AiConfig = load_config()
    assert config.embedding.frame_short_edge % PATCH_SIZE == 0
    assert set(config.detection.small_phrases) <= set(config.detection.default_phrases)
    assert re.fullmatch(r"[0-9a-f]{40}", config.models.detector_revision)
    assert re.fullmatch(r"[0-9a-f]{40}", config.models.embedder_revision)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('device = "auto"', 'device = "tpu"'),
        ("cpu_threads = 0", "cpu_threads = -1"),
        ("local_files_only = false", 'local_files_only = "no"'),
        ("box_threshold = 0.25", "box_threshold = 1.5"),
        ("chunk_size = 30", "chunk_size = 0"),
        ("min_matches = 30", "min_matches = 0"),
        ("match_threshold = 0.2", "match_threshold = 1.5"),
        ("enabled = true", 'enabled = "yes"'),
        (
            'augmentations = ["original", "box_smaller", "box_larger", "brighter"]',
            'augmentations = ["original", "flip"]',
        ),
        (
            'augmentations = ["original", "box_smaller", "box_larger", "brighter"]',
            'augmentations = ["original", "original"]',
        ),
        (
            'augmentations = ["original", "box_smaller", "box_larger", "brighter"]',
            "augmentations = []",
        ),
        ("box_scale = 0.10", "box_scale = 0.9"),
        ("brightness = 0.15", "brightness = 0"),
        ("tau_std_factor = 0.5", "tau_std_factor = -1"),
        ("tau_min = 0.45", "tau_min = 1.5"),
        ("surface_context = 1.0", "surface_context = -1.0"),
        ('prompt = "a photo of {label}"', 'prompt = "a photo of a room"'),
        ('  "a bedroom",\n  "a living room",', '  "a bedroom",\n  "a bedroom",'),
        ("tau_min = 0.45", 'tau_min = "high"'),
        ("margin_strong = 0.08", "margin_strong = 0.01"),
        ("min_photos = 3", "min_photos = 1"),
        ("chunk_size = 30", "chunk_size = true"),
        ("frame_short_edge = 448", "frame_short_edge = 450"),
        ("frame_long_edge_max = 784", "frame_long_edge_max = 434"),
        ('hand_phrase = "a hand"', 'hand_phrase = ""'),
        (
            'small_phrases = ["a set of keys", "a watch", "a pill box", "a hearing aid"]',
            'small_phrases = ["a spaceship"]',
        ),
        ("default_phrases = [", "default_phrases = []\nold_phrases = ["),
        ('detector_id = "IDEA-Research/grounding-dino-tiny"', "detector_id = 5"),
    ],
)
def test_bad_values_are_refused(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(AiConfigError):
        load_config(write(tmp_path, replace(DEFAULT_TEXT, old, new)))


@pytest.mark.parametrize(
    "table",
    [
        "models",
        "image",
        "quality",
        "detection",
        "embedding",
        "enrollment",
        "verification",
        "scene",
    ],
)
def test_missing_table_is_refused(tmp_path: Path, table: str) -> None:
    text: str = DEFAULT_TEXT.replace(f"[{table}]", f"[{table}_renamed]")
    with pytest.raises(AiConfigError, match=table):
        load_config(write(tmp_path, text))


def test_missing_key_names_the_key(tmp_path: Path) -> None:
    text: str = replace(DEFAULT_TEXT, "blur_min = 0.0\n", "")
    with pytest.raises(AiConfigError, match="quality.blur_min"):
        load_config(write(tmp_path, text))


def test_missing_and_broken_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(AiConfigError):
        load_config(tmp_path / "none.toml")
    with pytest.raises(AiConfigError):
        load_config(write(tmp_path, "not toml ["))


def test_the_query_encoder_is_pinned_to_a_revision() -> None:
    config = load_config()
    assert re.fullmatch(r"[0-9a-f]{40}", config.models.query_encoder_revision)
    assert config.models.query_encoder_id and config.query.max_tokens > 0


def test_the_verifier_is_pinned_to_a_revision() -> None:
    config = load_config()
    assert re.fullmatch(r"[0-9a-f]{40}", config.models.verifier_revision)
    assert config.models.verifier_id and config.verification.enabled
    assert config.verification.min_matches > 0


def test_the_scene_model_is_pinned_and_the_lists_are_not_empty() -> None:
    config = load_config()
    assert re.fullmatch(r"[0-9a-f]{40}", config.models.scene_revision)
    assert config.models.scene_id and config.scene.enabled
    assert config.scene.rooms and config.scene.surfaces
    assert "{label}" in config.scene.prompt
