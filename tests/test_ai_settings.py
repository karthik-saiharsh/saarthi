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
        ('mode = "keys"', 'mode = "magic"'),
        ("score_sure = 6.0", "score_sure = 1.0"),
        ("visual_scale = 0.05", "visual_scale = 0.0"),
        ("max_regions = 8", "max_regions = 0"),
        ("sig_weak = 0.72", "sig_weak = 0.80"),
        ("key_crop_size = 224", "key_crop_size = 225"),
        ("word_vote = 0.30", "word_vote = 1.5"),
        ("contrast_phrases = [", "contrast_phrases = []\nold_contrast = ["),
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
        "speech",
        "open_detection",
        "language",
        "matching",
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


def test_the_speech_model_is_pinned_and_the_limits_are_checked(tmp_path: Path) -> None:
    config = load_config()
    assert re.fullmatch(r"[0-9a-f]{40}", config.models.speech_revision)
    assert config.models.speech_id and config.speech.enabled
    for old, new in (
        ("max_seconds = 15", "max_seconds = 0"),
        ("max_seconds = 15", "max_seconds = 31"),
        ("max_new_tokens = 64", "max_new_tokens = 0"),
        ("enabled = true\n# The longest", "enabled = 1\n# The longest"),
    ):
        assert old in DEFAULT_TEXT
        with pytest.raises(AiConfigError):
            load_config(write(tmp_path, DEFAULT_TEXT.replace(old, new, 1)))


def test_the_broad_detector_settings_are_checked(tmp_path: Path) -> None:
    config = load_config()
    assert config.open_detection.enabled and config.open_detection.input_size % 32 == 0
    assert re.fullmatch(r"[0-9a-f]{64}", config.models.open_detector_sha256)
    assert config.models.open_detector_url.startswith("https://")
    for old, new in (
        ("confidence = 0.25", "confidence = 0.0"),
        ("confidence = 0.25", "confidence = 1.5"),
        ("max_boxes = 20", "max_boxes = 0"),
        ("input_size = 640", "input_size = 650"),
        ("min_box_side = 24", "min_box_side = 0"),
        (
            'open_detector_file = "yoloe-11s-seg-pf.pt"',
            'open_detector_file = "../evil.pt"',
        ),
        (
            'open_detector_file = "yoloe-11s-seg-pf.pt"',
            'open_detector_file = "/tmp/x.pt"',
        ),
        (
            'open_detector_file = "yoloe-11s-seg-pf.pt"',
            'open_detector_file = "model.bin"',
        ),
        ('open_detector_url = "https://', 'open_detector_url = "http://'),
        ('open_detector_sha256 = "292bdf', 'open_detector_sha256 = "ZZZZZZ'),
    ):
        assert old in DEFAULT_TEXT, old
        with pytest.raises(AiConfigError):
            load_config(write(tmp_path, DEFAULT_TEXT.replace(old, new, 1)))


def test_the_language_model_settings_are_checked(tmp_path: Path) -> None:
    config = load_config()
    assert re.fullmatch(r"[0-9a-f]{40}", config.models.language_revision)
    assert config.models.language_id and config.language.enabled
    assert len(config.language.rewrite_examples) == 2 and config.language.max_names == 3
    for old, new in (
        ("max_input_chars = 300", "max_input_chars = 5"),
        ("max_new_tokens = 24", "max_new_tokens = 0"),
        ("max_names = 3", "max_names = 0"),
        ("describe_max_new_tokens = 60", "describe_max_new_tokens = 0"),
        ("describe_image_side = 640", "describe_image_side = 10"),
        ('describe_system = "You describe', 'describe_system = ""  # You describe'),
        ("max_names = 3", "max_names = 11"),
        ("enabled = true\n# The longest text", "enabled = 1\n# The longest text"),
        ('rewrite_system = "You help', 'rewrite_system = ""  # You help'),
        (
            '  ["what I listen to music on", "phone, speaker, radio"],\n',
            '  ["only a question"],\n',
        ),
        (
            '  ["what I listen to music on", "phone, speaker, radio"],\n',
            '  "not a pair",\n',
        ),
    ):
        assert old in DEFAULT_TEXT, old
        with pytest.raises(AiConfigError):
            load_config(write(tmp_path, DEFAULT_TEXT.replace(old, new, 1)))


def test_no_example_is_allowed(tmp_path: Path) -> None:
    text = DEFAULT_TEXT.replace(
        '  ["what I listen to music on", "phone, speaker, radio"],\n', ""
    ).replace('  ["what I cook rice in", "pot, cooker, pan"],\n', "")
    assert load_config(write(tmp_path, text)).language.rewrite_examples == ()
