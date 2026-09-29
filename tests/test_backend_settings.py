from pathlib import Path

import pytest

from backend.errors import BackendConfigError
from backend.settings import DEFAULT_CONFIG_PATH, BackendConfig, load_config

DEFAULT_TEXT: str = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def write(tmp_path: Path, text: str) -> Path:
    path: Path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_loads() -> None:
    config: BackendConfig = load_config()
    assert config.max_pending >= 1 and config.token_env_var == "SAARTHI_API_TOKEN"
    assert (
        config.worker.idle_wait_seconds >= 1 and config.worker.generic_pass_every >= 0
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("idle_wait_seconds = 5", "idle_wait_seconds = 0"),
        ("generic_pass_every = 10", "generic_pass_every = -1"),
        ("generic_pass_every = 10", "generic_pass_every = true"),
        ("duplicate_window_minutes = 60", "duplicate_window_minutes = 0"),
        ("duplicate_max_compared = 200", "duplicate_max_compared = 0"),
        ("error_pause_seconds = 5", 'error_pause_seconds = "soon"'),
        ("interval_minutes = 60", "interval_minutes = 0"),
        ("orphan_delete_limit = 100", "orphan_delete_limit = -1"),
        ("retention_batch = 100", "retention_batch = 0"),
        ("retention_batch = 100", "retention_batch = true"),
        ("max_photos_per_object = 20", "max_photos_per_object = 0"),
        ("max_objects = 50", "max_objects = true"),
        ("hysteresis_window_seconds = 60", "hysteresis_window_seconds = 0"),
        ("hysteresis_hits = 2", "hysteresis_hits = 0"),
        ("hysteresis_hits = 2", "hysteresis_hits = true"),
        ("stale_after_hours = 24", "stale_after_hours = 0"),
        ("window_minutes = 60", "window_minutes = 0"),
        ("max_scan_hours = 72", "max_scan_hours = true"),
        ("max_regions_per_window = 5000", "max_regions_per_window = 0"),
        ("max_frames = 3000", "max_frames = -1"),
        ("matched_min_score = 0.35", "matched_min_score = 2.0"),
        ("matched_min_score = 0.35", 'matched_min_score = "high"'),
        ("matched_min_gap = 0.10", "matched_min_gap = -0.1"),
        (
            "suggest_min_score = 0.20",
            "suggest_min_score = 0.5",
        ),  # Above matched_min_score
        ("max_candidates = 3", "max_candidates = 0"),
        ("max_chars = 300", "max_chars = 0"),
        ("port = 8080", "port = 0"),
        ("port = 8080", "port = 70000"),
        ("port = 8080", 'port = "8080"'),
        ("worker_stop_seconds = 120", "worker_stop_seconds = 0"),
        ('host = "0.0.0.0"', 'host = ""'),
        ('host = "0.0.0.0"', "host = 5"),
        ("max_pending = 500", "max_pending = 0"),
        ("max_pending = 500", "max_pending = true"),
        ("max_pending = 500", 'max_pending = "many"'),
        ('token_env_var = "SAARTHI_API_TOKEN"', 'token_env_var = ""'),
        ('token_env_var = "SAARTHI_API_TOKEN"', 'token_env_var = "NOT A NAME"'),
        ('token_env_var = "SAARTHI_API_TOKEN"', "token_env_var = 5"),
    ],
)
def test_bad_values_are_refused(tmp_path: Path, old: str, new: str) -> None:
    assert old in DEFAULT_TEXT
    with pytest.raises(BackendConfigError):
        load_config(write(tmp_path, DEFAULT_TEXT.replace(old, new, 1)))


@pytest.mark.parametrize(
    "table",
    [
        "ingest",
        "api",
        "worker",
        "server",
        "maintenance",
        "enrollment",
        "memory",
        "search",
    ],
)
def test_missing_table_is_refused(tmp_path: Path, table: str) -> None:
    with pytest.raises(BackendConfigError, match=table):
        load_config(write(tmp_path, DEFAULT_TEXT.replace(f"[{table}]", f"[{table}_x]")))


def test_missing_key_names_the_key(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigError, match="ingest.max_pending"):
        load_config(write(tmp_path, DEFAULT_TEXT.replace("max_pending = 500", "")))


def test_a_missing_worker_key_names_the_key(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigError, match="worker.idle_wait_seconds"):
        load_config(write(tmp_path, DEFAULT_TEXT.replace("idle_wait_seconds = 5", "")))


def test_generic_pass_can_be_switched_off(tmp_path: Path) -> None:
    text: str = DEFAULT_TEXT.replace(
        "generic_pass_every = 10", "generic_pass_every = 0"
    )
    assert load_config(write(tmp_path, text)).worker.generic_pass_every == 0


def test_missing_and_broken_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigError):
        load_config(tmp_path / "none.toml")
    with pytest.raises(BackendConfigError):
        load_config(write(tmp_path, "not toml ["))
