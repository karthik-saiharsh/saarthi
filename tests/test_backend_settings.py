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
        ("open_link_iou = 0.5", "open_link_iou = 0.0"),
        ("open_link_iou = 0.5", "open_link_iou = 1.5"),
        ("open_link_iou = 0.5", 'open_link_iou = "half"'),
        ("interval_minutes = 60", "interval_minutes = 0"),
        ("orphan_delete_limit = 100", "orphan_delete_limit = -1"),
        ("retention_batch = 100", "retention_batch = 0"),
        ("retention_batch = 100", "retention_batch = true"),
        ("max_photos_per_object = 20", "max_photos_per_object = 0"),
        ("max_objects = 50", "max_objects = true"),
        ("hysteresis_window_seconds = 60", "hysteresis_window_seconds = 0"),
        ("hysteresis_hits = 2", "hysteresis_hits = 0"),
        ("hysteresis_hits = 2", "hysteresis_hits = true"),
        ("verify_weak = true", 'verify_weak = "yes"'),
        ("verify_max_references = 3", "verify_max_references = 0"),
        ("stale_after_hours = 24", "stale_after_hours = 0"),
        ("window_minutes = 60", "window_minutes = 0"),
        ("max_scan_hours = 24", "max_scan_hours = true"),
        ("review_hours = 24", "review_hours = 0"),
        ("review_min = 0.55", "review_min = 2.0"),
        ("max_regions_per_window = 5000", "max_regions_per_window = 0"),
        ("max_frames = 3000", "max_frames = -1"),
        ("crop_padding = 0.25", "crop_padding = -0.1"),
        ("crop_padding = 0.25", 'crop_padding = "wide"'),
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
        ("neighbour_window_seconds = 120", "neighbour_window_seconds = 0"),
        ("min_seconds = 0.5", "min_seconds = 0.0"),
        ("silence_rms = 0.005", "silence_rms = 2.0"),
        ("neighbour_max_frames = 4", "neighbour_max_frames = 0"),
        ("room_min_frames = 3", "room_min_frames = 0"),
        ("room_agree_fraction = 0.6", "room_agree_fraction = 1.5"),
        ("room_min_score = 0.4", "room_min_score = -0.1"),
        ("surface_min_score = 0.4", 'surface_min_score = "high"'),
        ('time_zone = ""', 'time_zone = "Mars/Olympus"'),
        ('time_zone = ""', "time_zone = 5"),
        ("day_part_min_hours = 6", "day_part_min_hours = 0"),
        ("day_part_min_hours = 6", "day_part_min_hours = 25"),
        ("rewrite_min_words = 3", "rewrite_min_words = 0"),
        ("write = true", 'write = "yes"'),
        ("name_min_confidence = 0.7", "name_min_confidence = 1.5"),
        ("max_words = 30", "max_words = 2"),
        ('safe_words = ["wall"', 'safe_words = ["two words"'),
        ('safe_words = ["wall"', 'safe_words = [5, "wall"'),
        ("rewrite_min_score = 0.9", "rewrite_min_score = 1.5"),
        ('mode = "classic"', 'mode = "fast"'),
        ('mode = "classic"', "mode = 1"),
        ("verify_top_boxes = 20", "verify_top_boxes = 0"),
        ("verify_top_boxes = 20", "verify_top_boxes = true"),
        ("verify_max_frames = 3", "verify_max_frames = 0"),
        ("verify_min_score = 0.35", "verify_min_score = 1.5"),
        ("verify_min_score = 0.35", 'verify_min_score = "high"'),
        ("verify_min_iou = 0.3", "verify_min_iou = -0.1"),
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
        "describe",
        "voice",
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


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("verify_weak = true", "verify_weak = 1"),
        ("verify_weak = true", 'verify_weak = "yes"'),
        ("verify_max_references = 3", "verify_max_references = 0"),
    ],
)
def test_the_verify_keys_of_the_search_are_checked(
    tmp_path: Path, old: str, new: str
) -> None:
    # Change only the table [search]. The table [memory] has the same keys.
    before, _, after = DEFAULT_TEXT.partition("[search]")
    text: str = before + "[search]" + after.replace(old, new, 1)
    with pytest.raises(BackendConfigError, match="search.verify"):
        load_config(write(tmp_path, text))


def test_the_open_search_is_classic_by_default(tmp_path: Path) -> None:
    assert load_config().open_search.mode == "classic"
    text: str = DEFAULT_TEXT.replace('mode = "classic"', 'mode = "verified"', 1)
    assert load_config(write(tmp_path, text)).open_search.mode == "verified"


def test_a_missing_open_search_mode_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigError, match="open_search.mode"):
        load_config(write(tmp_path, DEFAULT_TEXT.replace('mode = "classic"', "", 1)))


def test_tls_is_off_by_default() -> None:
    server = load_config().server
    assert not server.tls_enabled


def test_one_tls_key_alone_is_refused(tmp_path: Path) -> None:
    old: str = 'tls_certfile = ""'
    assert old in DEFAULT_TEXT
    with pytest.raises(BackendConfigError, match="both"):
        load_config(
            write(tmp_path, DEFAULT_TEXT.replace(old, 'tls_certfile = "a.pem"', 1))
        )


def test_tls_paths_must_be_files(tmp_path: Path) -> None:
    cert: Path = tmp_path / "c.pem"
    key: Path = tmp_path / "k.pem"
    text: str = DEFAULT_TEXT.replace(
        'tls_certfile = ""', f'tls_certfile = "{cert}"', 1
    ).replace('tls_keyfile = ""', f'tls_keyfile = "{key}"', 1)
    server = load_config(write(tmp_path, text)).server
    assert server.tls_enabled
    with pytest.raises(BackendConfigError, match="tls_certfile"):
        server.tls_paths()
    cert.write_text("x")
    with pytest.raises(BackendConfigError, match="tls_keyfile"):
        server.tls_paths()
    key.write_text("x")
    assert server.tls_paths() == (cert, key)


def test_the_tls_keys_must_be_strings(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigError, match="tls_keyfile"):
        load_config(
            write(tmp_path, DEFAULT_TEXT.replace('tls_keyfile = ""', "tls_keyfile = 5"))
        )


def test_a_known_time_zone_is_accepted(tmp_path: Path) -> None:
    text = DEFAULT_TEXT.replace('time_zone = ""', 'time_zone = "Asia/Kolkata"', 1)
    assert load_config(write(tmp_path, text)).describe.time_zone == "Asia/Kolkata"


def test_the_variables_replace_the_tls_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    monkeypatch.setenv("SAARTHI_TLS_CERT", str(cert))
    monkeypatch.setenv("SAARTHI_TLS_KEY", str(key))
    server = load_config().server
    assert server.tls_enabled and server.tls_paths() == (cert, key)


def test_one_tls_variable_alone_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAARTHI_TLS_CERT", "/tmp/c.pem")
    monkeypatch.delenv("SAARTHI_TLS_KEY", raising=False)
    with pytest.raises(BackendConfigError, match="both"):
        load_config()


def test_empty_tls_variables_change_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAARTHI_TLS_CERT", "  ")
    monkeypatch.setenv("SAARTHI_TLS_KEY", "")
    assert not load_config().server.tls_enabled
