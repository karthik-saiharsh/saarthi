from pathlib import Path

import pytest

from db.api import ConfigError, init_db, load_config
from db.schema import Image, Region

PATHS: str = "[paths]\nsqlite = 'a.db'\nfaiss_index = 'f'\nopen_faiss_index = 'o'\n"


def write_config(tmp_path: Path, body: str) -> Path:
    path: Path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_default_config_loads() -> None:
    config = load_config()
    assert config.sqlite_path.name == "saarthi.db"
    assert config.faiss_path.name == "regions.faiss"
    assert config.vector_dimension > 0 and config.checkpoint_every > 0
    assert config.open_faiss_path.name == "open_regions.faiss"
    assert config.open_vector_dimension > 0


@pytest.mark.parametrize(
    "body",
    [
        "",
        "[paths]\nsqlite = 'a.db'\n",
        f"{PATHS}[vectors]\ndimension = 8\n",
        f"{PATHS}[vectors]\ndimension = 8\ncheckpoint_every = 5\n",  # No open_dimension
        f"{PATHS}[vectors]\ndimension = 8\ncheckpoint_every = 5\nopen_dimension = 0\n",
        f"{PATHS}[vectors]\ndimension = 0\ncheckpoint_every = 5\n",
        f"{PATHS}[vectors]\ndimension = 8\ncheckpoint_every = true\n",
        f"{PATHS}[vectors]\ndimension = '8'\ncheckpoint_every = 5\n",
        "[paths]\nsqlite = '/abs/a.db'\nfaiss_index = 'f'\n",
        "[paths]\nsqlite = '../a.db'\nfaiss_index = 'f'\n",
        "[paths]\nsqlite = 5\nfaiss_index = 'f'\n",
        "not toml [",
    ],
)
def test_bad_config_raises(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, body))


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "none.toml")


def test_init_db_creates_file_and_tables_and_is_repeatable(tmp_path: Path) -> None:
    body: str = (
        "[paths]\nsqlite = 'data/test_init.db'\nfaiss_index = 'data/t.faiss'\n"
        "open_faiss_index = 'data/t_open.faiss'\n"
        "[vectors]\ndimension = 8\ncheckpoint_every = 5\nopen_dimension = 16\n"
    )
    path: Path = write_config(tmp_path, body)
    config = load_config(path)
    try:
        database, config = init_db(path)
        assert config.sqlite_path.exists()
        assert database.table_exists("image") and database.table_exists("region")
        assert database.execute_sql("PRAGMA foreign_keys").fetchone()[0] == 1
        init_db(path)
        assert Image.select().count() == 0 and Region.select().count() == 0
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(config.sqlite_path) + suffix).unlink(missing_ok=True)
        database.close()


def test_current_config_is_the_loaded_config_and_needs_init_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from db import api

    database, config = init_db()
    try:
        assert api.current_config() == config
        monkeypatch.setattr(api, "_config", None)
        with pytest.raises(api.DbError):
            api.current_config()
    finally:
        database.close()
