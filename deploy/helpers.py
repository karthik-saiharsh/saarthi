"""Helpers for the scripts `./start` and `./clear`. Run them with `uv run python -m deploy.helpers <command>`.

Commands:
- `paths`: print the files and the folder that hold the stored data, one for each line (`file <path>` or `folder <path>`).
- `summary`: print how many objects, photos and frames are stored, and how many files are in the storage folder.
- `port`: print the port of the web server.
- `models-cached`: exit with 0 if every model of `ai/config.toml` is in the Hugging Face cache, else 1. `./start` uses it to run without internet.

The helpers read the settings of the components, so they follow a changed path. They never write.
"""

import os
import sqlite3
import sys
from pathlib import Path

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
# The files of the vector indexes and the lock sit next to the database. The suffixes belong to SQLite.
SQLITE_SUFFIXES: tuple[str, ...] = ("", "-wal", "-shm", "-journal")
LOCK_NAME: str = "saarthi.lock"
LEGACY_OPEN_INDEX: str = "open_regions.faiss"
# A model has at least one file with these endings.
WEIGHT_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin")


def data_paths() -> tuple[list[Path], Path]:
    """Return the files of the data (database, indexes, lock) and the folder of the stored images."""
    from db.api import load_config as load_db_config
    from storage.settings import load_config as load_storage_config

    db_config = load_db_config()
    files: list[Path] = [
        Path(str(db_config.sqlite_path) + suffix) for suffix in SQLITE_SUFFIXES
    ]
    files += [db_config.faiss_path, db_config.crop_faiss_path, db_config.key_faiss_path]
    # The index of an older version (one table for the boxes of the broad detector). ./clear removes it too.
    files.append(db_config.sqlite_path.parent / LEGACY_OPEN_INDEX)
    files.append(db_config.sqlite_path.parent / LOCK_NAME)
    return files, load_storage_config().root


def counts() -> dict[str, int]:
    """Count the stored rows and files. A missing database counts as empty."""
    from db.api import load_config as load_db_config

    _, root = data_paths()
    result: dict[str, int] = {"objects": 0, "photos": 0, "frames": 0, "files": 0}
    sqlite_path: Path = load_db_config().sqlite_path
    if sqlite_path.is_file():
        connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            for key, table in (
                ("objects", "enrolled_object"),
                ("photos", "reference_photo"),
                ("frames", "image"),
            ):
                try:
                    result[key] = connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    pass  # The table does not exist yet
        finally:
            connection.close()
    if root.is_dir():
        result["files"] = sum(1 for item in root.rglob("*") if item.is_file())
    return result


def models_cached(cache_dir: Path | None = None) -> bool:
    """True if every Hugging Face model of `ai/config.toml` has a snapshot in the cache."""
    from ai.settings import load_config

    models = load_config().models
    hub: Path = cache_dir or (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    for model_id, revision in (
        (models.detector_id, models.detector_revision),
        (models.embedder_id, models.embedder_revision),
        (models.query_encoder_id, models.query_encoder_revision),
        (models.verifier_id, models.verifier_revision),
        (models.scene_id, models.scene_revision),
        (models.speech_id, models.speech_revision),
        (models.language_id, models.language_revision),
    ):
        snapshot: Path = (
            hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
        )
        if not snapshot.is_dir():
            return False
        # A file of the snapshot is a link to a blob. A copy of the cache can leave a broken link, so the weights must open.
        if not any(
            item.is_file() and item.suffix in WEIGHT_SUFFIXES
            for item in snapshot.iterdir()
        ):
            return False
    # The broad detector is a file in ai/models, not a Hugging Face model.
    return (PROJECT_DIR / "ai" / "models" / models.open_detector_file).is_file()


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"paths", "summary", "port", "models-cached"}:
        print(__doc__)
        return 2
    command: str = argv[1]
    if command == "paths":
        files, root = data_paths()
        for path in files:
            print(f"file {path}")
        print(f"folder {root}")
        return 0
    if command == "port":
        from backend.settings import load_config

        print(load_config().server.port)
        return 0
    if command == "summary":
        for key, number in counts().items():
            print(f"{key} {number}")
        return 0
    return 0 if models_cached() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
