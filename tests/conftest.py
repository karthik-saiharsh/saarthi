"""Safety guard for the whole test run.

The tests use the actual database, the actual storage folder and the actual vector index.
Some of them change every object, for example the calibration after a photo is confirmed.
If the data of a real user is in the database, the tests would damage it. So the run stops before
the first test when the actual data is not empty.

To run the tests on a machine with real data, work in a copy of the project folder that has no `db/data`
and no `storage/uploads`. Set `SAARTHI_TESTS_ALLOW_DATA=1` only if you accept that the data may change.
"""

import os
import sqlite3

import pytest

import db.api as db
import storage.api as storage

# The tables and the folder of the data that a person or the wearable made.
USER_TABLES: tuple[str, ...] = ("enrolled_object", "reference_photo", "image")


def _rows_of_the_actual_db() -> dict[str, int]:
    """Count the rows of the user tables. A missing file or table counts as empty. It only reads."""
    path = db.load_config().sqlite_path
    if not path.is_file():
        return {}
    counts: dict[str, int] = {}
    connection: sqlite3.Connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for table in USER_TABLES:
            try:
                count: int = connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                continue
            if count:
                counts[table] = count
    finally:
        connection.close()
    return counts


def _files_of_the_actual_storage() -> int:
    root = storage.load_config().root
    if not root.is_dir():
        return 0
    return sum(1 for item in root.rglob("*") if item.is_file())


def pytest_sessionstart(session: pytest.Session) -> None:
    if os.environ.get("SAARTHI_TESTS_ALLOW_DATA") == "1":
        return
    rows: dict[str, int] = _rows_of_the_actual_db()
    files: int = _files_of_the_actual_storage()
    if rows or files:
        pytest.exit(
            "The test run stopped: the actual data is not empty "
            f"(rows {rows}, files {files}). The tests would change it. "
            "Run them in a copy of the project folder without db/data and storage/uploads, "
            "or set SAARTHI_TESTS_ALLOW_DATA=1 if you accept the risk.",
            returncode=3,
        )
