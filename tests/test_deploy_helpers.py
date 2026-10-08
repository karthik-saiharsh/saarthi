"""Tests of the helpers of the scripts `./start` and `./clear`. They write nothing in the actual data."""

from pathlib import Path

import pytest

from ai.settings import load_config
from deploy import helpers


def test_the_paths_are_the_files_and_the_folder_of_the_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    files, root = helpers.data_paths()
    names = {path.name for path in files}
    assert {
        "saarthi.db",
        "saarthi.db-wal",
        "saarthi.db-shm",
        "regions.faiss",
        "open_regions.faiss",
        "saarthi.lock",
    } <= names
    assert root.name == "uploads"
    assert helpers.main(["helpers", "paths"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("file ") for line in lines) == len(files)
    assert lines[-1] == f"folder {root}"


def test_the_summary_counts_are_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    result = helpers.counts()
    assert set(result) == {"objects", "photos", "frames", "files"} and all(
        v >= 0 for v in result.values()
    )
    assert helpers.main(["helpers", "summary"]) == 0
    assert "objects" in capsys.readouterr().out


def test_the_port_is_the_port_of_the_server(capsys: pytest.CaptureFixture[str]) -> None:
    assert helpers.main(["helpers", "port"]) == 0
    assert capsys.readouterr().out.strip() == "8080"


@pytest.mark.parametrize(
    "arguments", [["helpers"], ["helpers", "nope"], ["helpers", "paths", "extra"]]
)
def test_a_wrong_command_gives_the_help_and_code_2(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert helpers.main(arguments) == 2
    assert "Commands:" in capsys.readouterr().out


def make_cache(tmp_path: Path, missing: str | None = None) -> Path:
    models = load_config().models
    hub = tmp_path / "hub"
    for model_id, revision in (
        (models.detector_id, models.detector_revision),
        (models.embedder_id, models.embedder_revision),
        (models.query_encoder_id, models.query_encoder_revision),
        (models.verifier_id, models.verifier_revision),
        (models.scene_id, models.scene_revision),
        (models.speech_id, models.speech_revision),
        (models.language_id, models.language_revision),
    ):
        if model_id != missing:
            (
                hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
            ).mkdir(parents=True)
    return hub


def test_models_are_cached_when_every_snapshot_and_the_detector_file_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    (project / "ai" / "models").mkdir(parents=True)
    (project / "ai" / "models" / load_config().models.open_detector_file).write_bytes(
        b"x"
    )
    monkeypatch.setattr(helpers, "PROJECT_DIR", project)
    assert helpers.models_cached(make_cache(tmp_path)) is True


def test_one_missing_model_means_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    (project / "ai" / "models").mkdir(parents=True)
    (project / "ai" / "models" / load_config().models.open_detector_file).write_bytes(
        b"x"
    )
    monkeypatch.setattr(helpers, "PROJECT_DIR", project)
    assert (
        helpers.models_cached(
            make_cache(tmp_path, missing=load_config().models.language_id)
        )
        is False
    )


def test_a_missing_detector_file_means_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpers, "PROJECT_DIR", tmp_path / "empty-project")
    assert helpers.models_cached(make_cache(tmp_path)) is False
