"""Tests of the FAISS wrapper. They use a temporary file, not the actual index."""

from pathlib import Path

import numpy as np
import pytest

from db.errors import VectorIndexError
from db.vector_index import VectorIndex

DIMENSION: int = 8


def unit_vectors(n: int, seed: int = 0) -> np.ndarray:
    rng: np.random.Generator = np.random.default_rng(seed)
    raw: np.ndarray = rng.normal(size=(n, DIMENSION)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def make_index(tmp_path: Path, checkpoint_every: int = 100) -> VectorIndex:
    return VectorIndex(tmp_path / "test.faiss", DIMENSION, checkpoint_every)


def test_add_and_remove(tmp_path: Path) -> None:
    index: VectorIndex = make_index(tmp_path)
    index.add([1, 2, 3], unit_vectors(3))
    assert index.count == 3 and index.ids() == {1, 2, 3}
    assert index.remove([2, 99]) == 1  # Unknown ids are ignored
    assert index.ids() == {1, 3}
    assert index.remove([]) == 0


def test_add_refuses_known_ids_and_changes_nothing(tmp_path: Path) -> None:
    index: VectorIndex = make_index(tmp_path)
    index.add([1], unit_vectors(1))
    with pytest.raises(VectorIndexError):
        index.add([2, 1], unit_vectors(2, seed=1))
    assert index.ids() == {1}


def test_add_refuses_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(VectorIndexError):
        make_index(tmp_path).add([1, 1], unit_vectors(2))


@pytest.mark.parametrize(
    "bad",
    [
        np.full((1, DIMENSION), np.nan, dtype=np.float32),
        np.full((1, DIMENSION), 2.0, dtype=np.float32),  # Not unit length
    ],
)
def test_add_refuses_bad_values(tmp_path: Path, bad: np.ndarray) -> None:
    with pytest.raises(VectorIndexError):
        make_index(tmp_path).add([1], bad)


def test_add_asserts_shape_and_type(tmp_path: Path) -> None:
    index: VectorIndex = make_index(tmp_path)
    with pytest.raises(AssertionError):
        index.add([1], unit_vectors(2))  # Wrong row count
    with pytest.raises(AssertionError):
        index.add([1], unit_vectors(1).astype(np.float64))
    with pytest.raises(AssertionError):
        index.add([0], unit_vectors(1))  # Ids must be positive


def test_save_and_reload(tmp_path: Path) -> None:
    index: VectorIndex = make_index(tmp_path)
    index.add([5, 6], unit_vectors(2))
    index.save()
    assert index.unsaved == 0
    assert not (tmp_path / "test.faiss.tmp").exists()
    assert make_index(tmp_path).ids() == {5, 6}


def test_checkpoint_runs_only_when_due(tmp_path: Path) -> None:
    index: VectorIndex = make_index(tmp_path, checkpoint_every=3)
    index.add([1, 2], unit_vectors(2))
    assert index.checkpoint_if_due() is False
    assert not (tmp_path / "test.faiss").exists()
    index.add([3], unit_vectors(1, seed=1))
    assert index.checkpoint_if_due() is True
    assert (tmp_path / "test.faiss").exists() and index.unsaved == 0


def test_failed_save_keeps_the_old_file_and_does_not_raise_from_checkpoint(
    tmp_path: Path,
) -> None:
    index: VectorIndex = make_index(tmp_path, checkpoint_every=1)
    index.add([1], unit_vectors(1))
    index.save()
    index.add([2], unit_vectors(1, seed=1))
    (tmp_path / "test.faiss.tmp").mkdir()  # The write to this path must fail
    assert index.checkpoint_if_due() is False
    assert index.unsaved == 1
    (tmp_path / "test.faiss.tmp").rmdir()
    assert make_index(tmp_path).ids() == {1}


def test_wrong_dimension_file_is_refused(tmp_path: Path) -> None:
    index: VectorIndex = make_index(tmp_path)
    index.add([1], unit_vectors(1))
    index.save()
    with pytest.raises(VectorIndexError):
        VectorIndex(tmp_path / "test.faiss", DIMENSION * 2, 10)


def test_broken_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "test.faiss").write_bytes(b"not an index")
    with pytest.raises(VectorIndexError):
        make_index(tmp_path)


def test_get_vectors_returns_the_stored_rows_in_the_order_of_the_ids(
    tmp_path: Path,
) -> None:
    index: VectorIndex = make_index(tmp_path)
    vectors: np.ndarray = unit_vectors(3)
    index.add([5, 6, 7], vectors)
    result: np.ndarray = index.get_vectors([7, 5])
    assert result.shape == (2, DIMENSION) and result.dtype == np.float32
    assert np.allclose(result[0], vectors[2]) and np.allclose(result[1], vectors[0])
    result[0, 0] = 99.0  # A copy: the index does not change
    assert np.allclose(index.get_vectors([7])[0], vectors[2])


def test_get_vectors_of_nothing_is_an_empty_array(tmp_path: Path) -> None:
    assert make_index(tmp_path).get_vectors([]).shape == (0, DIMENSION)


def test_get_vectors_refuses_an_unknown_id(tmp_path: Path) -> None:
    index: VectorIndex = make_index(tmp_path)
    index.add([1], unit_vectors(1))
    with pytest.raises(VectorIndexError, match="not in the index"):
        index.get_vectors([1, 2])
