"""Shared file-digest and fixed-size-list helpers (ADR-0002 sidecar discipline).

Expected digests come from ``sha256sum`` on the literal bytes, not from the helpers.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from avr.io import (
    read_fixed_list_matrix,
    read_sidecar_sha256,
    repo_relative,
    sha256_of_file,
    write_sha256_sidecar,
)

# printf 'abc' | sha256sum
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_of_file_matches_sha256sum(tmp_path: Path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_of_file(path) == SHA256_ABC


def test_sha256_of_file_is_chunk_independent(tmp_path: Path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_of_file(path, chunk_size=1) == SHA256_ABC


def test_write_sha256_sidecar_uses_sha256sum_format(tmp_path: Path):
    path = tmp_path / "table.parquet"
    path.write_bytes(b"abc")

    sidecar = write_sha256_sidecar(path)

    assert sidecar == tmp_path / "table.sha256"
    assert sidecar.read_text() == f"{SHA256_ABC}  table.parquet\n"
    assert read_sidecar_sha256(sidecar) == SHA256_ABC


def test_read_fixed_list_matrix_returns_rows_by_dim_float32():
    table = pa.table(
        {
            "image_id": ["a", "b"],
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], pa.float32()), 3
            ),
        }
    )
    matrix = read_fixed_list_matrix(table, "embedding")
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix, [[1, 2, 3], [4, 5, 6]])


def test_read_fixed_list_matrix_handles_chunked_columns():
    one = pa.table({"e": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 2.0], pa.float32()), 2)})
    two = pa.table({"e": pa.FixedSizeListArray.from_arrays(pa.array([3.0, 4.0], pa.float32()), 2)})
    matrix = read_fixed_list_matrix(pa.concat_tables([one, two]), "e")
    np.testing.assert_array_equal(matrix, [[1, 2], [3, 4]])


def test_read_fixed_list_matrix_rejects_a_non_fixed_list_column():
    with pytest.raises(TypeError, match="fixed-size list"):
        read_fixed_list_matrix(pa.table({"e": [[1.0], [2.0]]}), "e")


def test_repo_relative_inside_and_outside_the_root(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    assert repo_relative(root / "data" / "x.parquet", root) == "data/x.parquet"
    assert repo_relative(tmp_path / "other" / "y", root) == "../other/y"
    assert "/" not in repo_relative(root / "z", root)
