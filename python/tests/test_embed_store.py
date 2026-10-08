"""Unit tests for trustlayer.embed.store (pure Python, no torch)."""

from pathlib import Path

import pytest

from trustlayer.embed.store import (
    ChunkResult,
    check_vector,
    chunk_path,
    done_indices,
    read_chunk_file,
    write_chunk_file,
)

VEC = tuple([0.1] * 768)


def test_chunk_path_zero_padded(tmp_path: Path) -> None:
    assert chunk_path(tmp_path, 3).name == "chunk-00003.jsonl"


def test_done_indices_skips_junk(tmp_path: Path) -> None:
    (tmp_path / "chunk-00001.jsonl").write_text("")
    (tmp_path / "chunk-abc.jsonl").write_text("")
    (tmp_path / "other.txt").write_text("")
    assert done_indices(tmp_path) == {1}
    assert done_indices(tmp_path / "missing") == set()


def test_write_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "chunk-00000.jsonl"
    rows = [ChunkResult("train:ab12", 0, VEC), ChunkResult("train:ab12", 1, VEC)]
    write_chunk_file(path, rows)
    assert read_chunk_file(path) == rows


def test_write_rejects_bad_dim(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected dim 768"):
        write_chunk_file(tmp_path / "c.jsonl", [ChunkResult("d", 0, (0.1, 0.2))])


def test_write_rejects_nonfinite(tmp_path: Path) -> None:
    bad = tuple([0.1] * 767 + [float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        write_chunk_file(tmp_path / "c.jsonl", [ChunkResult("d", 0, bad)])


def test_read_rejects_bad_dim(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"doc_id": "d", "chunk_ord": 0, "vector": [1.0]}\n')
    with pytest.raises(ValueError, match="expected dim 768"):
        read_chunk_file(path)


def test_check_vector_ok() -> None:
    check_vector(VEC)
