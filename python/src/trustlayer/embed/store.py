"""Chunk file IO + resume bookkeeping. Pure Python, no torch."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DIM = 768


@dataclass(frozen=True)
class ChunkResult:
    """One embedded chunk: ids + normalized vector."""

    doc_id: str
    chunk_ord: int
    vector: tuple[float, ...]


def chunk_path(out_dir: Path, index: int) -> Path:
    """Path of chunk file ``index`` (zero-padded, sortable)."""
    return out_dir / f"chunk-{index:05d}.jsonl"


def done_indices(out_dir: Path) -> set[int]:
    """Return indices of complete chunk files (resume skips these)."""
    if not out_dir.is_dir():
        return set()
    found = set()
    for path in out_dir.glob("chunk-*.jsonl"):
        try:
            found.add(int(path.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return found


def write_chunk_file(path: Path, rows: list[ChunkResult]) -> None:
    """Write one chunk file, validating every vector first."""
    lines = []
    for row in rows:
        check_vector(row.vector)
        lines.append(
            json.dumps(
                {
                    "doc_id": row.doc_id,
                    "chunk_ord": row.chunk_ord,
                    "vector": list(row.vector),
                }
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" if lines else "")


def read_chunk_file(path: Path) -> list[ChunkResult]:
    """Read one chunk file, validating every vector on the way in."""
    rows = []
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        vector = tuple(float(x) for x in obj["vector"])
        check_vector(vector)
        rows.append(ChunkResult(obj["doc_id"], int(obj["chunk_ord"]), vector))
    return rows


def check_vector(vector: tuple[float, ...]) -> None:
    """Assert finite 768-d vector; raise ValueError otherwise."""
    if len(vector) != DIM:
        msg = f"expected dim {DIM}, got {len(vector)}"
        raise ValueError(msg)
    if not all(math.isfinite(x) for x in vector):
        msg = "non-finite value in embedding vector"
        raise ValueError(msg)
