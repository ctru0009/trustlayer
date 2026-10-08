"""Phase 4 embedding: text chunks → normalized vectors → pgvector.

Pipeline (run with ``uv run --extra embed python -m trustlayer.embed.run``):

    data/processed/corpus/ → encode (MPS fp32) → data/processed/embeddings/
    → COPY into Postgres + pgvector

Layout: ``model`` (loader + prefixes), ``store`` (pure-Python chunk IO,
resume bookkeeping — no torch, importable everywhere), ``load`` (pgvector
COPY, needs the ``embed`` extra), ``run`` (CLI wiring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trustlayer.embed.store import ChunkResult, read_chunk_file, write_chunk_file

if TYPE_CHECKING:
    from trustlayer.embed.run import main

__all__ = ["ChunkResult", "main", "read_chunk_file", "write_chunk_file"]


def __getattr__(name: str) -> Any:
    """Lazily expose ``run.main`` (imports torch) on first access."""
    if name == "main":
        from trustlayer.embed.run import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
