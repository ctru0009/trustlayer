"""Phase 2 data prep: clean, dedupe, and chunk the corpus into Parquet.

Pipeline (run with ``uv run --extra spark python -m trustlayer.prep.run``):

    raw Parquet → clean → dedupe → chunk → data/processed/corpus/

Layout: ``session`` (Spark builder), ``schema`` (output contract),
``clean``/``chunk`` (pure-Python text logic — no pyspark, importable
everywhere), ``dedupe``/``run`` (Spark stages, need the ``spark`` extra).

``run`` is imported lazily so ``import trustlayer.prep`` works without
pyspark installed (CI and ``make test`` run without the spark extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trustlayer.prep.chunk import chunk_text
from trustlayer.prep.clean import clean_text, thread_id_from_subject

if TYPE_CHECKING:
    from trustlayer.prep.run import main

__all__ = ["chunk_text", "clean_text", "main", "thread_id_from_subject"]


def __getattr__(name: str) -> Any:
    """Lazily expose ``run.main`` (imports pyspark) on first access."""
    if name == "main":
        from trustlayer.prep.run import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
