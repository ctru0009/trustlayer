"""Phase 3 labelling: weak labels, gold set review, frozen splits.

Pipeline (run with ``uv run --extra spark python -m trustlayer.labels.run``):

    data/processed/corpus/ → weak labels → labels.parquet
    200-item gold sample → data/gold/gold.jsonl (agent-reviewed)
    per-thread split manifests → data/gold/splits/{train,dev,test}.txt

Layout: ``rules`` (pure-Python labeller — no pyspark, importable everywhere),
``schema``/``run`` (Spark stages, need the ``spark`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trustlayer.labels.rules import Label, WeakLabel, weak_label

if TYPE_CHECKING:
    from trustlayer.labels.run import main

__all__ = ["Label", "WeakLabel", "main", "weak_label"]


def __getattr__(name: str) -> Any:
    """Lazily expose ``run.main`` (imports pyspark) on first access."""
    if name == "main":
        from trustlayer.labels.run import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
