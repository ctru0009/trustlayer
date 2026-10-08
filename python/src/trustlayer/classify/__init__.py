"""Phase 7 classifier comparison: data, metrics, calibration, eval.

Pipeline (run with ``make classify-data`` then the per-method targets)::

    labels + corpus + gold → data/classify/{train,eval}.jsonl
    per-method predictions → data/classify/results/<method>.json
    make classify-eval → comparison.json + reliability.json + table

Layout: ``metrics``/``calibrate`` (pure Python — no extras, importable
everywhere), ``data``/``eval`` (runners; ``data`` needs the ``spark``
extra). Method modules (``lr``, ``bert``, ``llm``, ``decide``) are owned
by sibling workers and intentionally not re-exported here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trustlayer.classify.calibrate import (
    ReliabilityBin,
    reliability_bins,
    write_reliability,
)
from trustlayer.classify.metrics import (
    CLASSES,
    ClassScores,
    accuracy,
    bootstrap_ci,
    confusion_matrix_3x3,
    expected_calibration_error,
    macro_f1,
    per_class_scores,
)

if TYPE_CHECKING:
    from trustlayer.classify.data import main

__all__ = [
    "CLASSES",
    "ClassScores",
    "ReliabilityBin",
    "accuracy",
    "bootstrap_ci",
    "confusion_matrix_3x3",
    "expected_calibration_error",
    "macro_f1",
    "main",
    "per_class_scores",
    "reliability_bins",
    "write_reliability",
]


def __getattr__(name: str) -> Any:
    """Lazily expose ``data.main`` (imports pyspark) on first access."""
    if name == "main":
        from trustlayer.classify.data import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
