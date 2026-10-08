"""Phase 7 decision row: MediaPipe Decision Maker sensitivity labels.

Pipeline (run with ``uv run --extra classify python -m trustlayer.decide.run``)::

    data/classify/eval.jsonl → Laya Choice → .../results/decide-laya.json
    data/classify/eval.jsonl → GLiNER probe → .../gliner-probe.json (NaN check)

Layout: ``run`` (DecisionMaker wrapper, Laya eval, GLiNER probe). Mediapipe is
imported lazily inside ``run`` so this package imports without extras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trustlayer.decide.run import CRITERIA, INSTRUCTIONS, LABELS, normalize_probs

if TYPE_CHECKING:
    from trustlayer.decide.run import main, probe

__all__ = [
    "CRITERIA",
    "INSTRUCTIONS",
    "LABELS",
    "main",
    "normalize_probs",
    "probe",
]


def __getattr__(name: str) -> Any:
    """Lazily expose run.main/probe on first access."""
    if name in {"main", "probe"}:
        from trustlayer.decide import run as _run

        return getattr(_run, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
