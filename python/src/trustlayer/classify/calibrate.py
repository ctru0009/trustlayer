"""Reliability-diagram data: per-bin accuracy/confidence/count JSON.

Pure Python; deliberately no matplotlib dependency — the diagram itself is
plotted from ``data/classify/reliability.json`` with ``uv run --with``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

__all__ = ["ReliabilityBin", "reliability_bins", "write_reliability"]


class ReliabilityBin(TypedDict):
    """One equal-width confidence bin (empty bins carry None scores)."""

    bin: int
    lo: float
    hi: float
    count: int
    accuracy: float | None
    confidence: float | None


def reliability_bins(
    gold: list[str], probs: list[dict[str, float]], n_bins: int = 10
) -> list[ReliabilityBin]:
    """Bin rows by max-prob confidence; per-bin accuracy/confidence/count."""
    if len(gold) != len(probs):
        msg = f"gold/probs length mismatch: {len(gold)} != {len(probs)}"
        raise ValueError(msg)
    if n_bins < 1:
        msg = f"n_bins must be >= 1, got {n_bins}"
        raise ValueError(msg)
    counts = [0] * n_bins
    hits = [0] * n_bins
    conf_sum = [0.0] * n_bins
    for g, prob in zip(gold, probs, strict=True):
        if not prob:
            msg = "empty probs dict"
            raise ValueError(msg)
        pred = max(prob.items(), key=lambda kv: kv[1])[0]
        conf = float(prob[pred])
        if not 0.0 <= conf <= 1.0:
            msg = f"confidence {conf} outside [0, 1]"
            raise ValueError(msg)
        idx = min(int(conf * n_bins), n_bins - 1)
        counts[idx] += 1
        conf_sum[idx] += conf
        if pred == g:
            hits[idx] += 1
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        n = counts[i]
        bins.append(
            {
                "bin": i,
                "lo": i / n_bins,
                "hi": (i + 1) / n_bins,
                "count": n,
                "accuracy": hits[i] / n if n else None,
                "confidence": conf_sum[i] / n if n else None,
            }
        )
    return bins


def write_reliability(path: Path, payload: dict[str, object]) -> None:
    """Write the reliability payload (per-method bins) as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
