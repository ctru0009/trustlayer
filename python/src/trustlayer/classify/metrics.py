"""Classification metrics: per-class scores, confusion, ECE.

Pure Python (no numpy or scikit-learn): unit-tested without extras. Class
order is fixed as ``CLASSES`` everywhere (confusion rows/columns, the
probability-dict keys in prediction files).
"""

from __future__ import annotations

from typing import TypedDict

from trustlayer.bench.metrics import bootstrap_ci
from trustlayer.classify.calibrate import reliability_bins

__all__ = [
    "CLASSES",
    "ClassScores",
    "accuracy",
    "bootstrap_ci",
    "confusion_matrix_3x3",
    "expected_calibration_error",
    "macro_f1",
    "per_class_scores",
]

CLASSES: tuple[str, str, str] = ("public", "internal", "confidential")


class ClassScores(TypedDict):
    """Per-class precision/recall/F1 plus the gold support count."""

    precision: float
    recall: float
    f1: float
    support: int


def _check_labels(gold: list[str], pred: list[str]) -> None:
    """Reject unequal-length lists and labels outside CLASSES."""
    if len(gold) != len(pred):
        msg = f"gold/pred length mismatch: {len(gold)} != {len(pred)}"
        raise ValueError(msg)
    for label in gold + pred:
        if label not in CLASSES:
            msg = f"unknown label {label!r}, want one of {CLASSES}"
            raise ValueError(msg)


def per_class_scores(gold: list[str], pred: list[str]) -> dict[str, ClassScores]:
    """Precision/recall/F1 per class (0.0 when the denominator is empty)."""
    _check_labels(gold, pred)
    pairs = list(zip(gold, pred, strict=True))
    out: dict[str, ClassScores] = {}
    for cls in CLASSES:
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        fp = sum(1 for g, p in pairs if g != cls and p == cls)
        fn = sum(1 for g, p in pairs if g == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0
        out[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return out


def macro_f1(gold: list[str], pred: list[str]) -> float:
    """Mean per-class F1 — the rare class counts the same as the common."""
    scores = per_class_scores(gold, pred)
    return sum(s["f1"] for s in scores.values()) / len(CLASSES)


def accuracy(gold: list[str], pred: list[str]) -> float:
    """Fraction of exact matches (0.0 for empty)."""
    _check_labels(gold, pred)
    if not gold:
        return 0.0
    hits = sum(1 for g, p in zip(gold, pred, strict=True) if g == p)
    return hits / len(gold)


def confusion_matrix_3x3(gold: list[str], pred: list[str]) -> list[list[int]]:
    """3x3 confusion counts: rows gold, columns pred, both in CLASSES order."""
    _check_labels(gold, pred)
    idx = {c: i for i, c in enumerate(CLASSES)}
    matrix = [[0, 0, 0] for _ in CLASSES]
    for g, p in zip(gold, pred, strict=True):
        matrix[idx[g]][idx[p]] += 1
    return matrix


def expected_calibration_error(
    gold: list[str], probs: list[dict[str, float]], n_bins: int = 10
) -> float:
    """Multiclass ECE over equal-width bins on the max-prob confidence."""
    bins = reliability_bins(gold, probs, n_bins=n_bins)
    n = len(gold)
    if n == 0:
        return 0.0
    total = 0.0
    for b in bins:
        if b["count"] and b["accuracy"] is not None and b["confidence"] is not None:
            total += b["count"] / n * abs(b["accuracy"] - b["confidence"])
    return total
