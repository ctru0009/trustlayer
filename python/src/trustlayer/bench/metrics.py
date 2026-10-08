"""Retrieval metrics: recall@k, MRR@k, nDCG@k + bootstrap CIs.

Pure Python (no numpy): unit-tested without extras. Single-relevant
judgments (one relevant doc per query) are the Phase 6 design — with
one relevant, recall@k equals hit@k.
"""

from __future__ import annotations

import math
import random


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """Fraction of relevant docs appearing in the top-k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant doc in the top-k (0 if absent)."""
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """Rank-aware gain with binary relevance, normalized by the ideal top-k."""
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def mean(values: list[float]) -> float:
    """Arithmetic mean (0.0 for empty)."""
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: list[float], n_boot: int = 1000, seed: int = 6, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean (spec section 10, protocol 4)."""
    rng = random.Random(seed)  # noqa: S311 (bootstrap resampling, not crypto)
    n = len(values)
    means = sorted(mean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo_q = (1.0 - level) / 2.0
    lo = means[int(lo_q * n_boot)]
    hi = means[min(int((1.0 - lo_q) * n_boot), n_boot - 1)]
    return (lo, hi)
