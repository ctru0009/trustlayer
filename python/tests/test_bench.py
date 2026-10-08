"""Phase 6: query freeze rules + retrieval metrics."""

from trustlayer.bench.freeze import _informative
from trustlayer.bench.metrics import (
    bootstrap_ci,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_informative_rejects_boilerplate() -> None:
    assert not _informative("Re: hello")
    assert not _informative("FW: x")
    assert not _informative("short")
    assert not _informative("   ")
    assert _informative("Gas Prices for Next Quarter Review")


def test_recall_at_k_single_relevant() -> None:
    assert recall_at_k(["a", "b"], {"b"}) == 1.0
    assert recall_at_k(["a", "b"], {"c"}) == 0.0
    assert recall_at_k(["a", "b"], set()) == 0.0


def test_mrr_at_k_rank_weighting() -> None:
    assert mrr_at_k(["a", "b", "c"], {"a"}) == 1.0
    assert mrr_at_k(["a", "b", "c"], {"c"}) == 1 / 3
    assert mrr_at_k(["a", "b"], {"z"}) == 0.0


def test_ndcg_at_k_perfect_and_partial() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}) < 1.0
    assert ndcg_at_k(["x", "y"], {"a"}) == 0.0
    assert ndcg_at_k([], set()) == 0.0


def test_bootstrap_ci_covers_mean() -> None:
    values = [float(i % 2) for i in range(100)]
    lo, hi = bootstrap_ci(values, n_boot=200, seed=6)
    assert lo <= 0.5 <= hi
    assert lo < hi


def test_bootstrap_ci_is_deterministic() -> None:
    values = [0.1 * (i % 7) for i in range(50)]
    assert bootstrap_ci(values) == bootstrap_ci(values)
