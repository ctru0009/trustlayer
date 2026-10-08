"""Phase 7: classification metrics on synthetic data (no extras, no files)."""

from pytest import approx, raises

from trustlayer.classify.calibrate import reliability_bins
from trustlayer.classify.metrics import (
    CLASSES,
    accuracy,
    confusion_matrix_3x3,
    expected_calibration_error,
    macro_f1,
    per_class_scores,
)

GOLD = ["public", "public", "internal", "confidential"]
PRED = ["public", "internal", "internal", "confidential"]


def _probs(pred: list[str], conf: float) -> list[dict[str, float]]:
    rest = (1.0 - conf) / 2
    return [{c: (conf if c == p else rest) for c in CLASSES} for p in pred]


def test_per_class_scores_known_values() -> None:
    scores = per_class_scores(GOLD, PRED)
    assert scores["public"]["precision"] == approx(1.0)
    assert scores["public"]["recall"] == approx(0.5)
    assert scores["public"]["f1"] == approx(2 / 3)
    assert scores["internal"]["precision"] == approx(0.5)
    assert scores["internal"]["recall"] == approx(1.0)
    assert scores["internal"]["f1"] == approx(2 / 3)
    assert scores["confidential"]["f1"] == approx(1.0)
    assert scores["confidential"]["support"] == 1


def test_per_class_scores_empty_denominator_is_zero() -> None:
    scores = per_class_scores(["public"], ["internal"])
    missing = scores["confidential"]
    assert (missing["precision"], missing["recall"], missing["f1"]) == (0.0, 0.0, 0.0)
    assert missing["support"] == 0
    assert scores["internal"]["precision"] == approx(0.0)
    assert scores["public"]["recall"] == approx(0.0)


def test_macro_f1_and_accuracy_known_values() -> None:
    assert macro_f1(GOLD, PRED) == approx(7 / 9)
    assert accuracy(GOLD, PRED) == approx(0.75)


def test_confusion_matrix_counts() -> None:
    assert confusion_matrix_3x3(GOLD, PRED) == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_unknown_label_rejected() -> None:
    with raises(ValueError):
        per_class_scores(["public"], ["secret"])
    with raises(ValueError):
        confusion_matrix_3x3(["secret"], ["public"])


def test_mismatched_lengths_rejected() -> None:
    with raises(ValueError):
        per_class_scores(["public"], ["public", "internal"])
    with raises(ValueError):
        reliability_bins(["public"], [])


def test_ece_perfect_calibration_near_zero() -> None:
    gold = ["public", "internal", "confidential", "public"]
    assert expected_calibration_error(gold, _probs(gold, 1.0)) == approx(0.0)
    assert expected_calibration_error([], []) == 0.0


def test_ece_miscalibrated_above_zero() -> None:
    gold = ["public", "internal", "confidential", "public"]
    underconfident = expected_calibration_error(gold, _probs(gold, 0.6))
    assert underconfident == approx(0.4)
    wrong = expected_calibration_error(gold, _probs(["internal"] * 4, 0.9))
    assert wrong > 0.5


def test_reliability_bins_count_and_shape() -> None:
    gold = ["public", "internal", "confidential", "public"]
    bins = reliability_bins(gold, _probs(gold, 1.0))
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == 4
    full = bins[9]
    assert full["count"] == 4
    assert full["accuracy"] == approx(1.0)
    assert full["confidence"] == approx(1.0)
    assert bins[0]["count"] == 0
    assert bins[0]["accuracy"] is None
    assert bins[0]["confidence"] is None
