"""Phase 7 comparison: score every method's predictions on the gold set.

Reads ``data/classify/results/*.json``, writes
``data/classify/comparison.json`` (plus ``reliability.json`` for the
reliability diagram), and prints markdown comparison tables.

Primary scores cover all rows; a secondary section re-scores the
``split == "test"`` subset (n ~ 20) with bootstrap CIs, joining the split
tag from ``data/classify/eval.jsonl``. Tolerates missing methods (scores
whatever exists); ``gliner-f16`` is a NaN probe, not a comparison row,
and is skipped.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

from trustlayer.classify.calibrate import reliability_bins, write_reliability
from trustlayer.classify.metrics import (
    CLASSES,
    accuracy,
    bootstrap_ci,
    confusion_matrix_3x3,
    expected_calibration_error,
    macro_f1,
    per_class_scores,
)

ROOT = Path(__file__).resolve().parents[4]
RESULTS_DEFAULT = ROOT / "data" / "classify" / "results"
EVAL_DEFAULT = ROOT / "data" / "classify" / "eval.jsonl"
COMPARISON_DEFAULT = ROOT / "data" / "classify" / "comparison.json"
RELIABILITY_DEFAULT = ROOT / "data" / "classify" / "reliability.json"

PROBE_METHODS = ("gliner-f16",)
SEED = 42
N_BOOT = 1000
NOTES = (
    "P/R/F1 are 0.0 when their denominator is empty (no predicted/gold"
    " instances); the paired CI is null.",
    "Primary scores cover every predicted row; test_subset re-scores the"
    ' split == "test" rows only (n ~ 20, wide CIs expected).',
)

_ROW_KEYS = ("doc_id", "chunk_ord", "gold", "pred", "probs", "latency_ms")


def load_results(results_dir: Path) -> dict[str, dict]:
    """Parse every results/*.json; skip probes and broken files with a note."""
    payloads: dict[str, dict] = {}
    if not results_dir.is_dir():
        print(f"note: {results_dir} missing, nothing to score", file=sys.stderr)
        return payloads
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"note: skipping {path.name} ({exc})", file=sys.stderr)
            continue
        config = payload.get("config", {})
        if isinstance(config, dict):
            method = str(config.get("method", path.stem))
        else:
            method = path.stem
        if method in PROBE_METHODS:
            print(f"note: skipping {method} (NaN probe, not a row)", file=sys.stderr)
            continue
        payloads[method] = payload
    return payloads


def load_split_tags(eval_path: Path) -> dict[tuple[str, int], str] | None:
    """Map (doc_id, chunk_ord) to split; None (with a note) if eval is missing."""
    if not eval_path.is_file():
        print(f"note: {eval_path} missing, skipping test subset", file=sys.stderr)
        return None
    tags = {}
    for line in eval_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tags[(str(row["doc_id"]), int(row["chunk_ord"]))] = str(row["split"])
    return tags


def _percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]); 0.0 for empty."""
    if not ordered:
        return 0.0
    rank = min(math.ceil(q * len(ordered)), len(ordered))
    return ordered[rank - 1]


def _checked_rows(method: str, payload: dict) -> list[dict]:
    """Rows with validated keys, CLASSES-keyed probs summing to 1."""
    rows = payload.get("rows", [])
    if not rows:
        msg = f"{method}: no rows to score"
        raise ValueError(msg)
    for i, row in enumerate(rows):
        for key in _ROW_KEYS:
            if key not in row:
                msg = f"{method} row {i}: missing {key!r}"
                raise ValueError(msg)
        try:
            prob = {c: float(row["probs"][c]) for c in CLASSES}
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"{method} row {i}: probs must map {CLASSES} to floats"
            raise ValueError(msg) from exc
        if abs(sum(prob.values()) - 1.0) > 1e-3:
            msg = f"{method} row {i}: probs sum to {sum(prob.values())}, want 1.0"
            raise ValueError(msg)
    return rows


def summarize_method(method: str, payload: dict) -> dict:
    """Score one method's rows; returns a JSON-serializable summary dict."""
    rows = _checked_rows(method, payload)
    gold = [str(r["gold"]) for r in rows]
    pred = [str(r["pred"]) for r in rows]
    probs = [{c: float(r["probs"][c]) for c in CLASSES} for r in rows]
    lat = sorted(float(r["latency_ms"]) for r in rows)
    scores = per_class_scores(gold, pred)
    pairs = list(zip(gold, pred, strict=True))
    correct = [1.0 if g == p else 0.0 for g, p in pairs]
    acc_lo, acc_hi = bootstrap_ci(correct, n_boot=N_BOOT, seed=SEED)
    ci95: dict[str, dict[str, list[float] | None]] = {}
    for cls in CLASSES:
        prec = [1.0 if g == cls else 0.0 for g, p in pairs if p == cls]
        rec = [1.0 if p == cls else 0.0 for g, p in pairs if g == cls]
        prec_ci = bootstrap_ci(prec, n_boot=N_BOOT, seed=SEED) if prec else None
        rec_ci = bootstrap_ci(rec, n_boot=N_BOOT, seed=SEED) if rec else None
        ci95[cls] = {
            "precision": list(prec_ci) if prec_ci else None,
            "recall": list(rec_ci) if rec_ci else None,
        }
    config = payload.get("config", {})
    cost = (
        float(config.get("cost_per_1k_usd", 0.0)) if isinstance(config, dict) else 0.0
    )
    return {
        "n": len(rows),
        "accuracy": accuracy(gold, pred),
        "accuracy_ci95": [acc_lo, acc_hi],
        "macro_f1": macro_f1(gold, pred),
        "per_class": {c: dict(scores[c]) for c in CLASSES},
        "per_class_ci95": ci95,
        "confusion_3x3": confusion_matrix_3x3(gold, pred),
        "ece_10": expected_calibration_error(gold, probs),
        "latency_ms": {
            "p50": _percentile(lat, 0.5),
            "p95": _percentile(lat, 0.95),
            "mean": sum(lat) / len(lat),
            "total_s": round(sum(lat) / 1000, 1),
        },
        "cost_per_1k_usd": cost,
    }


def comparison_table(summaries: dict[str, dict]) -> str:
    """Markdown table: one row per method, scores + ECE + latency + cost."""
    header = (
        "| method | n | acc [95% CI] | macro-F1 | P-pub | R-pub | P-int | R-int |"
        " P-con | R-con | ECE | p50 ms | p95 ms | $/1k |"
    )
    lines = [header, "|" + "---|" * 14]
    for method in sorted(summaries):
        s = summaries[method]
        lo, hi = s["accuracy_ci95"]
        pc = s["per_class"]
        lat = s["latency_ms"]
        lines.append(
            f"| {method} | {s['n']} | {s['accuracy']:.3f} [{lo:.3f}, {hi:.3f}] |"
            f" {s['macro_f1']:.3f} |"
            f" {pc['public']['precision']:.3f} | {pc['public']['recall']:.3f} |"
            f" {pc['internal']['precision']:.3f} | {pc['internal']['recall']:.3f} |"
            f" {pc['confidential']['precision']:.3f} |"
            f" {pc['confidential']['recall']:.3f} |"
            f" {s['ece_10']:.3f} | {lat['p50']:.1f} | {lat['p95']:.1f} |"
            f" ${s['cost_per_1k_usd']:.4f} |"
        )
    return "\n".join(lines)


def test_subset_table(summaries: dict[str, dict]) -> str:
    """Compact markdown table for the split == test secondary scores."""
    lines = [
        "| method | n | acc [95% CI] | macro-F1 | ECE |",
        "|" + "---|" * 5,
    ]
    for method in sorted(summaries):
        s = summaries[method].get("test_subset")
        if not s:
            continue
        lo, hi = s["accuracy_ci95"]
        lines.append(
            f"| {method} | {s['n']} | {s['accuracy']:.3f} [{lo:.3f}, {hi:.3f}] |"
            f" {s['macro_f1']:.3f} | {s['ece_10']:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Score all results, write comparison + reliability JSON, print tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--comparison", type=Path, default=COMPARISON_DEFAULT)
    parser.add_argument("--reliability", type=Path, default=RELIABILITY_DEFAULT)
    args = parser.parse_args(argv)
    payloads = load_results(args.results)
    tags = load_split_tags(args.eval)
    summaries: dict[str, dict] = {}
    for method, payload in payloads.items():
        summary = summarize_method(method, payload)
        if tags is not None:
            test_rows = [
                r
                for r in payload["rows"]
                if tags.get((str(r["doc_id"]), int(r["chunk_ord"]))) == "test"
            ]
            if test_rows:
                sub = {"config": payload.get("config", {}), "rows": test_rows}
                summary["test_subset"] = summarize_method(f"{method}/test", sub)
            else:
                summary["test_subset"] = None
        else:
            summary["test_subset"] = None
        summaries[method] = summary
    generated = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison = {"generated": generated, "notes": list(NOTES), "methods": summaries}
    args.comparison.write_text(json.dumps(comparison, indent=2) + "\n")
    rel = {}
    for method, payload in payloads.items():
        gold = [str(r["gold"]) for r in payload["rows"]]
        probs = [{c: float(r["probs"][c]) for c in CLASSES} for r in payload["rows"]]
        rel[method] = reliability_bins(gold, probs)
    write_reliability(args.reliability, {"generated": generated, "methods": rel})
    if summaries:
        print(comparison_table(summaries))
        if any(s.get("test_subset") for s in summaries.values()):
            print("\ntest subset (split == test):\n")
            print(test_subset_table(summaries))
        print("\nnotes:")
        for note in NOTES:
            print(f"- {note}")
    else:
        print("no method predictions yet; wrote empty comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
