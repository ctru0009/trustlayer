"""Phase 7 data builder: weak-label train JSONL + gold eval JSONL.

Needs the ``spark`` extra. From the repo root::

    make classify-data

Train rows join weak labels (``data/processed/labels/``) to chunk text
(``data/processed/corpus/``) on ``(doc_id, chunk_ord)``, restricted to
train-manifest docs minus every doc in a gold thread (spec section 5
thread discipline: no train/eval leakage at thread grain). Eval rows are
the 200 gold rows verbatim plus a ``split`` tag.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from trustlayer.prep.session import build_session

ROOT = Path(__file__).resolve().parents[4]
CORPUS_DEFAULT = ROOT / "data" / "processed" / "corpus"
LABELS_DEFAULT = ROOT / "data" / "processed" / "labels"
GOLD_DEFAULT = ROOT / "data" / "gold"
OUT_DEFAULT = ROOT / "data" / "classify"
SPLITS = ("train", "dev", "test")

TRAIN_MIN = 15_000
TRAIN_MAX = 21_000
EVAL_N = 200


def load_manifest(path: Path) -> set[str]:
    """Doc ids in a split manifest (one id per line)."""
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_gold_rows(gold_path: Path) -> list[dict]:
    """Gold rows in file order (each carries doc_id/chunk_ord/gold_label)."""
    return [
        json.loads(line) for line in gold_path.read_text().splitlines() if line.strip()
    ]


def thread_exclusion(
    corpus_rows: list[tuple[str, str, int]],
    gold_docs: set[str],
    train_docs: set[str],
) -> tuple[set[str], dict[str, object]]:
    """Train allow-list minus gold threads; exclusion impact stats.

    Args:
        corpus_rows: ``(doc_id, thread_id, chunk_ord)`` for every corpus chunk.
        gold_docs: doc ids appearing in the gold set.
        train_docs: doc ids in the train manifest.

    Returns:
        Allowed doc ids plus stats (gold thread count, doc- vs
        thread-exclusion doc/chunk counts, max gold-thread size in docs).

    """
    doc_thread: dict[str, str] = {}
    doc_chunks: Counter[str] = Counter()
    for doc_id, thread_id, _ in corpus_rows:
        doc_thread[doc_id] = thread_id
        doc_chunks[doc_id] += 1
    missing = gold_docs - set(doc_thread)
    if missing:
        msg = f"{len(missing)} gold docs missing from corpus: {sorted(missing)[:3]}"
        raise ValueError(msg)
    gold_threads = {doc_thread[d] for d in gold_docs}
    in_gold_thread = {d for d, t in doc_thread.items() if t in gold_threads}
    doc_cut = train_docs & gold_docs
    thread_cut = train_docs & in_gold_thread
    thread_docs: Counter[str] = Counter(doc_thread.values())
    stats: dict[str, object] = {
        "gold_threads": len(gold_threads),
        "doc_exclusion": {
            "docs": len(doc_cut),
            "chunks": sum(doc_chunks[d] for d in doc_cut),
        },
        "thread_exclusion": {
            "docs": len(thread_cut),
            "chunks": sum(doc_chunks[d] for d in thread_cut),
        },
        "max_gold_thread_docs": max(thread_docs[t] for t in gold_threads),
    }
    return train_docs - in_gold_thread, stats


def build_train_rows(
    spark: SparkSession, corpus: DataFrame, labels: DataFrame, allowed: set[str]
) -> list[dict]:
    """Weak-label train rows: join on (doc_id, chunk_ord), allow-listed docs."""
    allow = spark.createDataFrame([(d,) for d in sorted(allowed)], ["doc_id"])
    joined = (
        labels.join(corpus, ["doc_id", "chunk_ord"])
        .join(allow, "doc_id")
        .select(
            "doc_id", "chunk_ord", "title", "text", F.col("label").alias("weak_label")
        )
        .orderBy("doc_id", "chunk_ord")
    )
    return [
        {
            "doc_id": r.doc_id,
            "chunk_ord": r.chunk_ord,
            "title": r.title,
            "text": r.text,
            "weak_label": r.weak_label,
        }
        for r in joined.collect()
    ]


def build_eval_rows(
    gold_rows: list[dict], manifests: dict[str, set[str]]
) -> list[dict]:
    """Gold rows plus the split tag of their doc (train/dev/test)."""
    owner = {d: s for s, ids in manifests.items() for d in ids}
    rows = []
    for row in gold_rows:
        split = owner.get(row["doc_id"])
        if split is None:
            msg = f"gold doc {row['doc_id']} missing from every split manifest"
            raise ValueError(msg)
        rows.append({**row, "split": split})
    return rows


def main(argv: list[str] | None = None) -> int:
    """Build train/eval JSONL, check counts, write stats; exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--labels", type=Path, default=LABELS_DEFAULT)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--cores", type=int, default=2)
    args = parser.parse_args(argv)
    started = time.monotonic()
    manifests = {
        s: load_manifest(args.gold_dir / "splits" / f"{s}.txt") for s in SPLITS
    }
    gold_rows = load_gold_rows(args.gold_dir / "gold.jsonl")
    if len(gold_rows) != EVAL_N:
        msg = f"want {EVAL_N} gold rows, found {len(gold_rows)}"
        raise ValueError(msg)
    gold_docs = {r["doc_id"] for r in gold_rows}
    spark = build_session("trustlayer-classify-data", cores=args.cores)
    try:
        corpus = spark.read.parquet(str(args.corpus))
        labels = spark.read.parquet(str(args.labels))
        mapping = [
            (r.doc_id, r.thread_id, r.chunk_ord)
            for r in corpus.select("doc_id", "thread_id", "chunk_ord").collect()
        ]
        allowed, cut = thread_exclusion(mapping, gold_docs, manifests["train"])
        train_rows = build_train_rows(spark, corpus, labels, allowed)
    finally:
        spark.stop()
    if not TRAIN_MIN <= len(train_rows) <= TRAIN_MAX:
        msg = f"want {TRAIN_MIN}-{TRAIN_MAX} train rows, found {len(train_rows)}"
        raise ValueError(msg)
    leaked = {r["doc_id"] for r in train_rows} - allowed
    if leaked:
        msg = f"{len(leaked)} gold-thread docs leaked into train: {sorted(leaked)[:3]}"
        raise ValueError(msg)
    eval_rows = build_eval_rows(gold_rows, manifests)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_text = "".join(json.dumps(r) + "\n" for r in train_rows)
    eval_text = "".join(json.dumps(r) + "\n" for r in eval_rows)
    (args.out_dir / "train.jsonl").write_text(train_text)
    (args.out_dir / "eval.jsonl").write_text(eval_text)
    stats = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        **cut,
        "weak_dist": dict(Counter(r["weak_label"] for r in train_rows)),
        "seconds": round(time.monotonic() - started, 1),
    }
    text = json.dumps(stats, indent=2) + "\n"
    (args.out_dir / "data-stats.json").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
