"""Phase 3 pipeline: weak labels, gold sample, frozen split manifests.

Needs the ``spark`` extra. Single command from the repo root::

    make labels

Reads ``data/processed/corpus/``, writes ``data/processed/labels/`` (weak
labels), ``data/processed/labels-stats.json``, and — with ``--gold`` — the
stratified 200-item gold candidate sample to ``data/gold/candidates.jsonl``
for review, plus per-thread split manifests under ``data/gold/splits/``.

Gold review itself is a separate step (see ``review_gold.py``): the sample is
small enough to review outside Spark, and the manifests freeze the reviewed
labels, not the weak ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from trustlayer.labels.rules import weak_label as _weak_label
from trustlayer.labels.schema import LABELS_SCHEMA
from trustlayer.prep.session import build_session

ROOT = Path(__file__).resolve().parents[4]
CORPUS_DEFAULT = ROOT / "data" / "processed" / "corpus"
OUT_DEFAULT = ROOT / "data" / "processed" / "labels"
GOLD_DEFAULT = ROOT / "data" / "gold"
SPLITS = ("train", "dev", "test")

# Fixed seed: the gold sample and split manifests must be reproducible.
SEED = 42
GOLD_N = 200

_WEAK_SCHEMA = StructType(
    [
        StructField("label", StringType(), nullable=False),
        StructField("pii", ArrayType(StringType()), nullable=False),
        StructField("rules", ArrayType(StringType()), nullable=False),
    ]
)


def label_corpus(corpus: DataFrame) -> DataFrame:
    """Weak-label every chunk; returns doc_id/chunk_ord/label/pii/rules."""

    def _apply(title: str, text: str) -> tuple[str, list[str], list[str]]:
        weak = _weak_label(title, text)
        return weak.label.value, list(weak.pii), list(weak.rules)

    udf = F.udf(_apply, _WEAK_SCHEMA, useArrow=False)
    weak = corpus.withColumn("_weak", udf(F.col("title"), F.col("text")))
    return weak.select(
        "doc_id",
        "chunk_ord",
        F.col("_weak.label").alias("label"),
        F.col("_weak.pii").alias("pii"),
        F.col("_weak.rules").alias("rules"),
    )


def sample_gold_candidates(
    corpus: DataFrame, labels: DataFrame, n: int = GOLD_N
) -> list[dict]:
    """Stratified sample of n chunks: third per weak label, rest by label mix."""
    joined = corpus.join(labels, ["doc_id", "chunk_ord"])
    per_label = n // 3
    picks: list[dict] = []
    for label in ("public", "internal", "confidential"):
        rows = (
            joined.filter(F.col("label") == label)
            .orderBy(F.rand(SEED))
            .limit(per_label)
            .select("doc_id", "chunk_ord", "text", "title", "label", "pii")
            .collect()
        )
        picks.extend(
            {
                "doc_id": r.doc_id,
                "chunk_ord": r.chunk_ord,
                "text": r.text,
                "title": r.title,
                "weak_label": r.label,
                "weak_pii": list(r.pii),
            }
            for r in rows
        )
    rest = n - len(picks)
    if rest > 0:
        seen = {(p["doc_id"], p["chunk_ord"]) for p in picks}
        extra = (
            joined.orderBy(F.rand(SEED + 1))
            .select("doc_id", "chunk_ord", "text", "title", "label", "pii")
            .collect()
        )
        for r in extra:
            if (r.doc_id, r.chunk_ord) in seen:
                continue
            picks.append(
                {
                    "doc_id": r.doc_id,
                    "chunk_ord": r.chunk_ord,
                    "text": r.text,
                    "title": r.title,
                    "weak_label": r.label,
                    "weak_pii": list(r.pii),
                }
            )
            if len(picks) >= n:
                break
    if len(picks) < n:
        msg = f"corpus too small for {n} gold candidates ({len(picks)} found)"
        raise ValueError(msg)
    # Deterministic order so re-runs produce identical files.
    picks.sort(key=lambda p: (p["doc_id"], p["chunk_ord"]))
    return picks


def write_split_manifests(
    spark: SparkSession, corpus: DataFrame, gold_dir: Path
) -> dict[str, int]:
    """Split threads 80/10/10 by hash; write doc_id manifests per split."""
    threads = corpus.select("thread_id", "doc_id").distinct()
    hashed = threads.withColumn(
        "_bucket",
        F.abs(F.hash(F.col("thread_id")) % 100),
    )
    counts: dict[str, int] = {}
    splits_dir = gold_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, lo, hi in (("train", 0, 80), ("dev", 80, 90), ("test", 90, 100)):
        ids = (
            hashed.filter((F.col("_bucket") >= lo) & (F.col("_bucket") < hi))
            .select("doc_id")
            .orderBy("doc_id")
            .collect()
        )
        (splits_dir / f"{name}.txt").write_text("".join(f"{r.doc_id}\n" for r in ids))
        counts[name] = len(ids)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Parse args, weak-label the corpus, write labels + stats; exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DEFAULT)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--gold", action="store_true", help="also write gold sample")
    parser.add_argument("--splits", action="store_true", help="also write manifests")
    args = parser.parse_args(argv)
    started = time.monotonic()
    spark = build_session("trustlayer-labels", cores=args.cores)
    try:
        corpus = spark.read.parquet(str(args.corpus))
        labels = label_corpus(corpus)
        labels.write.mode("overwrite").parquet(str(args.out))
        check = spark.read.parquet(str(args.out))
        got = [(f.name, type(f.dataType).__name__) for f in check.schema.fields]
        want = [(f.name, type(f.dataType).__name__) for f in LABELS_SCHEMA.fields]
        if got != want:
            msg = f"output schema drift: got {got}, want {want}"
            raise ValueError(msg)
        dist = {r.label: r["count"] for r in check.groupBy("label").count().collect()}
        full: dict[str, int | float | dict] = {
            "chunks": check.count(),
            "label_dist": dist,
            "seconds": round(time.monotonic() - started, 1),
        }
        if args.gold:
            picks = sample_gold_candidates(corpus, labels)
            args.gold_dir.mkdir(parents=True, exist_ok=True)
            (args.gold_dir / "candidates.jsonl").write_text(
                "".join(json.dumps(p) + "\n" for p in picks)
            )
            full["gold_candidates"] = len(picks)
        if args.splits:
            full["splits"] = write_split_manifests(spark, corpus, args.gold_dir)
        text = json.dumps(full, indent=2) + "\n"
        (args.out.parent / f"{args.out.name}-stats.json").write_text(text)
        print(text, end="")
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
