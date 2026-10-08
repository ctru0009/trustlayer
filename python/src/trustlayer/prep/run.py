"""Phase 2 pipeline: raw AESLC Parquet → cleaned, deduped, chunked corpus.

Needs the ``spark`` extra. Single command from the repo root::

    make prep

Reads ``data/raw/aeslc/<split>-*.parquet`` (``split`` in train/validation/
test), writes ``data/processed/corpus/`` plus ``corpus-stats.json`` beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from trustlayer.prep.chunk import chunk_text as _chunk_text
from trustlayer.prep.clean import clean_text as _clean_text
from trustlayer.prep.clean import thread_id_from_subject as _thread_id_from_subject
from trustlayer.prep.dedupe import dedupe
from trustlayer.prep.schema import CORPUS_SCHEMA
from trustlayer.prep.session import build_session

ROOT = Path(__file__).resolve().parents[4]
RAW_DEFAULT = ROOT / "data" / "raw" / "aeslc"
OUT_DEFAULT = ROOT / "data" / "processed" / "corpus"
SPLITS = ("train", "validation", "test")


def read_splits(spark: SparkSession, raw: Path) -> DataFrame:
    """Read per-split Parquet files, tagging each row with its split name."""
    frames = []
    for split in SPLITS:
        paths = sorted(raw.glob(f"{split}-*.parquet"))
        if not paths:
            continue
        frame = spark.read.parquet(*[str(p) for p in paths])
        frames.append(frame.withColumn("split", F.lit(split)))
    if not frames:
        msg = f"no <split>-*.parquet found under {raw}"
        raise FileNotFoundError(msg)
    out = frames[0]
    for frame in frames[1:]:
        out = out.union(frame)
    return out


def build_corpus(raw: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Clean, dedupe, and chunk raw rows into the corpus layout."""
    # UDFs are built here, not at module import (@F.udf needs a live session).
    # useArrow=False skips the pandas/Arrow probe (neither installed) and the
    # eval-type UserWarning that comes with it; plain Python UDFs suffice.
    clean_udf = F.udf(_clean_text, StringType(), useArrow=False)
    thread_udf = F.udf(_thread_id_from_subject, StringType(), useArrow=False)
    chunks_udf = F.udf(_chunk_text, ArrayType(StringType()), useArrow=False)
    cleaned = (
        raw.withColumn("title", F.col("subject_line"))
        .withColumn("cleaned", clean_udf(F.col("email_body")))
        .withColumn("thread_id", thread_udf(F.col("subject_line")))
        .withColumn(
            "doc_id",
            F.concat(
                F.col("split"),
                F.lit(":"),
                F.substring(
                    F.sha2(
                        F.concat_ws("\x00", F.col("email_body"), F.col("subject_line")),
                        256,
                    ),
                    1,
                    8,
                ),
            ),
        )
        .filter(F.col("cleaned") != "")
        .drop("email_body", "subject_line")
    )
    input_rows = cleaned.count()
    deduped, dedupe_stats = dedupe(cleaned, "cleaned")
    corpus = (
        deduped.withColumn("_chunks", chunks_udf(F.col("cleaned")))
        .select("*", F.posexplode(F.col("_chunks")).alias("chunk_ord", "text"))
        .drop("_chunks", "cleaned")
        .withColumn("char_len", F.length(F.col("text")))
        .select(
            "doc_id", "thread_id", "split", "title", "chunk_ord", "text", "char_len"
        )
        .orderBy("doc_id", "chunk_ord")
    )
    return corpus, {"cleaned_rows": input_rows, **dedupe_stats}


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, write Parquet + stats; return exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap input rows (smoke runs)"
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    spark = build_session("trustlayer-prep", cores=args.cores)
    try:
        df = read_splits(spark, args.raw)
        if args.limit is not None:
            df = df.limit(args.limit)
        raw_rows = df.count()
        corpus, stats = build_corpus(df)
        corpus.write.mode("overwrite").parquet(str(args.out))
        check = spark.read.parquet(str(args.out))
        got = [(f.name, type(f.dataType).__name__) for f in check.schema.fields]
        want = [(f.name, type(f.dataType).__name__) for f in CORPUS_SCHEMA.fields]
        if got != want:
            msg = f"output schema drift: got {got}, want {want}"
            raise ValueError(msg)
        full: dict[str, int | float] = {
            **stats,
            "raw_rows": raw_rows,
            "chunk_rows": check.count(),
            "seconds": round(time.monotonic() - started, 1),
        }
        text = json.dumps(full, indent=2) + "\n"
        (args.out.parent / f"{args.out.name}-stats.json").write_text(text)
        print(text, end="")
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
