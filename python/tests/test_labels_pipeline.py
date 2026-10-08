"""End-to-end labels pipeline test on synthetic Parquet (needs spark)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

from trustlayer.labels.run import main  # noqa: E402
from trustlayer.labels.schema import GOLD_FIELDS, LABELS_SCHEMA  # noqa: E402

ROWS = [
    ("t1", "doc1", "Hello", 0, "Want to grab sandwiches at noon?", 32),
    ("t1", "doc1", "Hello", 1, "The cafe on 5th is good.", 24),
    ("t2", "doc2", "Payroll", 0, "SSN on file: 123-45-6789.", 26),
    ("t3", "doc3", "Q3 forecast", 0, "Budget review Thursday.", 23),
]


def test_labelling_produces_schema_valid_output(tmp_path: Path) -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    corpus, out, gold = tmp_path / "corpus", tmp_path / "labels", tmp_path / "gold"
    spark = SparkSession.builder.master("local[1]").appName("test-labels").getOrCreate()
    try:
        # createDataFrame infers LongType for Python ints; cast to match the
        # real corpus schema (IntegerType) so the drift check is meaningful.
        frame = spark.createDataFrame(
            ROWS, ["thread_id", "doc_id", "title", "chunk_ord", "text", "char_len"]
        )
        frame = frame.withColumn("chunk_ord", F.col("chunk_ord").cast("int"))
        frame = frame.withColumn("char_len", F.col("char_len").cast("int"))
        frame.write.parquet(str(corpus))
    finally:
        spark.stop()

    args = [
        "--corpus",
        str(corpus),
        "--out",
        str(out),
        "--gold-dir",
        str(gold),
        "--cores",
        "1",
        "--splits",
    ]
    assert main(args) == 0

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-labels-read")
        .getOrCreate()
    )
    try:
        got = spark.read.parquet(str(out))
        assert [f.name for f in got.schema.fields] == [
            f.name for f in LABELS_SCHEMA.fields
        ]
        rows = {r.doc_id: r for r in got.collect()}
    finally:
        spark.stop()

    assert rows["doc1"].label == "public"
    assert rows["doc2"].label == "confidential"
    assert "ssn" in rows["doc2"].pii
    assert rows["doc3"].label == "internal"

    stats = json.loads((out.parent / "labels-stats.json").read_text())
    assert stats["chunks"] == 4
    assert stats["label_dist"] == {"public": 2, "confidential": 1, "internal": 1}
    assert sum(stats["splits"].values()) == 3

    names = sorted(p.name for p in (gold / "splits").iterdir())
    assert names == ["dev.txt", "test.txt", "train.txt"]

    assert set(GOLD_FIELDS) == {
        "doc_id",
        "chunk_ord",
        "text",
        "title",
        "weak_label",
        "weak_pii",
        "gold_label",
        "note",
    }
