"""End-to-end prep pipeline test on synthetic Parquet (needs spark + Java)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

from trustlayer.prep.run import main  # noqa: E402
from trustlayer.prep.schema import CORPUS_SCHEMA  # noqa: E402

# Synthetic rows only — never real email content in fixtures.
ROWS = [
    ("Please review the attached quarterly report draft. Thanks!", "Q3 report draft"),
    ("Please review the attached quarterly report draft. Thanks!", "Q3 report draft"),
    ("please   review the attached QUARTERLY report draft. thanks!", "Fwd: Q3 report"),
    (" ".join(f"Sentence number {i} fills space." for i in range(60)), "Long memo"),
]


def test_pipeline_produces_schema_valid_corpus(tmp_path: Path) -> None:
    from pyspark.sql import SparkSession

    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    spark = SparkSession.builder.master("local[1]").appName("test-prep").getOrCreate()
    try:
        spark.createDataFrame(ROWS, ["email_body", "subject_line"]).write.parquet(
            str(raw / "train-00000-of-00001.parquet")
        )
    finally:
        spark.stop()

    assert main(["--raw", str(raw), "--out", str(out), "--cores", "1"]) == 0

    spark = (
        SparkSession.builder.master("local[1]").appName("test-prep-read").getOrCreate()
    )
    try:
        got = spark.read.parquet(str(out))
        assert [f.name for f in got.schema.fields] == [
            f.name for f in CORPUS_SCHEMA.fields
        ]
        rows = got.orderBy("doc_id", "chunk_ord").collect()
    finally:
        spark.stop()

    by_doc: dict[str, list] = {}
    for row in rows:
        by_doc.setdefault(row.doc_id, []).append(row)
    assert len(by_doc) == 2
    for doc_rows in by_doc.values():
        assert [r.chunk_ord for r in doc_rows] == list(range(len(doc_rows)))
        assert all(r.char_len == len(r.text) for r in doc_rows)
        assert all(r.text.strip() for r in doc_rows)

    stats = json.loads((out.parent / "out-stats.json").read_text())
    assert stats["raw_rows"] == 4
    assert stats["exact_removed"] == 1
    assert stats["normalized_removed"] == 1
    assert stats["chunk_rows"] == len(rows)
    assert len(rows) > 2
