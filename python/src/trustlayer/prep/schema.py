"""Output schema: the Parquet contract downstream phases rely on.

Columns:

    doc_id      stable id ``{split}:{row_sha8}`` (AESLC has no ids upstream)
    thread_id   pseudo-thread from the normalized subject line (AESLC ships no
                thread metadata; ``Re:``/``Fwd:`` prefixes stripped before
                hashing, so replies group together — an approximation,
                documented in docs/learnings/phase2.md)
    split       train | validation | test (upstream split name)
    title       subject line (raw, uncleaned — display only)
    chunk_ord   0-based position of this chunk within its document
    text        cleaned chunk text (empty chunks are dropped before write)
    char_len    ``len(text)`` — lets consumers filter tiny chunks without a scan
"""

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

CORPUS_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), nullable=False),
        StructField("thread_id", StringType(), nullable=False),
        StructField("split", StringType(), nullable=False),
        StructField("title", StringType(), nullable=False),
        StructField("chunk_ord", IntegerType(), nullable=False),
        StructField("text", StringType(), nullable=False),
        StructField("char_len", IntegerType(), nullable=False),
    ]
)
