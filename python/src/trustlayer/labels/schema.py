"""Label output schemas: labels Parquet + gold JSONL contracts.

labels.parquet columns:

    doc_id      joins back to the Phase 2 corpus
    chunk_ord   joins back to the Phase 2 corpus
    label       public | internal | confidential (weak label)
    pii         array<string> of PII flags (email, phone, ssn, ...)
    rules       array<string> of rule ids that fired (evidence)

gold.jsonl rows (one JSON object per line):

    {"doc_id": ..., "chunk_ord": ..., "text": ..., "title": ...,
     "weak_label": ..., "weak_pii": [...], "gold_label": ..., "note": ...}
    ``gold_label`` is the reviewed label; ``note`` records why it differs
    from the weak label (empty when they agree).
"""

from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

LABELS_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), nullable=False),
        StructField("chunk_ord", IntegerType(), nullable=False),
        StructField("label", StringType(), nullable=False),
        StructField("pii", ArrayType(StringType()), nullable=False),
        StructField("rules", ArrayType(StringType()), nullable=False),
    ]
)

GOLD_FIELDS = (
    "doc_id",
    "chunk_ord",
    "text",
    "title",
    "weak_label",
    "weak_pii",
    "gold_label",
    "note",
)
