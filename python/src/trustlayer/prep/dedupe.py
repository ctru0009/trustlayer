"""Exact + normalized-hash dedup. Needs the ``spark`` extra."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def dedupe(df: DataFrame, text_col: str = "text") -> tuple[DataFrame, dict[str, int]]:
    """Drop exact duplicates, then whitespace-normalized ones.

    Args:
        df: Input rows.
        text_col: Column holding the cleaned text to dedupe on.

    Returns:
        The deduped rows plus counts (``input``, ``after_exact``,
        ``after_normalized``, ``exact_removed``, ``normalized_removed``).

    """
    total = df.count()
    exact = df.dropDuplicates([text_col])
    after_exact = exact.count()
    normed = exact.withColumn(
        "_norm", F.lower(F.regexp_replace(F.col(text_col), r"\s+", " "))
    )
    out = normed.dropDuplicates(["_norm"]).drop("_norm")
    after_norm = out.count()
    stats = {
        "input": total,
        "after_exact": after_exact,
        "after_normalized": after_norm,
        "exact_removed": total - after_exact,
        "normalized_removed": after_exact - after_norm,
    }
    return out, stats
