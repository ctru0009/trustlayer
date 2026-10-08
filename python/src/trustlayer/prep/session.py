"""Spark session builder: single place that configures local Spark."""

from __future__ import annotations

from pyspark.sql import SparkSession


def build_session(app_name: str = "trustlayer-prep", *, cores: int = 2) -> SparkSession:
    """Build a local Spark session; caller owns stopping it.

    Args:
        app_name: Shown in the Spark UI and logs.
        cores: Local master parallelism (``local[cores]``).

    """
    session = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(cores * 2))
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    return session
