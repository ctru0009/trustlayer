"""pgvector load: chunk files → documents/chunks tables via COPY.

Needs the ``embed`` extra (psycopg). Schema matches spec section 8.5.
All loads are re-runnable: metadata upserts, vectors upsert via a temp
staging table (COPY has no ON CONFLICT).
"""

from __future__ import annotations

import json
from pathlib import Path

MIGRATION = (
    Path(__file__).resolve().parents[4]
    / "infra"
    / "db"
    / "migrations"
    / "001_pgvector_schema.sql"
)


def schema_sql() -> str:
    """Read the versioned schema (migration 001 is the source of truth)."""
    return MIGRATION.read_text()


def ensure_schema(conn: object) -> None:
    """Create extension + tables if missing (idempotent)."""
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(schema_sql())
    conn.commit()  # type: ignore[attr-defined]


def copy_chunk_file(conn: object, path: Path) -> int:
    """COPY one chunk file's vectors into chunks; returns rows loaded.

    Requires metadata rows to exist (``run._insert_metadata`` runs first).
    Uses UPDATE...FROM, not INSERT...ON CONFLICT: Postgres checks NOT NULL
    at tuple formation, before conflict detection, so a partial-column
    upsert against NOT NULL columns always fails — even on matching ids.
    """
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("CREATE TEMP TABLE stage (id TEXT, embedding vector(768))")
        with cur.copy("COPY stage (id, embedding) FROM STDIN") as copy:  # type: ignore[attr-defined]
            count = 0
            for line in path.read_text().splitlines():
                obj = json.loads(line)
                vec = "[" + ",".join(repr(x) for x in obj["vector"]) + "]"
                copy.write_row((f"{obj['doc_id']}:{obj['chunk_ord']}", vec))
                count += 1
        cur.execute(
            "UPDATE chunks c SET embedding = s.embedding FROM stage s WHERE c.id = s.id"
        )
        cur.execute("DROP TABLE stage")
    conn.commit()  # type: ignore[attr-defined]
    return count
