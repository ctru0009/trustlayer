"""pgvector load: chunk files → documents/chunks tables via COPY.

Needs the ``embed`` extra (psycopg). Schema matches spec section 8.5.
All loads are re-runnable: metadata upserts, vectors upsert via a temp
staging table (COPY has no ON CONFLICT).
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'aeslc',
    title TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT 'text',
    lang TEXT NOT NULL DEFAULT 'en',
    thread_id TEXT NOT NULL,
    label TEXT NOT NULL,
    allowed_roles TEXT[] NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents (id),
    ord INT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(768)
);
"""


def ensure_schema(conn: object) -> None:
    """Create extension + tables if missing (idempotent)."""
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(SCHEMA_SQL)
    conn.commit()  # type: ignore[attr-defined]


def copy_chunk_file(conn: object, path: Path) -> int:
    """COPY one chunk file's vectors into chunks; returns rows loaded.

    Requires metadata rows to exist (``run._insert_metadata`` runs first):
    the upsert only touches ``embedding``, and ``doc_id``/``ord``/``text``
    are NOT NULL.
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
            "INSERT INTO chunks (id, embedding) SELECT id, embedding FROM stage"
            " ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
        )
        cur.execute("DROP TABLE stage")
    conn.commit()  # type: ignore[attr-defined]
    return count
