"""Phase 4 pipeline: corpus chunks → normalized vectors → pgvector.

Needs the ``embed`` extra. Single command from the repo root::

    make embed

Reads ``data/processed/corpus/``, writes ``data/processed/embeddings/
chunk-NNNNN.jsonl`` (resumable: completed chunks are skipped), then COPYs
vectors into Postgres. ``--skip-load`` embeds without touching the DB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
CORPUS_DEFAULT = ROOT / "data" / "processed" / "corpus"
OUT_DEFAULT = ROOT / "data" / "processed" / "embeddings"
CHUNK_ROWS = 1000


def _read_corpus_texts(corpus: Path) -> list[tuple[str, int, str, str]]:
    """Read (doc_id, chunk_ord, title, text) via Spark (needs spark extra)."""
    from trustlayer.prep.session import build_session

    spark = build_session("trustlayer-embed-read", cores=2)
    try:
        rows = (
            spark.read.parquet(str(corpus))
            .select("doc_id", "chunk_ord", "title", "text")
            .orderBy("doc_id", "chunk_ord")
            .collect()
        )
        return [(r.doc_id, r.chunk_ord, r.title, r.text) for r in rows]
    finally:
        spark.stop()


def main(argv: list[str] | None = None) -> int:
    """Parse args, embed the corpus, write chunks, load pgvector."""
    from trustlayer.embed.model import DIM, embed_texts, load_text_model
    from trustlayer.embed.store import (
        ChunkResult,
        chunk_path,
        done_indices,
        write_chunk_file,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--limit", type=int, default=None, help="cap input rows")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args(argv)
    started = time.monotonic()

    rows = _read_corpus_texts(args.corpus)
    if args.limit is not None:
        rows = rows[: args.limit]
    n_chunks = math.ceil(len(rows) / args.chunk_rows)
    done = done_indices(args.out)
    todo = [i for i in range(n_chunks) if i not in done]
    print(f"rows={len(rows)} chunks={n_chunks} todo={len(todo)}", flush=True)
    if todo:
        model = load_text_model(args.device)
        for index in todo:
            part = rows[index * args.chunk_rows : (index + 1) * args.chunk_rows]
            texts = [f"{title.strip() or 'none'}\n{text}" for _, _, title, text in part]
            vectors = embed_texts(model, texts, batch_size=args.batch_size)
            if len(vectors) != len(part):
                msg = f"chunk {index}: {len(vectors)} vectors for {len(part)} rows"
                raise ValueError(msg)
            results = [
                ChunkResult(doc_id, ord_, tuple(vec))
                for (doc_id, ord_, _, _), vec in zip(part, vectors, strict=True)
            ]
            path = chunk_path(args.out, index)
            write_chunk_file(path, results)  # validates finite + dim
            print(f"wrote {path.name} ({len(results)} rows)", flush=True)

    stats: dict[str, int | float | str] = {
        "rows": len(rows),
        "chunks": n_chunks,
        "dim": DIM,
        "device": args.device,
        "embed_seconds": round(time.monotonic() - started, 1),
    }
    if torch.backends.mps.is_available() and args.device == "mps":
        stats["mps_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)

    if not args.skip_load:
        if not args.db_url:
            msg = "pgvector load needs --db-url or DATABASE_URL (or --skip-load)"
            raise ValueError(msg)
        import psycopg

        from trustlayer.embed.load import copy_chunk_file, ensure_schema

        t0 = time.monotonic()
        conn = psycopg.connect(args.db_url)
        try:
            ensure_schema(conn)
            # Documents/chunks metadata rows come from the corpus; vectors
            # land via COPY. Metadata insert first (FK), then vectors.
            _insert_metadata(conn, args.corpus)
            loaded = 0
            for index in range(n_chunks):
                loaded += copy_chunk_file(conn, chunk_path(args.out, index))
            stats["loaded_rows"] = loaded
        finally:
            conn.close()
        stats["load_seconds"] = round(time.monotonic() - t0, 1)

    text = json.dumps(stats, indent=2) + "\n"
    (args.out.parent / f"{args.out.name}-stats.json").write_text(text)
    print(text, end="")
    return 0


def _insert_metadata(conn: object, corpus: Path) -> None:
    """Upsert documents + chunk metadata (text, no vectors) from Parquet.

    Doc label is the max chunk weak label (confidential > internal >
    public) from Phase 3; docs missing labels default to internal.
    """
    from trustlayer.prep.session import build_session

    labels_path = corpus.parent / "labels"
    spark = build_session("trustlayer-embed-meta", cores=2)
    try:
        docs = spark.read.parquet(str(corpus)).select(
            "doc_id", "title", "thread_id", "chunk_ord", "text"
        )
        rank = {"public": 0, "internal": 1, "confidential": 2}
        doc_label: dict[str, str] = {}
        if labels_path.is_dir():
            for row in spark.read.parquet(str(labels_path)).collect():
                prev = doc_label.get(row.doc_id, "public")
                if rank.get(row.label, 0) > rank.get(prev, 0):
                    doc_label[row.doc_id] = row.label
                else:
                    doc_label.setdefault(row.doc_id, prev)
        doc_rows = [
            (r.doc_id, r.title, r.thread_id, doc_label.get(r.doc_id, "internal"))
            for r in docs.select("doc_id", "title", "thread_id").distinct().collect()
        ]
        chunk_rows = [
            (f"{r.doc_id}:{r.chunk_ord}", r.doc_id, r.chunk_ord, r.text)
            for r in docs.orderBy("doc_id", "chunk_ord").collect()
        ]
    finally:
        spark.stop()
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.executemany(
            "INSERT INTO documents (id, title, thread_id, label)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title,"
            " thread_id = EXCLUDED.thread_id, label = EXCLUDED.label",
            doc_rows,
        )
        cur.executemany(
            "INSERT INTO chunks (id, doc_id, ord, text) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text",
            chunk_rows,
        )
    conn.commit()  # type: ignore[attr-defined]


if __name__ == "__main__":
    sys.exit(main())
