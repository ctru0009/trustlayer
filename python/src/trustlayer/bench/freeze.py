"""Freeze the Phase 6 query sets (spec section 10, protocol step 1).

Two sets from the held-out test split, relevance = source document:

- ``subject``: informative email subjects (length >= 15 chars, not a
  bare Re:/Fw:); tests short-query → long-doc retrieval.
- ``chunk``: held-out chunk texts (ord > 0 preferred, >= 100 chars);
  tests paraphrase-adjacent retrieval.

Frozen output (git-ignored): ``data/bench/queries.jsonl`` — one row per
query: ``{qid, kind, text, relevant_doc}``. A content-hash manifest is
printed for the lesson so the freeze is auditable.

Usage: ``make bench-freeze`` (needs DATABASE_URL).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT_DEFAULT = ROOT / "data" / "bench" / "queries.jsonl"
PER_KIND = 150


def _informative(title: str) -> bool:
    s = title.strip()
    low = s.lower()
    return len(s) >= 15 and not low.startswith(("re:", "fw:", "fwd:"))


def main(argv: list[str] | None = None) -> int:
    """Select and freeze queries; print counts + sha256 manifest."""
    import psycopg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--per-kind", type=int, default=PER_KIND)
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args(argv)
    if not args.db_url:
        msg = "bench-freeze needs --db-url or DATABASE_URL"
        raise ValueError(msg)

    conn = psycopg.connect(args.db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT d.id, d.title FROM documents d"
                " WHERE d.id LIKE 'test:%%' AND d.modality = 'text'"
                " ORDER BY d.id"
            )
            docs = cur.fetchall()
            cur.execute(
                "SELECT c.doc_id, c.text FROM chunks c"
                " WHERE c.doc_id LIKE 'test:%%' AND c.ord > 0"
                " AND LENGTH(c.text) >= 100"
                " ORDER BY c.doc_id, c.ord"
            )
            chunks = cur.fetchall()
    finally:
        conn.close()

    subjects = [
        {
            "qid": f"subj-{i:03d}",
            "kind": "subject",
            "text": title.strip(),
            "relevant_doc": doc_id,
        }
        for i, (doc_id, title) in enumerate(d for d in docs if _informative(d[1]))
    ][: args.per_kind]
    seen: set[str] = set()
    chunk_qs = []
    for doc_id, text in chunks:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        head, _, rest = text[:500].partition(" ")
        clean = rest if head and rest else text[:500]
        chunk_qs.append(
            {
                "qid": f"chunk-{len(chunk_qs):03d}",
                "kind": "chunk",
                "text": clean,
                "relevant_doc": doc_id,
            }
        )
        if len(chunk_qs) >= args.per_kind:
            break
    queries = subjects + chunk_qs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]
    print(
        json.dumps(
            {
                "queries": len(queries),
                "subjects": len(subjects),
                "chunks": len(chunk_qs),
                "sha256_16": digest,
                "path": str(args.out.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    if len(subjects) < args.per_kind or len(chunk_qs) < args.per_kind:
        print("SHORTFALL: fewer queries than requested", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
