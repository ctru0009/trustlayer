"""Leak test: every user × every dev query must return zero forbidden docs.

Spec section 12: for every synthetic user and every query in the
evaluation set, assert zero returned documents they are not allowed to
see. Queries are dev-split first-chunk texts (truncated); query vectors
use the SearchQuery prompt via ``model.encode_query``.

Usage: ``make leak`` (needs DATABASE_URL + the ``embed`` extra).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEV_DEFAULT = ROOT / "data" / "gold" / "splits" / "dev.txt"


def main(argv: list[str] | None = None) -> int:
    """Run the leak test; exit 1 on any violation."""
    import numpy as np
    import psycopg

    from trustlayer.embed.model import load_text_model
    from trustlayer.retrieve.acl import USERS, visible
    from trustlayer.retrieve.search import search

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=DEV_DEFAULT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="cap queries")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args(argv)
    if not args.db_url:
        msg = "leak test needs --db-url or DATABASE_URL"
        raise ValueError(msg)
    started = time.monotonic()

    doc_ids = args.dev.read_text().split()
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]
    conn = psycopg.connect(args.db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ON (c.doc_id) c.text FROM chunks c"
                " WHERE c.doc_id = ANY(%s) ORDER BY c.doc_id, c.ord",
                (doc_ids,),
            )
            queries = [text[:200] for (text,) in cur.fetchall()]
        model = load_text_model("cpu")
        vectors = np.asarray(
            model.encode_query(
                queries,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )
        # search() re-checks internally and raises on mismatch; the audit
        # below re-verifies every hit against acl.visible without trusting
        # search()'s assert path.
        violations = 0
        checked = 0
        with conn.cursor() as cur:
            for user, roles in USERS.items():
                for vec in vectors:
                    for hit in search(conn, vec.tolist(), user, top_k=args.top_k):
                        checked += 1
                        cur.execute(
                            "SELECT label, allowed_roles FROM documents WHERE id = %s",
                            (hit.doc_id,),
                        )
                        label, allowed = cur.fetchone()
                        if not visible(label, list(allowed), list(roles)):
                            violations += 1
    finally:
        conn.close()
    stats = {
        "users": len(USERS),
        "queries": len(vectors),
        "hits_checked": checked,
        "violations": violations,
        "seconds": round(time.monotonic() - started, 1),
    }
    print(json.dumps(stats, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
