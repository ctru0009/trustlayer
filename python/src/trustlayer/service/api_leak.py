"""Leak test through the gateway API (Phase 8 exit criteria).

Spec section 12: for every synthetic user and every evaluation query,
assert zero returned documents they are not allowed to see — in BOTH
ask modes, through HTTP, not the Python search() path. The gateway owns
retrieval SQL + the re-check; this test audits its answers against
``acl.visible`` ground truth read straight from the database.

Unlike the direct leak test (1,729 queries × 4 users over function
calls), the API path pays HTTP + per-query embed latency, so the
default is a bounded sample (``--limit 20`` queries); ``--full`` runs
all dev queries. Answer mode is sampled separately (``--answer-n 2``
queries × 4 users) because each call pays a Gemma generation.

Usage: ``make api-leak`` (needs the stack up + DATABASE_URL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEV_DEFAULT = ROOT / "data" / "gold" / "splits" / "dev.txt"


def _login(base: str, user: str) -> str:
    """Log in a demo user; return the JWT (raises on failure)."""
    import httpx

    resp = httpx.post(f"{base}/auth/login", json={"username": user}, timeout=30.0)
    if resp.status_code != 200:
        msg = f"login {user} failed: {resp.status_code} {resp.text[:200]}"
        raise RuntimeError(msg)
    return str(resp.json()["token"])


def _ask(base: str, token: str, query: str, mode: str, top_k: int) -> dict[str, Any]:
    """POST /ask; return the parsed body (raises on non-200)."""
    import httpx

    resp = httpx.post(
        f"{base}/ask",
        json={"query": query, "mode": mode, "top_k": top_k},
        headers={"Authorization": f"Bearer {token}"},
        timeout=300.0,
    )
    if resp.status_code != 200:
        msg = f"/ask {mode} failed: {resp.status_code} {resp.text[:200]}"
        raise RuntimeError(msg)
    return dict(resp.json())


def main(argv: list[str] | None = None) -> int:
    """Run the API leak test; exit 1 on any violation or shortfall."""
    import httpx  # noqa: F401 — fail fast when the service extra is missing
    import psycopg

    from trustlayer.retrieve.acl import USERS, visible

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://localhost:8080")
    parser.add_argument("--dev", type=Path, default=DEV_DEFAULT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--answer-n", type=int, default=2)
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args(argv)
    if not args.db_url:
        msg = "api leak test needs --db-url or DATABASE_URL"
        raise ValueError(msg)
    started = time.monotonic()

    doc_ids = args.dev.read_text().split()
    if not args.full:
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

        tokens = {user: _login(args.gateway, user) for user in USERS}
        violations = 0
        checked = 0
        shortfalls: list[str] = []

        def audit(user: str, roles: list[str], hits: list[dict[str, Any]]) -> None:
            """Verify every hit is visible to the user (ground truth: DB)."""
            nonlocal violations, checked
            with conn.cursor() as cur:
                for hit in hits:
                    checked += 1
                    cur.execute(
                        "SELECT label, allowed_roles FROM documents WHERE id = %s",
                        (hit["doc_id"],),
                    )
                    row = cur.fetchone()
                    if row is None:
                        violations += 1
                        continue
                    label, allowed = row
                    if not visible(label, list(allowed), roles):
                        violations += 1
                        print(f"VIOLATION {user} saw {hit['doc_id']} ({label})")

        for user, roles in USERS.items():
            for qi, query in enumerate(queries):
                body = _ask(args.gateway, tokens[user], query, "fast", args.top_k)
                hits = list(body["results"])
                if len(hits) < args.top_k:
                    shortfalls.append(f"{user}/fast/q{qi}:{len(hits)}")
                if body["answer"] is not None:
                    print(f"MODE-LEAK {user}/q{qi}: fast mode returned an answer")
                    violations += 1
                audit(user, list(roles), hits)
            for qi in range(min(args.answer_n, len(queries))):
                body = _ask(args.gateway, tokens[user], queries[qi], "answer", 5)
                hits = list(body["results"])
                if len(hits) < 5:
                    shortfalls.append(f"{user}/answer/q{qi}:{len(hits)}")
                if not body.get("answer"):
                    print(f"EMPTY-ANSWER {user}/q{qi}")
                cited = {c["doc_id"] for c in body.get("citations", [])}
                returned = {h["doc_id"] for h in hits}
                if not cited <= returned:
                    print(f"CITATION-DRIFT {user}/q{qi}: {cited - returned}")
                    violations += 1
                audit(user, list(roles), hits)

        expected = len(USERS) * (
            len(queries) * args.top_k + min(args.answer_n, len(queries)) * 5
        )
        stats = {
            "gateway": args.gateway,
            "queries": len(queries),
            "checked": checked,
            "expected": expected,
            "violations": violations,
            "shortfalls": shortfalls[:10],
            "seconds": round(time.monotonic() - started, 1),
        }
        print(json.dumps(stats, indent=2))
        if checked != expected or shortfalls:
            print(f"SHORTFALL: {checked}/{expected} hits", file=sys.stderr)
            return 1
        return 1 if violations else 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
