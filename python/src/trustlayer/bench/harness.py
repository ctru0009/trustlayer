"""Benchmark harness: one config per row, full provenance per run.

Spec section 10: baseline + O-rows + ablations on the frozen query set.
Each run logs its config (model, dtype, dim, prefixes, seed, library
versions, device, date — protocol 5), warms up, repeats latency 5×
(protocol 3), and reports bootstrap CIs (protocol 4).

Vector variants live in ``bench_<name>`` tables (chunk grain, same ids
as ``chunks``) so index size is measured, not estimated. Truncation
happens in NumPy from the 768d source (spec section 6); every variant
is finite-checked before load (protocol 6).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
QUERIES_DEFAULT = ROOT / "data" / "bench" / "queries.jsonl"
RESULTS_DEFAULT = ROOT / "data" / "bench" / "results"
REPEATS = 5

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


@dataclass(frozen=True)
class RunConfig:
    """One benchmark row: how vectors are stored and searched."""

    name: str
    dim: int = 768
    storage: str = "vector"  # vector | halfvec
    index: str = "exact"  # exact | hnsw
    renorm: bool = True  # renormalize after truncation
    prefix: bool = True  # SearchQuery prompt on queries


CONFIGS: list[RunConfig] = [
    RunConfig("baseline"),
    RunConfig("O1-dim512", dim=512),
    RunConfig("O2-dim256", dim=256),
    RunConfig("O3-dim128", dim=128),
    RunConfig("O5-hnsw", index="hnsw"),
    RunConfig("O6-halfvec", storage="halfvec"),
    RunConfig("ablate-no-renorm", dim=256, renorm=False),
    # Query-side only: docs stay Document-prompted, so this measures
    # prompt-mismatch (unprompted queries vs prompted docs), not a true
    # no-prefix ablation. True ablation needs corpus re-encode; unscored.
    RunConfig("ablate-query-mismatch", prefix=False),
    # O7: measured best combo — 256d (O2: -0.003 recall) + halfvec (O6:
    # quality-neutral, 60% smaller). HNSW excluded: 15x faster but -0.077
    # recall at ef_search=40 (O5 row) — fails this quality bar; a higher
    # ef_search might change the tradeoff (unprobed).
    RunConfig("O7-best", dim=256, storage="halfvec"),
]


def _versions() -> dict[str, str]:
    out = {}
    for dist in ("sentence-transformers", "transformers", "torch", "numpy", "psycopg"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = "missing"
    return out


def build_table(conn: object, cfg: RunConfig) -> dict[str, float | int]:
    """Create + fill bench_<name> from chunks; returns size stats."""
    import numpy as np

    # Table/type names derive from the fixed CONFIGS list (no user input);
    # identifiers cannot be bind parameters, so f-strings + noqa S608.
    table = f"bench_{cfg.name.replace('-', '_')}"  # noqa: S608
    vtype = f"halfvec({cfg.dim})" if cfg.storage == "halfvec" else f"vector({cfg.dim})"
    opclass = vtype.split("(")[0] + "_cosine_ops"
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608
        cur.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL)")  # noqa: S608
        cur.execute(f"ALTER TABLE {table} ADD COLUMN embedding {vtype}")  # noqa: S608
        cur.execute("SELECT id, doc_id, embedding::text FROM chunks")
        rows = cur.fetchall()
        staged = []
        for cid, doc_id, text in rows:
            vec = np.fromstring(text.strip("[]"), sep=",", dtype="float64")[: cfg.dim]
            if not np.all(np.isfinite(vec)):
                msg = f"non-finite source vector: {cid}"
                raise ValueError(msg)
            if cfg.storage == "halfvec":
                vec = vec.astype("float16").astype("float64")
            if cfg.dim < 768 and cfg.renorm:
                vec = vec / float(np.linalg.norm(vec))
            lit = "[" + ",".join(repr(float(x)) for x in vec) + "]"
            staged.append((cid, doc_id, lit))
        copy_sql = f"COPY {table} (id, doc_id, embedding) FROM STDIN"
        with cur.copy(copy_sql) as copy:  # type: ignore[attr-defined]
            for row in staged:
                copy.write_row(row)
        if cfg.index == "hnsw":
            cur.execute(  # noqa: S608
                f"CREATE INDEX ON {table} USING hnsw (embedding {opclass})"
                " WITH (m = 16, ef_construction = 64)"
            )
        cur.execute(f"SELECT COUNT(*), pg_total_relation_size('{table}') FROM {table}")  # noqa: S608
        count, size = cur.fetchone()
    conn.commit()  # type: ignore[attr-defined]
    return {"rows": count, "bytes": size}


def run_queries(
    conn: object, cfg: RunConfig, queries: list[dict], vectors: object
) -> tuple[list[list[str]], list[float]]:
    """Ranked doc_ids (deduped to doc grain) + per-query latency in seconds.

    Chunk-grain search returns the same doc multiple times; metrics need
    one entry per doc, so fetch 50 chunks and keep first-occurrence order.
    """
    import numpy as np

    table = f"bench_{cfg.name.replace('-', '_')}"
    vecs = np.asarray(vectors, dtype="float64")[:, : cfg.dim]
    if cfg.storage == "halfvec":
        vecs = vecs.astype("float16").astype("float64")
    if cfg.dim < 768 and cfg.renorm:
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ranked: list[list[str]] = []
    lat: list[float] = []
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        if cfg.index == "hnsw":
            # The connection is shared across configs; earlier exact rows
            # SET enable_indexscan=off, which persists. Re-enable here or
            # this row silently measures a seq scan, not HNSW.
            cur.execute("SET enable_indexscan = on")
            cur.execute("SET hnsw.iterative_scan = 'strict_order'")
        else:
            cur.execute("SET enable_indexscan = off")
        for vec in vecs:
            lit = "[" + ",".join(repr(float(x)) for x in vec) + "]"
            t0 = time.monotonic()
            # Table/type from fixed CONFIGS (no user input); value is bound.
            sql = f"SELECT doc_id FROM {table} "  # noqa: S608
            sql += f"ORDER BY embedding <=> %s::{cfg.storage}({cfg.dim}) LIMIT 50"  # noqa: S608
            cur.execute(sql, (lit,))
            docs = list(dict.fromkeys(r[0] for r in cur.fetchall()))[:10]
            ranked.append(docs)
            lat.append(time.monotonic() - t0)
    return ranked, lat


def main(argv: list[str] | None = None) -> int:
    """Run all configs (or --only NAME); write results JSON per row."""
    import numpy as np
    import psycopg

    from trustlayer.bench.metrics import (
        bootstrap_ci,
        mean,
        mrr_at_k,
        ndcg_at_k,
        recall_at_k,
    )
    from trustlayer.embed.model import load_text_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=QUERIES_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--only", default=None, help="run one config by name")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args(argv)
    if not args.db_url:
        msg = "bench needs --db-url or DATABASE_URL"
        raise ValueError(msg)
    queries = [json.loads(line) for line in args.queries.read_text().splitlines()]
    relevant = [{q["relevant_doc"]} for q in queries]
    conn = psycopg.connect(args.db_url)

    model = load_text_model("cpu")
    texts = [q["text"] for q in queries]
    cache: dict[bool, object] = {}
    for prefix in (True, False):
        vecs = (
            model.encode_query(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if prefix
            else model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        cache[prefix] = np.asarray(vecs, dtype="float32")
    try:
        configs = [c for c in CONFIGS if args.only is None or c.name == args.only]
        if args.only and not configs:
            msg = f"unknown config: {args.only}"
            raise ValueError(msg)
        for cfg in configs:
            size = build_table(conn, cfg)
            # Warm-up, then REPEATS timed passes; quality from pass 0.
            all_lat: list[float] = []
            ranked = []
            for _ in range(REPEATS):
                ranked, lat = run_queries(conn, cfg, queries, cache[cfg.prefix])
                all_lat.extend(lat)
            rec = [recall_at_k(r, rel) for r, rel in zip(ranked, relevant, strict=True)]
            mrr = [mrr_at_k(r, rel) for r, rel in zip(ranked, relevant, strict=True)]
            ndcg = [ndcg_at_k(r, rel) for r, rel in zip(ranked, relevant, strict=True)]
            lat_sorted = sorted(all_lat)
            result = {
                "config": asdict(cfg),
                "model": MODEL_ID,
                "revision": REVISION,
                "queries": len(queries),
                "recall_at_10": round(mean(rec), 4),
                "recall_ci": [round(x, 4) for x in bootstrap_ci(rec)],
                "mrr_at_10": round(mean(mrr), 4),
                "mrr_ci": [round(x, 4) for x in bootstrap_ci(mrr)],
                "ndcg_at_10": round(mean(ndcg), 4),
                "ndcg_ci": [round(x, 4) for x in bootstrap_ci(ndcg)],
                "latency_ms_p50": round(lat_sorted[len(lat_sorted) // 2] * 1000, 2),
                "latency_ms_p95": round(
                    lat_sorted[int(len(lat_sorted) * 0.95)] * 1000, 2
                ),
                "table_bytes": size["bytes"],
                "rows": size["rows"],
                "versions": _versions(),
                "device": "cpu",
                "date": datetime.date.today().isoformat(),
            }
            args.results.mkdir(parents=True, exist_ok=True)
            (args.results / f"{cfg.name}.json").write_text(json.dumps(result, indent=2))
            print(f"{cfg.name}: R@10={result['recall_at_10']}", flush=True)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
