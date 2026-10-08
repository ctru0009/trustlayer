"""Cross-lingual rows: EN→VI and VI→VI on the 500-article VI sample.

EN queries are hand-written (this file) against known-relevant VI
articles — proper nouns and concepts with unambiguous targets. VI
queries reuse article titles (source article relevant). Metrics and
bootstrap CIs match the main harness; results land in
``data/bench/results/xling-*.json``.

VI→EN is NOT measured: it needs a parallel EN corpus on the same
topics, and translating 500 articles is out of scope. Recorded limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS_DEFAULT = ROOT / "data" / "bench" / "results"

# (query, relevant vi:id) — targets verified against /tmp/vi_titles.txt.
EN_QUERIES: list[tuple[str, str]] = [
    ("Internet Society nonprofit organization", "vi:000"),
    ("Vietnamese language", "vi:001"),
    ("Ho Chi Minh City", "vi:005"),
    ("World Wide Web Consortium standards", "vi:007"),
    ("United States of America", "vi:010"),
    ("capital city Hanoi", "vi:014"),
    ("Cambodia country", "vi:015"),
    ("Vietnam War history", "vi:066"),
    ("English language", "vi:071"),
    ("Canada country profile", "vi:072"),
    ("Linux operating system", "vi:094"),
    ("General Vo Nguyen Giap", "vi:096"),
    ("computer and computing", "vi:130"),
    ("Firefox web browser", "vi:137"),
    ("GNU free software project", "vi:143"),
    ("Internet history and technology", "vi:145"),
    ("PHP programming language", "vi:152"),
    ("Java technology programming", "vi:155"),
    ("World Wide Web invention", "vi:184"),
    ("Jesus Christ biography", "vi:185"),
    ("Wikipedia online encyclopedia", "vi:188"),
    ("HTML markup language", "vi:204"),
    ("physics science", "vi:207"),
    ("astronomy planets solar system", "vi:210"),
    ("China country", "vi:224"),
    ("Albert Einstein relativity", "vi:326"),
    ("Isaac Newton gravity", "vi:327"),
    ("Paris capital of France", "vi:359"),
    ("Nobel Prize laureates", "vi:334"),
    ("Harry Potter novels", "vi:316"),
]


def main(argv: list[str] | None = None) -> int:
    """Run EN→VI + VI→VI rows; write results JSON."""
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
    parser.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args(argv)
    if not args.db_url:
        msg = "xling needs --db-url or DATABASE_URL"
        raise ValueError(msg)

    conn = psycopg.connect(args.db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, title FROM bench_vi ORDER BY id")
            articles = cur.fetchall()
    finally:
        conn.close()

    model = load_text_model("cpu")

    def encode(texts: list[str]) -> object:
        return np.asarray(
            model.encode_query(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )

    def search(vecs: object) -> list[list[str]]:
        conn = psycopg.connect(args.db_url)
        try:
            with conn.cursor() as cur:  # type: ignore[attr-defined]
                out = []
                for vec in np.asarray(vecs):
                    lit = "[" + ",".join(repr(float(x)) for x in vec) + "]"
                    cur.execute(
                        "SELECT id FROM bench_vi"
                        " ORDER BY embedding <=> %s::vector LIMIT 10",
                        (lit,),
                    )
                    out.append([r[0] for r in cur.fetchall()])
                return out
        finally:
            conn.close()

    rows = {}
    # EN→VI: hand-written queries.
    vecs = encode([q for q, _ in EN_QUERIES])
    ranked = search(vecs)
    rel = [{doc} for _, doc in EN_QUERIES]
    rows["xling-en-vi"] = (ranked, rel)
    # VI→VI: article titles (first 100), source article relevant.
    vi_qs = [(t, i) for i, t in articles[:100]]
    vecs = encode([t for t, _ in vi_qs])
    ranked = search(vecs)
    rel = [{i} for _, i in vi_qs]
    rows["xling-vi-vi"] = (ranked, rel)

    for name, (ranked, rel) in rows.items():
        rec = [recall_at_k(r, s) for r, s in zip(ranked, rel, strict=True)]
        mrr = [mrr_at_k(r, s) for r, s in zip(ranked, rel, strict=True)]
        ndcg = [ndcg_at_k(r, s) for r, s in zip(ranked, rel, strict=True)]
        result = {
            "queries": len(ranked),
            "recall_at_10": round(mean(rec), 4),
            "recall_ci": [round(x, 4) for x in bootstrap_ci(rec)],
            "mrr_at_10": round(mean(mrr), 4),
            "mrr_ci": [round(x, 4) for x in bootstrap_ci(mrr)],
            "ndcg_at_10": round(mean(ndcg), 4),
            "ndcg_ci": [round(x, 4) for x in bootstrap_ci(ndcg)],
            "device": "cpu",
        }
        args.results.mkdir(parents=True, exist_ok=True)
        (args.results / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(f"{name}: R@10={result['recall_at_10']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
