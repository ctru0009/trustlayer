"""Classification-prefix embeddings: corpus → 768-d fp32 vectors.

Needs the ``embed`` extra (torch + sentence-transformers) and the ``spark``
extra (corpus read). Phase 7 re-encodes the full corpus with the
``Classification`` task prompt — stored Phase 4 vectors are
Document-prompted, and the LR rows need the classifier-tuned prefix.

Run from the repo root::

    make classify-vectors        # full run → data/processed/embeddings-cls/
    make classify-vectors PROBE=1  # first 1,000 chunks + extrapolation

Layout mirrors ``trustlayer.embed.run``: Spark read, resumable
``chunk-NNNNN.jsonl`` files via ``trustlayer.embed.store``, stats JSON
beside the output. No pgvector load — LR training reads the JSONL files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CORPUS_DEFAULT = ROOT / "data" / "processed" / "corpus"
OUT_DEFAULT = ROOT / "data" / "processed" / "embeddings-cls"
DOC_VEC_DEFAULT = ROOT / "data" / "processed" / "embeddings"
CHUNK_ROWS = 1000
PROBE_ROWS = 1000
PROMPT_NAME = "Classification"


def _read_corpus_texts(corpus: Path) -> list[tuple[str, int, str, str]]:
    """Read (doc_id, chunk_ord, title, text) via Spark (needs spark extra)."""
    from trustlayer.prep.session import build_session

    spark = build_session("trustlayer-vectors-cls", cores=2)
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


def _encode_classification(
    model: object, texts: list[str], batch_size: int
) -> list[list[float]]:
    """Encode texts with the Classification prompt; normalized fp32 768-d out."""
    from trustlayer.embed.model import MAX_LENGTH

    vectors = model.encode(  # type: ignore[attr-defined]
        texts,
        prompt_name=PROMPT_NAME,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        processing_kwargs={"text": {"max_length": MAX_LENGTH, "truncation": True}},
    )
    return [row.astype("float32").tolist() for row in vectors]


def _check_batch(vectors: list[list[float]], dim: int) -> None:
    """Assert every vector is finite with the expected dim (fail fast)."""
    for vec in vectors:
        if len(vec) != dim:
            msg = f"expected dim {dim}, got {len(vec)}"
            raise ValueError(msg)
        if not all(math.isfinite(x) for x in vec):
            msg = "non-finite value in Classification embedding batch"
            raise ValueError(msg)


def mean_cosine(a: list[list[float]], b: list[list[float]]) -> float:
    """Mean row-wise cosine similarity between two same-shape vector lists."""
    total = 0.0
    for va, vb in zip(a, b, strict=True):
        num = sum(x * y for x, y in zip(va, vb, strict=True))
        da = math.sqrt(sum(x * x for x in va))
        db = math.sqrt(sum(y * y for y in vb))
        total += num / (da * db)
    return total / len(a) if a else 0.0


def prompt_sanity_check(out_dir: Path, doc_dir: Path, n_rows: int = 100) -> float:
    """Compare cls vs Document vectors on the first ``n_rows`` shared rows.

    Returns the mean cosine similarity; Classification and Document
    prompts must produce clearly different vectors (< 0.99), proving the
    prompt applied rather than silently falling back to a default.
    Rows are joined by (doc_id, chunk_ord) so chunk sizing can't
    silently misalign the comparison.
    """
    from trustlayer.embed.store import read_chunk_file

    cls_rows = read_chunk_file(out_dir / "chunk-00000.jsonl")[:n_rows]
    doc_by_key = {
        (r.doc_id, r.chunk_ord): r.vector
        for r in read_chunk_file(doc_dir / "chunk-00000.jsonl")
    }
    pairs = [(r.vector, doc_by_key[(r.doc_id, r.chunk_ord)]) for r in cls_rows]
    cls_vecs = [[float(x) for x in v] for v, _ in pairs]
    doc_vecs = [[float(x) for x in v] for _, v in pairs]
    return mean_cosine(cls_vecs, doc_vecs)


def main(argv: list[str] | None = None) -> int:
    """Parse args, encode the corpus with the Classification prompt."""
    import torch

    from trustlayer.embed.model import DIM, load_text_model
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
    parser.add_argument(
        "--doc-vectors",
        type=Path,
        default=DOC_VEC_DEFAULT,
        help="Document-prompt vectors for the prompt sanity check",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip the cls-vs-Document cosine sanity check",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()

    rows = _read_corpus_texts(args.corpus)
    if args.limit is not None:
        rows = rows[: args.limit]
    n_chunks = math.ceil(len(rows) / args.chunk_rows)
    done = done_indices(args.out)
    todo = [i for i in range(n_chunks) if i not in done]
    print(f"rows={len(rows)} chunks={n_chunks} todo={len(todo)}", flush=True)
    prompt_text = ""
    if todo:
        model = load_text_model(args.device)
        if PROMPT_NAME not in model.prompts:
            msg = f"prompt {PROMPT_NAME!r} missing from model.prompts"
            raise ValueError(msg)
        prompt_text = str(model.prompts[PROMPT_NAME])
        print(f"prompt {PROMPT_NAME!r} -> {prompt_text!r}", flush=True)
        for index in todo:
            part = rows[index * args.chunk_rows : (index + 1) * args.chunk_rows]
            texts = [f"{title.strip() or 'none'}\n{text}" for _, _, title, text in part]
            vectors = _encode_classification(model, texts, args.batch_size)
            if len(vectors) != len(part):
                msg = f"chunk {index}: {len(vectors)} vectors for {len(part)} rows"
                raise ValueError(msg)
            _check_batch(vectors, DIM)
            results = [
                ChunkResult(doc_id, ord_, tuple(vec))
                for (doc_id, ord_, _, _), vec in zip(part, vectors, strict=True)
            ]
            path = chunk_path(args.out, index)
            write_chunk_file(path, results)  # validates finite + dim again
            print(f"wrote {path.name} ({len(results)} rows)", flush=True)

    stats: dict[str, int | float | str] = {
        "rows": len(rows),
        "chunks": n_chunks,
        "dim": DIM,
        "prompt_name": PROMPT_NAME,
        "prompt_text": prompt_text,
        "device": args.device,
        "embed_seconds": round(time.monotonic() - started, 1),
    }
    if torch.backends.mps.is_available() and args.device == "mps":
        stats["mps_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)

    if not args.skip_sanity:
        sim = prompt_sanity_check(args.out, args.doc_vectors)
        stats["cls_vs_doc_mean_cosine"] = round(sim, 4)
        print(f"cls-vs-Document mean cosine on 100 rows: {sim:.4f}", flush=True)
        if sim >= 0.99:
            msg = f"sanity check failed: mean cosine {sim:.4f} >= 0.99"
            raise ValueError(msg)

    text = json.dumps(stats, indent=2) + "\n"
    (args.out.parent / f"{args.out.name}-stats.json").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
