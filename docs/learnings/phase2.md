# Phase 2 — Data prep with PySpark: lesson

Dev corpus: AESLC (`Yale-LILY/aeslc` rev `2305f2e`), 18,302 emails across
train/validation/test. One command reproduces everything: `make prep`
(download separately via `data/scripts/download_aeslc.py`; data is git-ignored).

## Run results (2026-10-08, Mac, `local[2]`, 8.9s)

| Stage | Rows |
|---|---|
| Raw input | 18,302 |
| After cleaning (2 emptied) | 18,300 |
| After exact dedupe (−1,301) | 16,999 |
| After normalized dedupe (−38) | 16,961 docs |
| Chunked output | 23,267 chunks |

Output: `data/processed/corpus/` (7-column schema, see `schema.py`) +
`corpus-stats.json`. Integrity verified: 0 empty texts, 0 length mismatches,
0 nulls, 0 non-contiguous ordinals. 16,961 docs across 14,603 pseudo-threads;
chunks/doc median 1, q99 44.

## What a Spark shuffle costs

`dedupe` (`dropDuplicates`) and `orderBy` are the only shuffles here: each
forces a full data exchange across partitions — every row serialized, hashed,
sent over the (loopback) network, and re-sorted. On 18k rows it's milliseconds;
the lesson is structural: shuffles scale with data × partitions, and
`orderBy("doc_id", "chunk_ord")` on the full corpus forces a single global
sort. Kept because deterministic output ordering makes re-runs diffable; on
the full corpus, drop it or partition by `doc_id` first.

`shuffle.partitions` is set to `cores * 2` in `session.py` — the default 200
partitions would mean 200 tiny tasks on a laptop.

## Working-rule override (recorded per your instruction)

The roadmap's working rule said Spark transforms are human-written first, with
the agent scaffolding boilerplate only. You overrode this for Phase 2
("No human write"): the agent implemented the full pipeline — `clean`,
`chunk`, `dedupe`, `run`, tests — and recorded the decisions here instead.
The rule's intent (you understand the core logic, not just run it) is
preserved by this lesson: read it as the design review you would have
written, and the override stands for future phases unless you say otherwise.

## Why dedupe before splitting

Duplicates that straddle a train/test split leak test content into training:
the model "recalls" instead of generalizing, and benchmark scores inflate.
Dedupe first (exact, then whitespace/case-normalized: −1,339 here, 7.3%),
split after. The normalized pass catches near-identical forwards with
different whitespace — cheap insurance on email data.

Order within the pipeline also matters: clean → dedupe → chunk. Cleaning
first means quotes/signatures don't create false distinctions between
identical messages; chunking last means dedupe compares whole documents, not
fragments that could falsely match.

## Why split by thread

Random splits put a message and its reply in different sets — same content,
both sides of the evaluation, same leakage as duplicates. Threads (here:
pseudo-threads from normalized subject lines, since AESLC ships no thread
metadata) keep reply chains together. An approximation, documented in
`schema.py`: `Re:`/`Fwd:` prefixes are stripped before hashing so replies
group with the original.

## Design decisions and why

- **Pure-Python `clean`/`chunk`, Spark only in `dedupe`/`run`.** Text logic is
  unit-testable without Java (15 tests, 0.01s); Spark stages get one slower
  end-to-end test. Also keeps `import trustlayer.prep` working without the
  spark extra (CI runs `uv sync` without extras) — `run` is lazily imported.
- **UDFs built inside `build_corpus`, `useArrow=False`.** `@F.udf` at module
  scope needs a live session at import time (collection error); pandas/Arrow
  aren't installed, and the probe emits a `UserWarning` — `useArrow=False`
  skips it. Plain Python UDFs are fast enough at this scale.
- **Chunking 1000/100 chars with word-boundary snap.** Median body is ~430
  chars, q90 ~1.3k: most emails yield 1 chunk, long ones split at word
  boundaries with 100 chars of overlap so boundary-spanning sentences stay
  readable in one chunk. q99 is 44 chunks/doc (the 39k-char outliers).
- **`doc_id = {split}:{sha8}`** — AESLC has no ids; hashing body+subject gives
  stable ids across re-runs without a state file.
- **`char_len` materialized** — lets Phase 4 filter tiny chunks without
  re-scanning text.

## Tips and tricks

- Inspect with aggregates, never eyeball: row counts, length quantiles,
  distinct counts told us the chunk size without reading a single email.
- `approxQuantile` with 0.01 error is instant; exact percentiles would sort.
- Parquet-native source (`Yale-LILY/aeslc` ships `.parquet`) means no
  `datasets` dependency — `urllib` + pinned revision URL suffices.
- Run the pipeline once with `--limit 100` before the full run; schema drift
  is asserted after every write (`CORPUS_SCHEMA` comparison in `run.main`).
- Spark's default log level buries real output; `setLogLevel("WARN")` in the
  session builder keeps `make prep` readable.

Spec pointers: §5, F1.
