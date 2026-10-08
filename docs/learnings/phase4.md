# Phase 4 — Embedding pipeline (text, image): lesson

Text embeddings via EmbeddingGemma-2 on Colab CUDA; pgvector load locally.
`make embed` runs the full pipeline (embed + load); `--skip-load` embeds
only. Resume is chunk-atomic: completed `chunk-NNNNN.jsonl` files are
skipped on re-run.

## Run results (2026-10-08)

| Stage | Result |
|---|---|
| Colab CUDA embed (T4) | 23,267 rows, 24 chunks, 726.3s (~32 docs/s) |
| Local pgvector load | 23,267 vectors, 15.3s |
| Index state | 16,961 docs, 23,267 chunks, 0 null embeddings |
| Doc labels (max chunk weak label) | public 11,523 / internal 4,724 / confidential 714 |
| Self-search sanity | dist 0.0 self-hit, neighbours at 0.19–0.20 |

Model pinned: `google/embeddinggemma-2` rev `914f7f8` (Apache-2.0,
ungated). 1.43 GB download (single bf16 safetensors; text-only selection
at load), 1,084 MB MPS RSS for text-only fp32 (measured).

## Why not float16?

The model card warns of NaN or silently degraded embeddings in float16 —
and "silently" is the operative word: no error, just worse vectors that
poison every downstream metric. fp32 throughout (checkpoint is bf16;
override at load). The pipeline asserts finite + dim-768 on every vector
at write time and read time, so a numerical failure fails loudly.

## What is an embedding space?

768 numbers per chunk where cosine distance ≈ meaning distance. Verified
concretely: a chunk searched against itself returns distance 0.0, and
neighbours at 0.19–0.20 share obvious topical similarity. Prefixes shape
this space (below); truncation + renorm preserve it (Phase 6).

## Why do prefixes matter?

Retrieval is asymmetric: queries get `task: search result | query: `,
documents get `title: none | text: `. The prefix tokens participate in
mean pooling (`include_prompt: true`), so omitting them shifts every
vector silently — same dims, worse neighbours, no error. The double-prefix
trap is real: `encode_document` applies the Document prompt itself, so
manually formatting `title: X | text: Y` then calling `encode_document`
prefixes twice. Raw text + `encode_document` is the one correct path.

## MPS stall: why Colab won

Local MPS run: 4 chunks at ~1 min each, then chunk 4 took 16 min (eager
attention L² on long-tail batches + MPS graph-cache growth per unique
shape). Killed at 5/24. Colab T4 did all 24 in 726s with zero drama.
Lesson: MPS is fine for verification (CPU-vs-MPS cosines ≥ 0.9999999 on
the canary set — numerically identical) but untrusted for unattended bulk
runs. The notebook (`notebooks/phase4_embed.ipynb`) is now the proven
bulk path: Java install + corpus build + embed + per-chunk Drive backup.

## Headline finding: partial-column upsert vs NOT NULL

`INSERT INTO chunks (id, embedding) ... ON CONFLICT (id) DO UPDATE` fails
with NotNullViolation on `doc_id` — even when the id provably exists, even
on a fresh table, even on a plain table with no vector type. Postgres
checks NOT NULL at tuple formation, BEFORE conflict detection. A
partial-column upsert against NOT NULL columns can never work; this is
not pgvector, not corruption (survived REINDEX + table recreation), not
the arbiter (plan was correct). Fix: metadata upserts first (full rows),
then `UPDATE ... FROM stage` for vectors. One-line summary: ON CONFLICT
needs a fully valid proposed row, not just a matching key.

## Design decisions and why

- **Chunk files as the resume unit** (1,000 rows/JSONL): atomic writes,
  Drive-friendly, re-runnable. The pipeline prints `todo=N` so resume
  state is visible, not silent.
- **Migration 001 is the schema source of truth**; `load.py` reads the
  file instead of embedding SQL. Schema reviewable without Python.
- **Doc label = max chunk weak label** (confidential > internal > public):
  one label per doc for the `documents` table; chunk-grain labels stay in
  Parquet until Phase 5 needs per-chunk filtering.
- **Image path on CPU** (`load_image_model`, text+image, ~440M params):
  batch-50 FUNSD run, 199.5s CPU, all 768-d/finite/unit-norm, indexed with
  `modality='image'`. Images need the Document prompt too: unprompted vs
  prompted image vectors sit at cos 0.89 — different regions, silently
  worse cross-modal retrieval without it. CPU avoids the MPS stall class.
- **FUNSD over RVL-CDIP**: RVL-CDIP is a single 38.7 GB tarball
  (`license: other`) — no subset without the full download. FUNSD test
  split is 50 forms, 4.4 MB, parquet-native. 199 total forms is far short
  of the roadmap's 2k–5k image target: recorded shortfall, right dev size.

## Known limits

- **MPS numerics verified but MPS bulk untrusted** (above). Long-batch
  behaviour on MPS is uncharacterized beyond "pathological".
- **Image subset tiny**: 50 FUNSD forms indexed; no image gold queries yet
  (Phase 5 needs them for F9). 199 total forms is far short of the
  roadmap's 2k–5k target — right dev size, recorded shortfall.
- **CUDA/MPS equivalence assumed from CPU**: MPS-vs-CPU verified
  (≥0.9999999); CUDA-vs-CPU not directly measured. Same fp32+eager path
  makes divergence unlikely, but unproven.
- **FUNSD licence is EPFL research-only** (non-commercial); PII-43k custom
  small-team licence; AESLC NC-assumed. All fit local learning; all need
  re-check before any hosted demo.

## Tips and tricks

- Time 1,000 docs before any full run (71.2s here → predicted ~28 min;
  actual Colab 726s CUDA). Extrapolation beats hope.
- When ON CONFLICT misbehaves, test the minimal case on a plain table
  first — isolates Postgres semantics from your schema in 30 seconds.
- `bt_index_check` (amcheck) + REINDEX rule out corruption fast; when both
  pass and the failure persists, question your mental model, not the data.
- Per-chunk Drive backup inside the stdout loop: disconnect insurance
  costs three lines.
- `torchvision` is required even for text-only loads (processor chain
  imports it unconditionally) — pin it with the stack, not when it breaks.

Spec pointers: §4, §6, F3, F4.
