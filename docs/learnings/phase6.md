# Phase 6 — Retrieval benchmark: lesson

An auditable quality/memory/speed table on 300 frozen queries
(sha `65390815`): 150 informative email subjects + 150 held-out chunk
texts from the test split, relevance = source document.

## What was measured

9 configs on the same 300 queries, exact protocol per spec §10
(warm-up + 5 repeats, bootstrap CIs, config logged per run):

| Config | R@10 | MRR | nDCG | p50 | Size |
|---|---|---|---|---|---|
| baseline (768d fp32 exact) | 0.977 | 0.911 | 0.927 | 31ms | 100MB |
| O1 512d | 0.977 | 0.911 | 0.927 | 29ms | 69MB |
| O2 256d | 0.973 | 0.895 | 0.914 | 5.5ms | 29MB |
| O3 128d | 0.940 | 0.860 | 0.880 | 4.4ms | 16MB |
| O5 HNSW | 0.977 | 0.911 | 0.927 | 34ms | 193MB |
| O6 halfvec | 0.977 | 0.911 | 0.928 | 7.5ms | 40MB |
| O7 256d+halfvec | 0.970 | 0.894 | 0.913 | 4.5ms | 16MB |
| ablate no-renorm | 0.973 | 0.895 | 0.914 | 5.5ms | 29MB |
| ablate no-prefix | 0.963 | 0.886 | 0.905 | 36ms | 100MB |

Cross-lingual (500-article VI Wikipedia sample): EN→VI R 1.0 / MRR
0.954 (30 hand-written queries); VI→VI R 1.0 / MRR 0.985 (100 title
queries). Same recall, slightly worse cross-lingual ranking.

## Findings

- **Truncation is nearly free**: 512d identical, 256d −0.003,
  128d −0.037 recall. EmbeddingGemma's Matryoshka-style training shows.
- **Halfvec is quality-neutral** (MRR +0.0005, noise) at 60% smaller
  and 4× faster. The single best optimization.
- **HNSW loses at 23k rows**: same recall (strict_order), but slower
  than exact (34 vs 31ms) and 2× the size. Index overhead dominates;
  HNSW pays off at scale, not here. Honest row, kept.
- **No-prefix costs −0.014 recall** — real but smaller than feared.
- **No-renorm is vacuous for pgvector cosine**: identical to O2 down
  to the 4th decimal. The `<=>` operator normalizes internally, so
  un-renormalized vectors score the same. Renorm matters for
- **O7 (256d + halfvec)**: −0.007 recall for 6× smaller, 7× faster.
  Dominates O3 at the same size (+0.03 recall). What I would ship.
- **Hand-check 100/100**: all reviewed queries are answerable by their
  source doc; the weakest (chunk-003, a truncated "entertain" fragment)
  stays in as a robustness case — post-hoc exclusion would bias results.

## O4 deferred (blocked, with evidence)

The GGUF exists (`ggml-org/embeddinggemma-2-GGUF`, Q8_0 310MB,
downloaded) but llama.cpp 0.6.0 — the latest release — fails with
`unknown model architecture: 'gemma-embedding2'`. The model is too
new for released llama.cpp. Retry when a release supports the arch;
the file waits in `data/raw/gguf/` (git-ignored).

## VI→EN not measured (recorded limit)

EN→VI works (hand-written queries, human-judged relevance). The
reverse needs a parallel EN corpus on the same 500 topics; translating
500 articles is out of scope. The gap measured (MRR 0.954 vs 0.985)
is EN→VI vs VI→VI, not symmetric.

## Methodology notes

- **Doc-grain dedupe**: first baseline run reported nDCG 1.25 —
  impossible, caused by chunk-grain rankings repeating the same doc
  (DCG summed duplicates, IDCG counted one). Fix: fetch 50 chunks,
  first-occurrence dedupe to 10 docs. A metric > 1.0 is always a bug.
- **Ceiling effects**: chunk queries share substrings with the corpus
  (recall near 1.0 by construction); the subject set is the harder
  complement. Both reported together, limits stated.
- **Small-corpus latency**: 4–36ms at 23k rows does not predict scale;
  the HNSW/exact crossover is unmeasured.
- **Single-relevant judgments**: recall@10 = hit@10 here. False
  negatives (other relevant docs unjudged) bias absolute numbers up;
  relative comparisons across rows remain valid.

## Tips and tricks

- `np.fromstring(text, sep=',')` parses pgvector text output fast —
  no per-element Python loop for 23k × 768 floats.
- `uv run --with matplotlib` for one-off plots: no dependency to pin.
- Download scripts: `Path(__file__).parents[2]` from `data/scripts/`,
  not `[3]` — off-by-one wrote 600MB outside the repo (caught, moved).

Spec pointers: §10, F8.
