# Trust Layer

A private, local search experiment: I want to find documents and screenshots by meaning, have
sensitive items labelled, and be confident a search never returns something I am not allowed to see.

I am a software engineer (TypeScript, C#, Postgres) teaching myself ML and big data, and this is the
project I am learning it with. The question I am starting from: how much of "private semantic search"
is the model, and how much is the plumbing around it — access control, evaluation and data handling?

## Status

Phase 6 of 11. The pipeline runs end to end: PySpark prep → weak labels →
EmbeddingGemma-2 embeddings (23k text + 50 image vectors in pgvector) →
permission-filtered search with a zero-violation leak test (0/69,160) →
audited retrieval benchmark below. No classification, API, or demo code yet.

The plan, with exit criteria per phase, is in [`docs/roadmap.md`](docs/roadmap.md);
the design is in [`docs/spec.md`](docs/spec.md); per-phase lessons (what I actually
understood) are in [`docs/learnings/`](docs/learnings/phase1.md).

## Retrieval results (Phase 6)

300 frozen queries (150 subjects + 150 held-out chunks, test split) against
23,317 chunks. Recall@10 / MRR@10, bootstrap 95% CIs, CPU latencies. Full
provenance per run in `data/bench/results/`; method in
[`docs/learnings/phase6.md`](docs/learnings/phase6.md).

| Config | R@10 | MRR | p50 | Index |
|---|---|---|---|---|
| baseline (768d fp32, exact) | 0.977 | 0.911 | 31ms | 100MB |
| O1 512d | 0.977 | 0.911 | 29ms | 69MB |
| O2 256d | 0.973 | 0.895 | 5.5ms | 29MB |
| O3 128d | 0.940 | 0.860 | 4.4ms | 16MB |
| O5 HNSW | 0.977 | 0.911 | 34ms | 193MB |
| O6 halfvec | 0.977 | 0.911 | 7.5ms | 40MB |
| **O7 256d + halfvec (ship this)** | **0.970** | **0.894** | **4.5ms** | **16MB** |
| no-prefix ablation | 0.963 | 0.886 | 36ms | 100MB |

Cross-lingual (500-article Vietnamese Wikipedia sample): EN→VI MRR 0.954,
VI→VI MRR 0.985 — same recall, slightly worse cross-lingual ranking.

![quality vs index size](docs/bench-pareto.png)

Limits: single-relevant judgments (recall = hit@10); chunk queries favour
easy docs; 23k-row latency doesn't predict scale. O4 (GGUF) blocked:
released llama.cpp can't load this model's architecture yet.

## Getting started

```bash
make setup   # install Python and .NET dependencies
make test    # run both test suites
make up      # start Postgres + pgvector, wait until healthy
make down    # stop it
```

You need Docker, the .NET 10 SDK, and Java 17 or newer (Java is for PySpark in Phase 2; `uv` fetches
the pinned Python 3.12 itself). [`docs/architecture.md`](docs/architecture.md) has the details.

## Why two languages

Python for the data and model work, C# for the gateway that decides who may see what. I want the
permission logic to live somewhere a probability cannot reach it — the reasoning is in section 8 of
[`docs/spec.md`](docs/spec.md).

## Licence

MIT — see [`LICENSE`](LICENSE). Data sources and their licences are recorded in
[`data/SOURCES.md`](data/SOURCES.md); no data is committed to this repository.
