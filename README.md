# Trust Layer

A private, local search experiment: I want to find documents and screenshots by meaning, have
sensitive items labelled, and be confident a search never returns something I am not allowed to see.

I am a software engineer (TypeScript, C#, Postgres) teaching myself ML and big data, and this is the
project I am learning it with. The question I am starting from: how much of "private semantic search"
is the model, and how much is the plumbing around it — access control, evaluation and data handling?

## Status

Phase 1 of 11. The repository builds, both test suites run, and Postgres with pgvector starts.

There is no embedding, retrieval, classification or API code yet. I am adding those phase by phase
and writing down what I actually understand in [`LEARNING.md`](LEARNING.md). The plan, with exit
criteria per phase, is in [`docs/roadmap.md`](docs/roadmap.md); the design is in
[`docs/spec.md`](docs/spec.md).

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
