# Repository Guidelines

## Project Overview

Trust Layer: a learning project building private on-device semantic search over
documents and screenshots — ranked by meaning, sensitivity-labelled, and
permission-aware (search must never leak restricted documents). Currently Phase 8
of 11: offline pipelines (prep, labels, embed, retrieve, bench, classify) plus
a FastAPI model service and a C# gateway with JWT auth and ACL enforcement.
Author is learning ML/big data; the core question is how much is model vs
plumbing (access control, evaluation, data handling).

Working rules (binding, from `docs/roadmap.md`):

- **Agent implements, lesson records**: the agent builds; the per-phase lesson
  in `docs/learnings/` records decisions and rationale instead.
- **Lessons over rituals**: each phase gets `docs/learnings/phaseN.md`
  (concepts, tips and tricks, spec pointers), written alongside the work.
- **Every README number traces to a run log or notebook.**
- **NaN + dimension check before trusting any embedding run.**
- Never cut: leak test, frozen gold set, confidence intervals, limitations
  section, `docs/learnings/`. Never list unbuilt features as roadmap items.

## Architecture & Data Flow

```
Offline (Mac mini M4 16GB dev, Colab T4 for bulk):
  public datasets → PySpark (clean, dedupe, chunk) → Parquet
  → embedding job (EmbeddingGemma-2) → Postgres 18 + pgvector 0.8.7

Online:
  gateway (C# :8080) → permission-filtered SQL → pgvector
                     → /embed /classify /decide /answer → FastAPI service (:8000)
  (Gradio demo → gateway arrives in Phase 9)
```

Component responsibilities and must-nots (`docs/spec.md` §4):

| Component | Language | Owns | Must not |
|---|---|---|---|
| Data prep | Python (PySpark) | Cleaning, dedup, chunking, Parquet | Touch the DB directly |
| Embedding job | Python | Batch embeddings, resume, finite checks | Run in float16 |
| Model service | Python (FastAPI) | `/embed`, `/classify`, `/decide`, `/answer` | Know users/permissions |
| Gateway | C# (ASP.NET Core) | Auth, ACL, retrieval SQL, `/ask`, `/classify` | Run models |
| Database | Postgres + pgvector | Vectors, metadata, ACL columns | Be the sole permission check |
| Demo UI | Python (Gradio, Phase 9) | Thin client over gateway | Contain business rules |

Security model (`docs/spec.md` §8): synthetic users alice (HR), bob (Finance),
carol (Everyone), admin. ACL = `allowed_roles text[]` + `public|internal|
confidential` label; confidential needs an explicit grant, never
Everyone-inherited. Defence in depth: SQL pre-filter → app re-check (`Acl.Visible`,
drift throws) → LLM sees only passed passages. Missing-or-invisible → identical
404; errors must never reveal whether a restricted document exists.

Runtime flows: `/ask` fast = embed query → filtered search → ranked passages +
latency breakdown; answer mode adds one `/answer` call over passages truncated
to ≤1500 chars, citations align 1:1 with hits. `/classify` takes text XOR
`doc_id`. Score = 1 − cosine distance. Gateway rejects `dimension != 768`
(`chunks.embedding` is `vector(768)`; spec §6 requires query/doc dims match).

## Key Directories

- `python/src/trustlayer/` — `prep/` (Phase 2: clean/chunk pure-Python,
  dedupe/run/session/schema Spark), `labels/` (Phase 3: rules pure-Python,
  run/schema Spark), `embed/` (Phase 4: config pins, model loaders, store
  chunk-IO, load pgvector COPY, run CLI), `retrieve/` (Phase 5:
  acl/search/seed/leak), `bench/` (Phase 6: freeze/harness/xling/metrics),
  `classify/` (Phase 7: data/vectors/lr/bert/llm/eval/metrics/calibrate),
  `decide/` (Phase 7: MediaPipe Laya run + GLiNER probe), `service/`
  (Phase 8: FastAPI app, lazy holders, committed `lr-weights.json`, API
  leak test).
- `dotnet/src/TrustLayer.Gateway/` — `Program.cs` (minimal-API host: JWT,
  DI, request-id middleware, 5 routes), `Auth/` (Acl port, TokenService
  HS256 8h), `Models/` (records with snake_case `[JsonPropertyName]`),
  `Services/` (`IDocumentStore`/`IModelService` + Npgsql store + HttpClient).
- `python/tests/` — flat `test_<area>.py` (12 files); `dotnet/tests/` —
  `AclTests`, `GatewayContractTests` (WebApplicationFactory + fakes),
  `DocumentStoreTests` (live Postgres, see Testing), smoke test.
- `data/` — git-ignored EXCEPT `SOURCES.md`, `scripts/`, `gold/splits/*.txt`.
  `raw/` (downloads), `processed/` (corpus/labels/embeddings chunk-JSONL +
  `*-stats.json`), `bench/` (300 frozen queries + 11 result JSONs),
  `classify/` (train/eval JSONL, 5 row JSONs, comparison + reliability),
  `gold/` (`gold.jsonl` ignored, split manifests committed).
- `data/scripts/` — only committed code under `data/`: 5 pinned download
  scripts (`urllib`, no extras except viwiki needs `datasets`) + README.
- `infra/` — `docker-compose.yml` (`db` default; `stack` profile adds
  service + gateway), `dockerfiles/` (service + gateway), `db/migrations/`
  (`001` schema, `002` HNSW — NOT auto-applied), `hooks/pre-commit`.
- `docs/` — `spec.md` (authoritative design, cite by §), `roadmap.md`
  (11 phases, exit criteria, cut list), `architecture.md` (request flow,
  runtime map, pins), `learnings/phase1..11.md` (1–8 written, 9–11 stubs),
  `colab-phase7.md` (T4 runbook), 2 committed PNG figures.
- `notebooks/` — thin Colab wrappers only (clone, sync, run ONE module,
  per-chunk Drive backup). No logic. `demo/` — Phase 9 Gradio stub.
  `demo-video/` — Phase 10 isolated fframes workspace.

## Development Commands

Run from repo root (path contains a space; the Makefile's `COMPOSE` variable
handles quoting — don't `cd` into subdirs for make targets):

```bash
make setup   # uv sync + dotnet restore + install pre-commit hook
make test    # pytest (+spark extra), then dotnet test (fails on first stack)
make lint    # ruff check + ruff format --check + dotnet format verify (check-only)
make up      # start Postgres, block on healthcheck
make down    # stop it (volume survives; wipe with down --volumes)
make prep    # Phase 2: raw AESLC → data/processed/corpus/ + stats
make labels  # Phase 3: corpus → weak labels + gold sample + split manifests
make embed   # Phase 4: corpus → embeddings/ + pgvector (DATABASE_URL or --skip-load)
make embed-images  # Phase 4: FUNSD → embeddings-images/ + pgvector (CPU)
make seed-acls     # Phase 5: HNSW index + seeded ACLs (DATABASE_URL)
make leak          # Phase 5: every user × every dev query, zero violations
make bench-freeze  # Phase 6: freeze 300 queries (DATABASE_URL)
make bench         # Phase 6: run baseline + O-rows + ablations (DATABASE_URL)
make classify-data # Phase 7: train.jsonl + eval.jsonl (thread-excluded)
make classify-lr   # Phase 7: LR rows (needs embeddings-cls/ from Colab 7a)
make classify-bert # Phase 7: DistilBERT (local) — Colab 7b is faster
make classify-llm  # Phase 7: Gemma zero-shot (needs HF token + license)
make decide        # Phase 7: Laya row + GLiNER probe
make classify-eval # Phase 7: comparison table + reliability JSON
make serve         # Phase 8: model service locally (uvicorn :8000)
make stack         # Phase 8: db + service + gateway via compose (:8080)
make api-leak      # Phase 8: leak test through the API (stack + DATABASE_URL)
```

Per-stack equivalents: `cd python && uv run --extra spark pytest`,
`cd python && uv run ruff check . ../data/scripts`,
`dotnet test dotnet/TrustLayer.sln --nologo`. Make targets self-add
extras — no separate sync step. Gateway locally:
`cd dotnet/src/TrustLayer.Gateway && dotnet run --urls http://localhost:8080`
(`dotnet run` ignores `ASPNETCORE_URLS` when launchSettings.json exists).
Spark needs Java 17+ (Homebrew's keg-only openjdk@21 requires an explicit
`JAVA_HOME` — see `docs/architecture.md`).

## Code Conventions & Common Patterns

- **Pure-Python core, heavy shell**: text/ACL/metric logic has no
  torch/pyspark/mediapipe import — unit-testable without extras or Java
  (e.g. `prep/clean.py`, `retrieve/acl.py`, `bench/metrics.py`,
  `service/state.py::lr_predict`). Spark stages and model loaders are
  the shell around them.
- **Lazy heavy imports**: package `__init__` exposes Spark/model entry
  points via `__getattr__` (e.g. `prep/__init__.py`); torch/spark
  imported inside functions. `embed/config.py` holds pins with no torch
  so health checks stay light. (Caveat: `from package import name` binds
  a same-named SUBMODULE before `__getattr__` fires — import from the
  submodule directly.)
- **CLI shape**: sync `main(argv) -> int` + `sys.exit(main())`
  (e.g. `prep/run.py`, `bench/harness.py`). UDFs built inside functions
  (need live session); `useArrow=False`; explicit `DataType` instances.
- **Fail loud, never leak**: raise instead of degrading. Python:
  `_check_vector` → `HTTPException(500)` in `service/app.py`; `search.py`
  raises `PermissionError` on pre-filter/re-check drift. C#:
  `AclDriftException` → generic 500 in `DocumentStore.cs`.
- **HTTP mapping**: unavailable models → 503, wrong dim → 502, timeout →
  503 (`catch (HttpRequestException)` / `(TaskCanceledException)` in
  `Program.cs`). Gateway `HttpClient` timeout is 10 min (CPU generation).
- **.NET DI + async**: `AddSingleton(TokenService)`,
  `AddHttpClient<IModelService, ModelServiceClient>`,
  `AddSingleton<IDocumentStore>` in `Program.cs`; interfaces in
  `Services/Contracts.cs`. Minimal APIs are `async` with
  `ConfigureAwait(false)` + `CancellationToken ct` threaded everywhere;
  fresh `NpgsqlConnection` per call.
- **State**: stateless services + lazy holders (`LazyModels`:
  text_model/lr_weights/llm/laya, failures recorded in `errors` for
  `/health`). `LayaSession` holds one maker + question open (evaluate-only
  ~70ms; never reopen per request).
- **Observability**: `X-Request-Id` accepted/generated by the gateway,
  echoed back, propagated to the service (`AddRequestId`), C# log scope
  via `BeginScope` + source-generated `[LoggerMessage]` (`GatewayLog`);
  Python catch-all prints `request_id=… unhandled …`. Every model
  response carries `latency_ms` (`LatencyBreakdown`: embed/search/llm/total).
- **Wire format is snake_case** (spec §8.2): `top_k`, `doc_id`,
  `chunk_id`, `latency_ms` need `[JsonPropertyName]` — System.Text.Json
  binds case-insensitively but underscore-sensitively.
- **Python**: ruff, `target-version = "py312"`, `line-length = 88`. Rules:
  `E F I UP B SIM C4 PTH TID Q S D C901`; ignore `D203 D213`;
  `tests/**` exempt from `S101` (assert) and `D` only. `ruff format` is
  the formatter. `snake_case`, `from __future__ import annotations`,
  one-line docstrings on every function. Complexity ≤ 40 (41+ fails).
- **.NET**: `net10.0`, `Nullable` + `ImplicitUsings` (no redundant usings).
  `AnalysisMode=Recommended` with `TreatWarningsAsErrors` and
  `EnforceCodeStyleInBuild` — any warning, including style, fails
  `dotnet build`. `dotnet format` is the formatter. `PascalCase` dotted
  (`TrustLayer.Gateway`); `sealed record` DTOs; xUnit methods
  `Method_Scenario_Expected` (underscores exempt from CA1707 via
  test-only `.editorconfig` — which must NOT gain `root = true`).
  CA1502 ≤ 40 via `CodeMetricsConfig.txt` (`AdditionalFile`).
  `.editorconfig`: `[*.cs] max_line_length=120`.
- **Schema-drift checks**: pipelines assert written-Parquet schema against
  `*_SCHEMA` after write. **Stats JSON beside output**:
  `<out>-stats.json` with row counts + seconds; every lesson number
  traces to one.
- **Data rules**: `data/` git-ignored — commit download scripts and split
  manifests only. Every dataset gets a row in `data/SOURCES.md` (exact
  version, licence, check date) BEFORE use. Enron is real people's email:
  script download only, never commit raw data, never show individual
  messages in demos/screenshots/fixtures. Dedupe and split BY EMAIL
  THREAD before train/eval.
- **Model rules**: NEVER float16 (silent NaN — the #1 silent killer); fp32
  default; per-batch NaN/Inf + dimension checks; task prefixes
  `SearchQuery`/`Document`/`Classification` (quality drops silently
  without them — never encode unprompted); L2-renormalise after any
  truncation (768/512/256/128); queries and docs same dimension. Train
  and serve text formatting must be byte-identical
  (`f"{title.strip() or 'none'}\n{text}"`).
- **Spec voice**: first person, curious, specific; never oversell; audio is
  a stretch — MUST NOT appear in README until built and measured.

## Important Files

- Entry points: `dotnet/src/TrustLayer.Gateway/Program.cs` (gateway host;
  `partial class Program` for `WebApplicationFactory`),
  `python/src/trustlayer/service/app.py` (`trustlayer.service.app:app`),
  pipeline CLIs as `python -m trustlayer.<prep/labels/embed.run|
  retrieve.leak|bench.freeze|bench.harness|bench.xling|classify.*|
  decide.run|service.api_leak>`.
- Configs: `Makefile`, `python/pyproject.toml` (pins + ruff/pytest),
  `python/uv.lock`, `python/.python-version`, `dotnet/global.json` (SDK
  single source of truth), `dotnet/TrustLayer.sln`, `dotnet/**/*.csproj`,
  `dotnet/Directory.Build.props`, `dotnet/CodeMetricsConfig.txt`,
  `.editorconfig` (+ test overlay), `.github/workflows/ci.yml`,
  `infra/docker-compose.yml`, `infra/dockerfiles/*.Dockerfile`,
  `infra/db/migrations/001_*.sql` + `002_*.sql`, `infra/hooks/pre-commit`,
  `.env.example`, `.gitignore`, `.gitattributes`.
- Design: `docs/spec.md` (authoritative), `docs/roadmap.md` (phases + exit
  criteria), `docs/architecture.md` (request flow, runtime map, pins).
- State: `README.md` (status + results + quickstart), `data/SOURCES.md`
  (dataset registry), `docs/colab-phase7.md` (T4 runbook).

## Runtime/Tooling Preferences

Pinned toolchain — never bump without recording the decision:

| Piece | Pin | Where |
|---|---|---|
| Python | 3.12 | `.python-version`, `requires-python >=3.12,<3.13` |
| Python deps | locked | `uv.lock` (CI uses `uv sync --locked`) |
| .NET SDK | 10.0.401 (`latestPatch`) | `dotnet/global.json` |
| .NET target | net10.0 | both `.csproj` files |
| Postgres + pgvector | 18 + 0.8.7 | `infra/docker-compose.yml` (CI asserts extversion) |
| uv (CI + image) | 0.10.9 | `ci.yml`, `service.Dockerfile` |
| Java (Spark) | Temurin 21 (min 17+) | `ci.yml` |
| pytest / ruff | 9.1.1 / 0.16.10 | `[dependency-groups] dev` |
| Python extras | pyspark 4.1.3; numpy/pillow/psycopg/st/s-torch/torchvision/transformers; mediapipe/sklearn; fastapi/httpx/uvicorn | `[optional-dependencies]` |
| NuGet | JwtBearer 10.0.12, Npgsql 10.0.3, Mvc.Testing 10.0.12, Test SDK 17.14.1, xUnit 2.9.3 | `dotnet/**/*.csproj` |

- Package managers: `uv` (Python), NuGet via `dotnet` CLI (.NET). No pip/npm.
  No JS tooling anywhere (no `package.json`).
- Runners: ubuntu-24.04 in CI (3 jobs: `python`, `dotnet`, `stack`
  fresh-clone smoke incl. pgvector `0.8.7` version assert). No compose e2e
  in CI (torch image cost — local `make stack` + `make api-leak` only).
- Dev machine: macOS arm64, repo on external SSD. If uv warns about hardlinks
  falling back to copy, that's environmental (cross-filesystem); suppress with
  `UV_LINK_MODE=copy`, not a code issue.
- Heavy jobs run one at a time on the 16GB Mac; Docker images: service
  (`python:3.12-slim-trixie` + torch CPU, `modelcache` volume for weights),
  gateway (SDK build → `aspnet:10.0` + curl healthcheck).
- Migrations are raw SQL, manually ordered, re-runnable (`IF NOT EXISTS`);
  explicitly NOT auto-applied by compose.

## Testing & QA

- **Python**: pytest 9.1.1, `testpaths = ["tests"]`, no markers/conftest.
  Flat `test_<area>.py`, functions `test_<behavior>_<expectation>`
  (`-> None`, plain `assert`). `tmp_path` scratch dirs; synthetic in-test
  data only (never real email content); `test_package.py` asserts install
  correctness (module version == dist metadata; resolves to
  `src/trustlayer`) — deliberately not `assert True`.
- **.NET**: xUnit 2.9.3. `AclTests` ([Theory] ACL matrix),
  `GatewayContractTests` (`IClassFixture<WebApplicationFactory<Program>>`
  with `FakeStore`/`FakeModels` fakes + `LoginAsync`/`Auth` helpers),
  `GatewaySmokeTests` (assembly/entry-point), `DocumentStoreTests`
  (live-Postgres). Methods `Method_Scenario_Expected`; must satisfy
  nullable + analyzers (except CA1707). DB gating via custom
  `RequiresTestDbFactAttribute` (xUnit 2.9 has NO `Assert.Skip` — the
  attribute sets `Skip` at discovery when `TRUSTLAYER_TEST_DB` is unset).
- No mocking library on either stack — hand-rolled fakes + real fixtures.
  Service tests stub models via module patching (`MODELS`, `_encode`) and
  `TestClient` — no torch/mediapipe needed.
- Env-gated skips: .NET `DocumentStoreTests` (needs `TRUSTLAYER_TEST_DB`);
  Python `test_retrieve` DB tests (needs `DATABASE_URL` + `psycopg`);
  `pytest.importorskip("pyspark")` for pipeline tests (+ Java 17+);
  `test_service` skips when git-ignored Phase 7 data is absent.
- Anti-vacuous rule: leak tests assert FULL expected hit counts, not just
  zero violations — a pass with no results would hide an over-restrictive
  predicate. Direct: `make leak` (0/69,160); API: `make api-leak` (0/840,
  both modes, plus citation-drift + no-answer-in-fast checks).
- No coverage gate anywhere (coverlet referenced but uninvoked; no
  pytest-cov). Adding coverage tooling is new scope, not restoration.
- Pre-commit hook (`infra/hooks/pre-commit`, installed by `make setup`;
  plain sh, NOT pre-commit-framework) runs lint/format only (~10s budget)
  — tests are CI's job.
