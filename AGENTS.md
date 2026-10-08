# Repository Guidelines

## Project Overview

Trust Layer: a learning project building private on-device semantic search over
documents and screenshots — ranked by meaning, sensitivity-labelled, and
permission-aware (search must never leak restricted documents). Currently Phase 8
of 11: offline pipelines (prep, labels, embed, retrieve, bench, classify) plus
a FastAPI model service and a C# gateway with JWT auth and ACL enforcement;
pgvector holds 23k text + 50 image vectors with seeded ACLs and an HNSW index.
README carries the retrieval table (O7 ships), the classifier comparison (LR
ships), and the API section (0/840 leak test through the API). No demo yet.
Author is learning ML/big data; the core question is how much is model vs
plumbing (access control, evaluation, data handling).

## Working rules (binding on agents)

From `docs/roadmap.md` — these override default agent instincts:

- **Agent implements, lesson records**: you overrode human-writes-core-logic
  ("No human write") — the agent builds, and the per-phase lesson in
  `docs/learnings/` records decisions and rationale instead.
- **Lessons over rituals**: each phase gets a lesson in `docs/learnings/phaseN.md`
  (concepts, tips and tricks, spec pointers), written alongside the work — no
  before/after notes, no quizzes gating progress.
- **Every README number traces to a run log or notebook.**
- **NaN + dimension check before trusting any embedding run.**
- Never cut: leak test, frozen gold set, confidence intervals, limitations
  section, `docs/learnings/`. Never list unbuilt features as roadmap items.

## Architecture & Data Flow

```
Offline (Mac mini M4 16GB dev, Colab for bulk):
  public datasets → PySpark (clean, dedupe, chunk) → Parquet
  → embedding job (EmbeddingGemma-2) → Postgres 18 + pgvector 0.8.7

Online:
  Gradio demo → C# gateway → permission-filtered SQL → pgvector
                            → /embed /classify /decide → Python FastAPI
                            → LLM (answer mode only, permitted passages only)
```

Component responsibilities and must-nots (`docs/spec.md` §4):

| Component | Language | Owns | Must not |
|---|---|---|---|
| Data prep | Python (PySpark) | Cleaning, dedup, chunking, Parquet | Touch the DB directly |
| Embedding job | Python | Batch embeddings, resume, finite checks | Run in float16 |
| Model service | Python (FastAPI) | `/embed`, `/classify`, `/decide` | Know users/permissions |
| Gateway | C# (ASP.NET Core) | Auth, ACL, retrieval SQL, `/ask`, `/classify` | Run models |
| Database | Postgres + pgvector | Vectors, metadata, ACL columns | Be the sole permission check |
| Demo UI | Python (Gradio) | Thin client over gateway | Contain business rules |

Security model (`docs/spec.md` §8): synthetic users alice (HR), bob (Finance),
carol (Everyone), admin. ACL = `allowed_roles text[]` + `public|internal|
confidential` label; confidential needs an explicit grant, never
Everyone-inherited. Defence in depth: SQL pre-filter → app re-check → LLM sees
only passed passages. Errors must never reveal whether a restricted document
exists.

## Key Directories

- `python/src/trustlayer/` — `prep/` (Phase 2), `labels/` (Phase 3), `embed/`
  (Phase 4), `retrieve/` (Phase 5: acl/search/seed/leak), `bench/` (Phase 6),
  `classify/` + `decide/` (Phase 7), `service/` (Phase 8: FastAPI app, lazy
  holders, committed `lr-weights.json`, API leak test).
- `dotnet/src/TrustLayer.Gateway/` — C# gateway (Phase 8: `Auth/`, `Models/`,
  `Services/` + minimal-API `Program.cs`; JWT demo auth, ACL port, retrieval
  SQL port, `/ask` fast+answer, `/classify`, `/documents/{id}`).
- `python/tests/` — unit + Spark e2e + service contract tests;
  `dotnet/tests/` — xUnit ACL/auth/contract tests + DB-gated integration
  (`RequiresTestDbFact`, needs `TRUSTLAYER_TEST_DB`).
- `data/` — git-ignored; `SOURCES.md` (dataset registry), `scripts/` (download
  scripts), and `gold/splits/*.txt` (doc_id manifests, content hashes) are
  committed. Never commit raw data or gold content (embeds chunk text).
- `infra/` — `docker-compose.yml` (Postgres + pgvector), `db/migrations/`
  (empty until Phase 4), `dockerfiles/` (empty until Phase 8),
  `hooks/pre-commit` (installed by `make setup`).
- `docs/` — `spec.md` (authoritative design), `roadmap.md` (11 phases, exit
  criteria, no calendar dates), `architecture.md` (stub until Phase 8),
  `learnings/phase1..11.md` (per-phase lessons: concepts, tips, spec pointers).
- `notebooks/` — thin Colab wrappers only (clone, install, run module, save to
  Drive per chunk). No logic.
- `demo/` — Gradio thin client (Phase 9). `demo-video/` — isolated fframes
  workspace, marked documentation in `.gitattributes` (Phase 10).

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
`--extra spark` — no separate sync step. Spark needs Java 17+ (Homebrew's
keg-only openjdk@21 requires an explicit `JAVA_HOME` — see
`docs/architecture.md`).

## Code Conventions & Common Patterns

Established patterns (prep/labels set the template for future phases):

- **Pure-Python core, Spark shell**: text logic (`clean`, `chunk`, `rules`)
  has no pyspark import — unit-testable without Java; Spark stages
  (`dedupe`, `run`, `session`, `schema`) need the `spark` extra.
- **Lazy spark imports**: package `__init__` exposes Spark entry points via
  `__getattr__` so `import trustlayer.<phase>` works without pyspark (CI
  syncs base env for lint; tests self-add the extra).
- **UDF rules**: build inside functions (needs live session, never module
  scope); `useArrow=False` (pandas/Arrow not installed, skips the probe
  warning); explicit `DataType` instances for return types.
- **Schema-drift checks**: every pipeline asserts written-Parquet schema
  against its `*_SCHEMA` contract after write.
- **Stats JSON beside output**: `<out>-stats.json` with row counts + seconds;
  every lesson number traces to one.

What governs future code:

- **Python**: ruff, `target-version = "py312"`, `line-length = 88`. Rules:
  `E F I UP B SIM C4 PTH TID Q S D C901`; ignore `D203 D213` (conflict-pair
  pin); `tests/**` exempt from `S101` (assert) and `D` (docstrings) only.
  `ruff format` is the formatter — no hand styling.
- **.NET**: `net10.0`, `Nullable` + `ImplicitUsings` enabled (no redundant
  usings; annotate nullability). `AnalysisMode=Recommended` with
  `TreatWarningsAsErrors` and `EnforceCodeStyleInBuild` — any warning,
  including style, fails `dotnet build`. `dotnet format` is the formatter.
- **Complexity ≤ 40 both stacks**: ruff `max-complexity = 40`, CA1502 `40` in
  `dotnet/CodeMetricsConfig.txt` (wired as `AdditionalFile` — analyzers ignore
  `.editorconfig` for thresholds; both flag 41+).
- **Naming**: Python `snake_case` package `trustlayer`; .NET `PascalCase`
  dotted (`TrustLayer.Gateway`); xUnit methods `Method_Scenario_Expected`
  (underscores exempt from CA1707 via test-only `.editorconfig` — which must
  NOT gain `root = true`, or it drops parent rules).
- **Data rules**: `data/` git-ignored — commit download scripts and handmade
  fixtures only. Every dataset gets a row in `data/SOURCES.md` (exact version,
  licence, check date) BEFORE use. Enron is real people's email: script
  download only, never commit raw data, never show individual messages in
  demos/screenshots/fixtures. Dedupe and split BY EMAIL THREAD before train/eval.
- **Model rules** (for later phases): NEVER float16 (silent NaN — the #1
  silent killer); fp32 default; per-batch NaN/Inf + dimension checks; task
  prefixes `SearchQuery`/`Document`/`Classification`; L2-renormalise after any
  truncation (768/512/256/128); queries and docs same dimension.
- **Spec voice**: first person, curious, specific; never oversell; audio is a
  stretch — MUST NOT appear in README until built and measured.

## Important Files

- Entry points: `python/src/trustlayer/__init__.py`,
  `dotnet/src/TrustLayer.Gateway/Program.cs` (top-level statements).
- Configs: `Makefile`, `python/pyproject.toml`, `python/uv.lock`,
  `python/.python-version`, `dotnet/global.json` (SDK single source of truth),
  `dotnet/TrustLayer.sln`, `dotnet/Directory.Build.props`,
  `dotnet/CodeMetricsConfig.txt`, `.editorconfig` (+ test overlay),
  `.github/workflows/ci.yml`, `infra/docker-compose.yml`, `.env.example`.
- Design: `docs/spec.md` (authoritative), `docs/roadmap.md` (phases + exit
  criteria), `docs/architecture.md` (toolchain pins + preconditions).
- State: `README.md` (status + quickstart), `data/SOURCES.md` (dataset registry).

## Runtime/Tooling Preferences

Pinned toolchain — never bump without recording the decision:

| Piece | Pin | Where |
|---|---|---|
| Python | 3.12 | `.python-version`, `requires-python >=3.12,<3.13` |
| Python deps | locked | `uv.lock` (CI uses `uv sync --locked`) |
| .NET SDK | 10.0.401 (`latestPatch`) | `dotnet/global.json` |
| Postgres + pgvector | 18 + 0.8.7 | `infra/docker-compose.yml` |
| uv (CI) | 0.10.9 | `ci.yml` via `setup-uv@v10.2.0` |
| pytest / ruff | 9.1.1 / 0.16.10 | `[dependency-groups] dev` |
| xUnit / Test SDK / coverlet | 2.9.3 / 17.14.1 / 6.0.4 | test `.csproj` |

- Package managers: `uv` (Python), NuGet via `dotnet` CLI (.NET). No pip/npm.
- Runners: ubuntu-24.04 in CI (3 jobs: `python`, `dotnet`, `stack`
  fresh-clone smoke incl. pgvector `0.8.7` version assert).
- Dev machine: macOS arm64, repo on external SSD. If uv warns about hardlinks
  falling back to copy, that's environmental (cross-filesystem); suppress with
  `UV_LINK_MODE=copy`, not a code issue.
- Heavy jobs run one at a time on the 16GB Mac; CI is the backstop for anyone
  who skips `make setup` (pre-commit hook lives in git-ignored `.git/hooks`).

## Testing & QA

- **Python**: pytest 9.1.1, `testpaths = ["tests"]`, no markers/conftest.
  `python/tests/test_package.py` asserts install correctness (module version ==
  dist metadata; resolves to `src/trustlayer`) — deliberately not `assert True`.
- **.NET**: xUnit 2.9.3. `AclTests` (truth table), `GatewayContractTests`
  (WebApplicationFactory + fakes), `DocumentStoreTests` (live Postgres,
  gated on `TRUSTLAYER_TEST_DB` via `RequiresTestDbFact` — v2 has no
  `Assert.Skip`). The Phase 1 smoke test stays (assembly/entry-point).
- Adding tests: Python files `python/tests/test_*.py`, plain `assert`,
  `-> None` annotations; .NET classes in `TrustLayer.Gateway.Tests`, `[Fact]`/
  `[Theory]`, must satisfy nullable + analyzers (except CA1707).
- No coverage gate anywhere (coverlet referenced but uninvoked; no
  pytest-cov). Adding coverage tooling is new scope, not restoration.
- Pre-commit hook runs lint/format only (~5s budget) — tests are CI's job.
- Gates (spec §12): direct leak test (`make leak`, 0/69,160), API leak
  test (`make api-leak`, 0/840, both modes), kill-and-resume embedding
  test (Phase 4). No compose e2e in CI (torch image cost — local only).
