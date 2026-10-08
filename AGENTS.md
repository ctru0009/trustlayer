# Repository Guidelines

Trust Layer: a learning project building private on-device semantic search over
documents and screenshots — ranked by meaning, sensitivity-labelled, and
permission-aware (search must never leak restricted documents). Currently Phase 1
of 11: scaffolding only, no embedding/retrieval/classification/API code yet.
Author is learning ML/big data; the core question is how much is model vs
plumbing (access control, evaluation, data handling).

## Working rules (binding on agents)

From `docs/roadmap.md` — these override default agent instincts:

- **Core logic is human-written**: training loops, metrics, retrieval filter,
  Spark transforms. Agent reviews and suggests; agent MAY scaffold boilerplate.
- **Concept first, code second**: 3–5 `LEARNING.md` lines before and after each
  phase (human writes them; never fabricate).
- **Quiz ritual**: after each phase, 5 questions one at a time; fix gaps.
- **Every README number traces to a run log or notebook.**
- **NaN + dimension check before trusting any embedding run.**
- Never cut: leak test, frozen gold set, confidence intervals, limitations
  section, `LEARNING.md`. Never list unbuilt features as roadmap items.

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

- `python/src/trustlayer/` — Python package (currently `__init__.py` stub with
  `__version__` only; `prep`, `embed`, `classify`, `decide`, `bench`, `service`
  modules arrive in later phases).
- `dotnet/src/TrustLayer.Gateway/` — C# gateway (currently stock minimal-API
  `Program.cs`, `GET /` → "Hello World!"; real endpoints in Phase 8).
- `python/tests/`, `dotnet/tests/` — one placeholder suite each (see Testing).
- `data/` — git-ignored; only `SOURCES.md` (dataset registry) and `scripts/`
  (download scripts) are committed. Never commit raw data.
- `infra/` — `docker-compose.yml` (Postgres + pgvector), `db/migrations/`
  (empty until Phase 4), `dockerfiles/` (empty until Phase 8),
  `hooks/pre-commit` (installed by `make setup`).
- `docs/` — `spec.md` (authoritative design), `roadmap.md` (11 phases, exit
  criteria, no calendar dates), `architecture.md` (stub until Phase 8).
- `notebooks/` — thin Colab wrappers only (clone, install, run module, save to
  Drive per chunk). No logic.
- `demo/` — Gradio thin client (Phase 9). `demo-video/` — isolated fframes
  workspace, marked documentation in `.gitattributes` (Phase 10).

## Development Commands

Run from repo root (path contains a space; the Makefile's `COMPOSE` variable
handles quoting — don't `cd` into subdirs for make targets):

```bash
make setup   # uv sync + dotnet restore + install pre-commit hook
make test    # pytest, then dotnet test (fails on first failing stack)
make lint    # ruff check + ruff format --check + dotnet format verify (check-only)
make up      # start Postgres, block on healthcheck
make down    # stop it (volume survives; wipe with down --volumes)
```

Per-stack equivalents: `cd python && uv run pytest`, `cd python && uv run ruff
check .`, `dotnet test dotnet/TrustLayer.sln --nologo`. PySpark is NOT installed
by setup; Phase 2 uses `cd python && uv sync --extra spark` (needs Java 17+;
Homebrew's keg-only openjdk@21 requires an explicit `JAVA_HOME` — see
`docs/architecture.md`).

## Code Conventions & Common Patterns

No domain patterns exist yet (scaffolding phase). What governs future code:

- **Python**: ruff, `target-version = "py312"`, `line-length = 88`. Rules:
  `E F I UP B SIM C4 PTH TID Q S D C901`; ignore `D203 D213` (conflict-pair
  pin); `tests/**` exempt from `S101` (assert) and `D` (docstrings) only.
  `ruff format` is the formatter — no hand styling.
- **.NET**: `net10.0`, `Nullable` + `ImplicitUsings` enabled (no redundant
  usings; annotate nullability). `AnalysisMode=Recommended` with
  `TreatWarningsAsErrors` and `EnforceCodeStyleInBuild` — any warning,
  including style, fails `dotnet build`. `dotnet format` is the formatter.
- **Complexity < 40 both stacks**: ruff `max-complexity = 39`, CA1502 `39` in
  `dotnet/CodeMetricsConfig.txt` (wired as `AdditionalFile` — analyzers ignore
  `.editorconfig` for thresholds; both mean ≥ 40 fails).
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
- State: `README.md` (status + quickstart), `LEARNING.md` (human learning log),
  `data/SOURCES.md` (dataset registry).

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
- **.NET**: xUnit 2.9.3. `GatewaySmokeTests.cs` asserts the gateway assembly
  resolves and has an entry point — deliberately not an empty `[Fact]`; Phase 8
  replaces it with auth/ACL tests.
- Adding tests: Python files `python/tests/test_*.py`, plain `assert`,
  `-> None` annotations; .NET classes in `TrustLayer.Gateway.Tests`, `[Fact]`/
  `[Theory]`, must satisfy nullable + analyzers (except CA1707). Extend the
  placeholders; don't delete until real tests subsume their assertions.
- No coverage gate anywhere (coverlet referenced but uninvoked; no
  pytest-cov). Adding coverage tooling is new scope, not restoration.
- Pre-commit hook runs lint/format only (~5s budget) — tests are CI's job.
- Future gates (spec §12): leak test (every user × every eval query, both
  modes, zero violations), kill-and-resume embedding test, compose-based e2e on
  a tiny fixture.
