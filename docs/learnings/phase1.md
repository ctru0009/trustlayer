# Phase 1 — Foundations and repo setup: lesson

Retrospective: the phase built the repo skeleton, pinned toolchain, database,
CI, and (afterward) strict linting. The concepts, in the order they bite.

## Why pin versions?

Every pin answers "what changes under me if I rebuild in a year?"

- `python/uv.lock` + `uv sync --locked`: exact dependency tree; CI fails if the
  lock disagrees with `pyproject.toml` instead of drifting silently.
- `dotnet/global.json` (10.0.401, `latestPatch`): the SDK is the compiler;
  a different SDK can change analyzer behavior and build output.
- `pgvector/pgvector:0.8.7-pg18-trixie`: exact Postgres AND pgvector, because
  Phase 6 benchmark rows depend on pgvector behavior (HNSW under filtering,
  half-precision storage) that changes between releases. `latest` would make
  old numbers unreproducible — and worse, silently: same commands, different
  results, no error.
- CI pins the runners and actions too (`ubuntu-24.04`, `setup-uv@v10.2.0` with
  uv 0.10.9): the "fresh machine" must itself be a known machine.

Rule of thumb: pin anything whose silent change could invalidate a number you
already published.

## What does a health check give you?

`make up` blocks on `pg_isready` via `up --wait`. That is **liveness**: "can
this server accept connections". It says nothing about readiness: schema,
migrations, or the `vector` extension may all be missing on a healthy
container. The CI stack job closes that gap by asserting
`extversion = '0.8.7'` after `make up`.

Takeaway: a green health check means "the process is up", never "the system
works". Name what your check actually proves (the compose file comments do).

## What breaks on a different machine?

Things this phase hit or guarded against:

- **Java for PySpark**: Spark 4.x needs Java 17+; Homebrew's `openjdk@21` is
  keg-only, so `JAVA_HOME` must be set explicitly (see `architecture.md`).
- **Repo path with a space**: `make` cannot quote it, hence the `COMPOSE`
  variable with explicit `--project-directory`.
- **Pre-commit hook lives in `.git/hooks`** (installed by `make setup`), so it
  is not versioned — a fresh clone without `make setup` has no hook. CI is the
  backstop, and it runs the same commands.
- **Postgres 18 moved the data directory** (`/var/lib/postgresql/18/docker`);
  mounting the pre-18 path silently loses data on recreate. Volume mounts must
  match the image's layout, not last year's tutorial.
- **Environment vs code**: uv's hardlink warning on the external SSD is a
  filesystem property, not a project bug (`UV_LINK_MODE=copy`).

## Why warnings-as-errors (the strict-lint lesson)?

Advisory warnings are invisible to agents: nothing in the loop forces a fix.
`TreatWarningsAsErrors` + `EnforceCodeStyleInBuild` turn `dotnet build` into a
hard gate, and ruff gates run identically in `make lint`, the pre-commit hook,
and CI — one standard, three enforcers, fastest feedback first (~5s hook).

Curated strictness beats max strictness: `--select ALL` flagged pytest idioms
(`assert`) and `AnalysisMode=All` fought xUnit's own naming convention
(`CA1707`). The repo pins the conflicts it resolved (`D203`/`D213` ignore,
test-only `CA1707` suppression) so the choice is recorded, not re-litigated.

Complexity ≤ 40 on both stacks (ruff `max-complexity = 40`, CA1502 `40` via
`CodeMetricsConfig.txt` — analyzers ignore `.editorconfig` for thresholds; both
flag 41+). Both were proven with throwaway probes (raw 40 passes, raw 41
fails) before being trusted.

## Tips and tricks

Tricks used while building this phase (and worth reusing):

- **Prove gates with probes, not faith.** A lint rule you never saw fire might
  be misconfigured. A 40-branch throwaway method takes 30 seconds and settles
  "does ≥40 or >40 fail?" forever. Delete the probe after.
- **Measure strictness before adopting it.** `ruff check --select ALL` and
  `AnalysisMode=All` on the real codebase showed the noise (pytest `assert`,
  xUnit underscores) before any config was written — curated strictness won
  over max strictness on evidence, not taste.
- **One standard, three enforcers, fastest first.** Same ruff/dotnet-format
  commands in the ~5s pre-commit hook, `make lint`, and CI. The hook catches
  it now, CI catches it if you skipped the hook.
- **Comment the "why", not the "what".** Config comments here explain the
  non-obvious choice (`CodeMetricsConfig.txt` exists because analyzers ignore
  `.editorconfig` for thresholds; no `root = true` in the test overlay) so a
  future reader doesn't "simplify" it into breakage.
- **Assert the version, not just the presence.** CI checks
  `extversion = '0.8.7'` exactly — `CREATE EXTENSION IF NOT EXISTS` alone
  would pass on any pgvector and silently bless a drifted image.
- **Placeholders that actually test something.** Both placeholder tests assert
  installability/runnability (version agreement, assembly entry point),
  not `assert True` — a broken build config fails them.

## Pointers

- Pins: `python/uv.lock`, `dotnet/global.json`, `infra/docker-compose.yml`,
  `.github/workflows/ci.yml`
- Strictness: `python/pyproject.toml` (`[tool.ruff.*]`),
  `dotnet/Directory.Build.props`, `dotnet/CodeMetricsConfig.txt`,
  `.editorconfig`, `infra/hooks/pre-commit`
