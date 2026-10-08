# Contributing

Small, focused PRs. One concern per PR, every changed line tracing to the
request. I review everything myself; a tight diff gets merged fast, a sprawling
one sits.

## Workflow

1. Fork and branch from `main`.
2. `make setup && make test` before you start. Both suites must pass before
   and after your change.
3. `make lint` before you push. Ruff and `dotnet format` are check-only gates
   in CI; the pre-commit hook (installed by `make setup`) runs them locally.
4. Open the PR against `main` with what changed and how you verified it. CI
   runs both stacks plus a fresh-clone smoke test.

## Rules that will get a PR sent back

- Touch only what the request requires. Do not reformat, refactor, or
  "improve" adjacent code.
- Match existing style, even if you would do it differently. Python: ruff,
  88 columns, `snake_case`. C#: `dotnet format`, nullable enabled, analyzers
  as errors.
- No speculative features or abstractions. No flexibility that was not asked
  for. If 200 lines could be 50, rewrite it.
- Every README number must trace to a run log or notebook. Do not add
  unmeasured claims.
- Never commit data. `data/` is git-ignored except download scripts, source
  records, and split manifests. Never include individual Enron messages in
  code, tests, fixtures, or screenshots. Test data is synthetic and hand-made.
- Test the behaviour, not the wiring. A test that passes without the change,
  restates the implementation, or asserts trivia is worse than none.

## What not to propose

These are explicit non-goals, not oversight: audio support, a frontend
framework, coverage tooling, Kubernetes or microservices sprawl, training an
embedding model from scratch. See `docs/spec.md` §2 and the roadmap cut list.

## Reporting issues

Use the issue templates. Bugs need reproduction steps and the observed vs
expected behaviour; numbers need the command that produced them. Security
issues do not go in the tracker. See [SECURITY.md](SECURITY.md).
