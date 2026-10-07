# Architecture

Stub. The component responsibilities and the request flow are in sections 4 and 8 of
[`spec.md`](spec.md); this file will describe how the running pieces fit together once there is
something to describe (Phase 8). Right now it records the local preconditions and the pinned
toolchain, which is the part that exists.

## Pinned toolchain

| Piece | Pin | Where |
|---|---|---|
| Python | 3.12 | `python/pyproject.toml` (`requires-python`), `python/.python-version` |
| Python dependencies | locked | `python/uv.lock` |
| .NET | 10 (LTS) | `dotnet/global.json` |
| Postgres + pgvector | 18 + 0.8.7 | `infra/docker-compose.yml` |

## Local preconditions

- **Docker** with Compose v2 — for Postgres.
- **.NET 10 SDK** — for the gateway and its tests.
- **`uv`** — resolves and installs the pinned Python itself, so the system `python3` does not matter.
- **Java 17 or newer for PySpark** (Phase 2). PySpark 4.x requires it, and Spark 4.x is tested
  against Java 17 and 21. Homebrew's `openjdk@21` is keg-only, so PySpark will need an explicit
  `JAVA_HOME`:

  ```bash
  export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
  ```
