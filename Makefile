# Trust Layer — one entry point for both stacks.
#
# Run from the repository root. The compose commands use relative paths, and this
# repository's path contains a space, which make cannot quote for you.
#
# Requires: uv, the .NET 10 SDK, and Docker with Compose v2.

COMPOSE := docker compose --project-directory . -f infra/docker-compose.yml

.PHONY: setup test lint up down prep labels

# Install both stacks' dependencies and the pre-commit hook.
# This installs the base Python env (no PySpark); `make test` and `make prep`
# add `--extra spark` themselves, so no separate sync step is needed.
setup:
	cd python && uv sync
	dotnet restore dotnet/TrustLayer.sln
	install -m 755 infra/hooks/pre-commit .git/hooks/pre-commit

# Run both test suites. Fails on the first stack that fails.
test:
	cd python && uv run --extra spark pytest
	dotnet test dotnet/TrustLayer.sln --nologo

# Check both stacks without modifying any file.
lint:
	cd python && uv run ruff check . ../data/scripts
	cd python && uv run ruff format --check . ../data/scripts
	dotnet format dotnet/TrustLayer.sln --verify-no-changes

# Start Postgres + pgvector and block until the health check reports healthy.
up:
	$(COMPOSE) up --detach --wait
	$(COMPOSE) ps

# Stop the database. The named volume survives, so data persists across down/up.
# To start from an empty database: docker compose --project-directory . -f infra/docker-compose.yml down --volumes
down:
	$(COMPOSE) down

# Phase 2 pipeline: raw AESLC Parquet → data/processed/corpus/ + stats JSON.
# Needs Java 17+ for Spark; run `uv sync --extra spark` first (once).
prep:
	cd python && uv run --extra spark python -m trustlayer.prep.run

# Phase 3 pipeline: corpus → weak labels + gold sample + split manifests.
# Needs Java 17+ for Spark. Gold review is a separate step afterwards.
labels:
	cd python && uv run --extra spark python -m trustlayer.labels.run --gold --splits
