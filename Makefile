# Trust Layer — one entry point for both stacks.
#
# Run from the repository root. The compose commands use relative paths, and this
# repository's path contains a space, which make cannot quote for you.
#
# Requires: uv, the .NET 10 SDK, and Docker with Compose v2.

COMPOSE := docker compose --project-directory . -f infra/docker-compose.yml

.PHONY: setup test lint up down prep labels embed embed-images

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

# Phase 4 pipeline: corpus → embeddings + pgvector load. Needs the embed
# extra (torch) and DATABASE_URL unless --skip-load is passed through.
embed:
	cd python && UV_LINK_MODE=copy uv run --extra embed --extra spark python -m trustlayer.embed.run

# Phase 4 image path: FUNSD forms → embeddings-images/ → pgvector.
# CPU default (50 forms, ~3 min); needs DATABASE_URL unless --skip-load.
embed-images:
	cd python && UV_LINK_MODE=copy uv run --extra embed --extra spark python -m trustlayer.embed.run --images ../data/raw/funsd/test-00000-of-00001.parquet --device cpu --batch-size 8
