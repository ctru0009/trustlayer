# Trust Layer — one entry point for both stacks.
#
# Run from the repository root. The compose commands use relative paths, and this
# repository's path contains a space, which make cannot quote for you.
#
# Requires: uv, the .NET 10 SDK, and Docker with Compose v2.

COMPOSE := docker compose --project-directory . -f infra/docker-compose.yml

.PHONY: setup test lint up down

# Install both stacks' dependencies.
# PySpark is deliberately not installed here; Phase 2 adds it with:
#   cd python && uv sync --extra spark
setup:
	cd python && uv sync
	dotnet restore dotnet/TrustLayer.sln

# Run both test suites. Fails on the first stack that fails.
test:
	cd python && uv run pytest
	dotnet test dotnet/TrustLayer.sln --nologo

# Check both stacks without modifying any file.
lint:
	cd python && uv run ruff check .
	dotnet format dotnet/TrustLayer.sln --verify-no-changes

# Start Postgres + pgvector and block until the health check reports healthy.
up:
	$(COMPOSE) up --detach --wait
	$(COMPOSE) ps

# Stop the database. The named volume survives, so data persists across down/up.
# To start from an empty database: docker compose --project-directory . -f infra/docker-compose.yml down --volumes
down:
	$(COMPOSE) down
