# Trust Layer — one entry point for both stacks.
#
# Run from the repository root. The compose commands use relative paths, and this
# repository's path contains a space, which make cannot quote for you.
#
# Requires: uv, the .NET 10 SDK, and Docker with Compose v2.

COMPOSE := docker compose --project-directory . -f infra/docker-compose.yml

.PHONY: setup test lint up down prep labels embed embed-images seed-acls leak bench-freeze bench classify-data classify-lr classify-bert classify-llm decide classify-eval
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

# Phase 5: seeded ACLs + HNSW index. Needs DATABASE_URL.
seed-acls:
	docker compose --project-directory . -f infra/docker-compose.yml exec -T db psql -U trustlayer -d trustlayer -v ON_ERROR_STOP=1 -f - < infra/db/migrations/002_hnsw_index.sql
	cd python && UV_LINK_MODE=copy uv run --extra embed python -c "import os, psycopg, json; from trustlayer.retrieve.seed import seed_acls; conn = psycopg.connect(os.environ['DATABASE_URL']); print(json.dumps(seed_acls(conn))); conn.close()"

# Phase 5 leak test: every user × every dev query, zero violations.
# Needs DATABASE_URL; CPU encode of ~1.7k queries takes a few minutes.
leak:
	cd python && UV_LINK_MODE=copy uv run --extra embed python -m trustlayer.retrieve.leak

# Phase 6: freeze query sets, then run the benchmark sweep.
# Both need DATABASE_URL; bench needs the embed extra (CPU query encode).
bench-freeze:
	cd python && UV_LINK_MODE=copy uv run --extra embed python -m trustlayer.bench.freeze

bench:
	cd python && UV_LINK_MODE=copy uv run --extra embed python -m trustlayer.bench.harness

# Phase 7: weak-label train JSONL + gold eval JSONL (thread-level exclusion).
classify-data:
	cd python && UV_LINK_MODE=copy uv run --extra classify --extra spark python -m trustlayer.classify.data

# Phase 7 rows: LR on frozen Classification-prefix embeddings (needs vectors first).
classify-lr:
	cd python && UV_LINK_MODE=copy uv run --extra classify --extra embed --extra spark python -m trustlayer.classify.lr

# Phase 7 row: fine-tuned DistilBERT (GPU job; embed extra covers torch).
classify-bert:
	cd python && UV_LINK_MODE=copy uv run --extra embed python -m trustlayer.classify.bert

# Phase 7 row: Gemma zero-shot baseline (embed extra covers transformers).
classify-llm:
	cd python && UV_LINK_MODE=copy uv run --extra embed python -m trustlayer.classify.llm

# Phase 7 row: MediaPipe Decision Maker Laya (zero-shot; classify extra).
decide:
	cd python && UV_LINK_MODE=copy uv run --extra classify python -m trustlayer.decide.run

# Phase 7: score all rows, write comparison.json + reliability.json, print table.
classify-eval:
	cd python && UV_LINK_MODE=copy uv run python -m trustlayer.classify.eval
