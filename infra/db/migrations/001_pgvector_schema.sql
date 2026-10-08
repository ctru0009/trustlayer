-- 001: pgvector extension + documents/chunks tables (spec section 8.5).
--
-- Applied by trustlayer.embed on first load (ensure_schema runs this same
-- SQL). Kept versioned here so the schema is reviewable and replayable
-- without running Python. Re-runnable: IF NOT EXISTS throughout.
--
-- documents.label is the max chunk weak label from Phase 3
-- (confidential > internal > public). allowed_roles is Phase 5's ACL
-- column, defaulting to "visible to nobody" until seeded.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'aeslc',
    title TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT 'text',
    lang TEXT NOT NULL DEFAULT 'en',
    thread_id TEXT NOT NULL,
    label TEXT NOT NULL,
    allowed_roles TEXT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents (id),
    ord INT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(768)
);
