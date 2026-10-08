-- 002: HNSW index for cosine search (spec section 8.4, Phase 5).
--
-- m=16, ef_construction=64 are pgvector's recommended starting values.
-- Filtered HNSW (WHERE + ORDER BY <=> LIMIT) can return fewer than LIMIT
-- rows: the graph walk visits ef_search candidates, and the filter applies
-- 'strict_order' (enum: off | relaxed_order | strict_order, default off),
-- or a higher hnsw.ef_search. Phase 5 lesson: 1 shortfall in 6,916 leak
-- queries (carol/q553, 7/10) — strict_order ships in `search()` itself.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
