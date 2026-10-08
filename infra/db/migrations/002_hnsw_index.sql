-- 002: HNSW index for cosine search (spec section 8.4, Phase 5).
--
-- m=16, ef_construction=64 are pgvector's recommended starting values.
-- Filtered HNSW (WHERE + ORDER BY <=> LIMIT) can return fewer than LIMIT
-- rows: the graph walk visits ef_search candidates, and the filter applies
-- after the walk. In pgvector 0.8.7 the fix is hnsw.iterative_scan =
-- 'strict_order' (enum: off | relaxed_order | strict_order, default off),
-- or a higher hnsw.ef_search. Probed in the Phase 5 lesson: no shortfall
-- at 23k rows even under a 2%-selective filter.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
