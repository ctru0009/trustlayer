"""Resume semantics test: completed chunks are skipped, no duplicates."""

from pathlib import Path

from trustlayer.embed.store import (
    ChunkResult,
    chunk_path,
    done_indices,
    read_chunk_file,
    write_chunk_file,
)

VEC = tuple([0.2] * 768)


def test_resume_skips_done_chunks_without_duplicates(tmp_path: Path) -> None:
    out = tmp_path / "embeddings"
    # Simulate an interrupted run: chunks 0-1 written, chunk 2 missing.
    for index in range(2):
        rows = [
            ChunkResult(f"doc{i}", 0, VEC) for i in range(index * 10, index * 10 + 10)
        ]
        write_chunk_file(chunk_path(out, index), rows)
    assert done_indices(out) == {0, 1}

    # Resume writes only the missing chunk.
    n_chunks = 3
    todo = [i for i in range(n_chunks) if i not in done_indices(out)]
    assert todo == [2]
    write_chunk_file(
        chunk_path(out, 2),
        [ChunkResult(f"doc{i}", 0, VEC) for i in range(20, 30)],
    )

    seen = []
    for index in range(n_chunks):
        seen.extend(read_chunk_file(chunk_path(out, index)))
    ids = [(r.doc_id, r.chunk_ord) for r in seen]
    assert len(ids) == 30
    assert len(set(ids)) == 30
