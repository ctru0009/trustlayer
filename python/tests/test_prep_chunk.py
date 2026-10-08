"""Unit tests for trustlayer.prep.chunk (pure Python, no Spark)."""

import pytest

from trustlayer.prep.chunk import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("hello world") == ["hello world"]


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_word_boundary_snap() -> None:
    assert chunk_text("one two three four", size=8, overlap=0) == [
        "one two",
        "three",
        "four",
    ]


def test_overlap_covers_long_text() -> None:
    text = " ".join(f"token{i:04d}" for i in range(400))
    chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    assert len(chunks) > 1
    assert chunks[0] == text[: len(chunks[0])]
    assert chunks[-1] == text[-len(chunks[-1]) :]
    assert sum(map(len, chunks)) > len(text)


def test_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", size=0)
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=11)


def test_default_size_is_documented_choice() -> None:
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (1000, 100)
