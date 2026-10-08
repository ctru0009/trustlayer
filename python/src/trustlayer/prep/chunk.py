"""Character chunking with word-boundary snapping. Pure Python, no pyspark."""

from __future__ import annotations

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

_MIN_SNAP_BACK = 200


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping windows, snapping ends to word boundaries.

    Args:
        text: Cleaned document text.
        size: Target window length in characters.
        overlap: Characters carried from the end of one chunk into the next,
            so sentences spanning a boundary stay readable in one chunk.

    Returns:
        Stripped, non-empty chunks. Short texts yield a single chunk.

    """
    if size <= 0 or overlap < 0 or overlap >= size:
        msg = f"need 0 <= overlap < size, got size={size} overlap={overlap}"
        raise ValueError(msg)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = _snap_end(text, start, end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end <= start:
            break
        start = end - overlap if end < len(text) else end
    return chunks


def _snap_end(text: str, start: int, end: int) -> int:
    """Move end back to the last whitespace, if one is reasonably close."""
    floor = max(start + (end - start - _MIN_SNAP_BACK), start + 1)
    snapped = text.rfind(" ", floor, end)
    return snapped if snapped > start else end
