"""Embedding model pins: id, revision, dims (no torch — import anywhere).

``model.py`` re-exports these so existing imports keep working; new code
that only needs the pins (health checks, shape asserts) imports from
here and stays light.
"""

from __future__ import annotations

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIM = 768
MAX_LENGTH = 512
