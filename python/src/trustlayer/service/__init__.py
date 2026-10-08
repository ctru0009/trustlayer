"""Phase 8 model service: FastAPI over embed/classify/decide/answer.

Run (from ``python/``)::

    uv run --extra service --extra embed uvicorn trustlayer.service.app:app

Add ``--extra classify`` for ``/decide`` (mediapipe). The service knows
nothing about users or permissions — the C# gateway enforces ACLs and
passes permitted passages only.

Layout: ``state`` (lazy model holders + pure LR inference, no FastAPI),
``app`` (routes; import the FastAPI instance from there, not here),
``lr-weights.json`` (committed LR artifact from the Phase 7 training path
via ``lr.py --export-weights``).
"""

from __future__ import annotations

from trustlayer.service.state import (
    DIM,
    LABELS,
    LazyModels,
    load_lr_weights,
    lr_predict,
    softmax,
)

__all__ = [
    "DIM",
    "LABELS",
    "LazyModels",
    "load_lr_weights",
    "lr_predict",
    "softmax",
]
