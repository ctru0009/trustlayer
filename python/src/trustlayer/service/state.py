"""Phase 8 model service: lazy model holders for the FastAPI app.

Each holder loads once on first use (service startup stays fast and
`/health` can report per-model readiness). Pure holder logic — no
FastAPI import, so unit tests can exercise the pure parts without the
``service`` extra.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

WEIGHTS_PATH = Path(__file__).resolve().parent / "lr-weights.json"
LABELS = ("public", "internal", "confidential")
DIM = 768


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax (pure Python, no numpy)."""
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def lr_predict(weights: dict[str, Any], vector: list[float]) -> dict[str, float]:
    """Multinomial LR inference from exported coef/intercept (no sklearn).

    Matches ``lr.py`` main(): argmax over LABELS order after renormalizing
    (softmax already sums to 1; the division is a no-op kept for parity).
    """
    if len(vector) != DIM:
        msg = f"expected dim {DIM}, got {len(vector)}"
        raise ValueError(msg)
    coef = weights["coef"]
    intercept = weights["intercept"]
    labels = list(weights["labels"])
    logits = [
        sum(c * v for c, v in zip(row, vector, strict=True)) + b
        for row, b in zip(coef, intercept, strict=True)
    ]
    probs = dict(zip(labels, softmax(logits), strict=True))
    total = sum(probs.values())
    return {name: probs[name] / total for name in LABELS}


def load_lr_weights(path: Path = WEIGHTS_PATH) -> dict[str, Any]:
    """Load + shape-check the committed LR weights artifact."""
    weights = json.loads(path.read_text())
    if list(weights["labels"]) != list(LABELS):
        msg = f"unexpected labels {weights['labels']!r}"
        raise ValueError(msg)
    coef = weights["coef"]
    if len(coef) != len(LABELS) or any(len(row) != DIM for row in coef):
        msg = f"expected coef {(len(LABELS), DIM)}"
        raise ValueError(msg)
    if len(weights["intercept"]) != len(LABELS):
        msg = "intercept/labels length mismatch"
        raise ValueError(msg)
    return weights


class LazyModels:
    """Holds lazily-loaded models; records load failures for /health."""

    def __init__(self) -> None:
        """Create empty holders (nothing loads until first use)."""
        self.text_model: Any = None
        self.text_device: str = ""
        self.lr_weights: dict[str, Any] | None = None
        self.llm: tuple[Any, Any, Any, str] | None = None
        self.laya: Any = None
        self.errors: dict[str, str] = {}

    def text_encoder(self, device: str = "cpu") -> Any:
        """Load the text-only embedding encoder once (fp32, pinned rev)."""
        if self.text_model is None:
            try:
                from trustlayer.embed.model import load_text_model

                self.text_model = load_text_model(device)
                self.text_device = device
            except Exception as exc:  # noqa: BLE001 — surfaced via /health
                self.errors["embed"] = f"{type(exc).__name__}: {exc}"
                raise
        return self.text_model

    def classifier(self) -> dict[str, Any]:
        """Load the committed LR weights once (no sklearn needed)."""
        if self.lr_weights is None:
            try:
                self.lr_weights = load_lr_weights()
            except Exception as exc:  # noqa: BLE001 — surfaced via /health
                self.errors["classify"] = f"{type(exc).__name__}: {exc}"
                raise
        return self.lr_weights

    def answer_llm(self, device: str = "auto") -> tuple[Any, Any, Any, str]:
        """Load Gemma 3 1B IT + label token ids once (fp32, greedy)."""
        if self.llm is None:
            try:
                from trustlayer.classify import llm as llm_mod

                model, tokenizer, resolved = llm_mod.load_model(device)
                token_ids = llm_mod.label_token_ids(tokenizer)
                self.llm = (model, tokenizer, token_ids, resolved)
            except Exception as exc:  # noqa: BLE001 — surfaced via /health
                self.errors["answer"] = f"{type(exc).__name__}: {exc}"
                raise
        return self.llm

    def decide_session(self, model_path: Path) -> Any:
        """Open the Laya session once; later calls reuse it (evaluate-only)."""
        if self.laya is None:
            try:
                from trustlayer.decide.run import LayaSession

                session = LayaSession(model_path)
                self.laya = session.__enter__()
            except Exception as exc:  # noqa: BLE001 — surfaced via /health
                self.errors["decide"] = f"{type(exc).__name__}: {exc}"
                raise
        return self.laya
