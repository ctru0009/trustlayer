"""Phase 8 model service: FastAPI over embed/classify/decide/answer.

Spec section 8.3: the model service owns model loading and prefixes and
knows NOTHING about users or permissions (that is the gateway's job).
Every endpoint takes raw content in and returns model outputs; no ACL,
no user fields anywhere in this module.

Run: ``uv run --extra service --extra embed uvicorn
trustlayer.service.app:app`` (add ``--extra classify`` for /decide).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from trustlayer.service.state import DIM, LABELS, LazyModels, lr_predict

ROOT = Path(__file__).resolve().parents[4]
LAYA_DEFAULT = ROOT / "data" / "raw" / "decision" / "laya_s256.task"

MODELS = LazyModels()

app = FastAPI(title="trustlayer-model-service", version="0.1.0")


class EmbedRequest(BaseModel):
    """Texts to encode; kind selects the task prompt."""

    texts: list[str] = Field(min_length=1, max_length=64)
    kind: str = Field(default="query", pattern="^(query|document|classification)$")
    dim: int = Field(default=768, ge=1, le=768)


class ClassifyRequest(BaseModel):
    """Raw text to classify (service encodes + runs the LR head)."""

    title: str = ""
    text: str = Field(min_length=1)


class DecideRequest(BaseModel):
    """Raw text for a zero-shot sensitivity decision (Laya Choice)."""

    title: str = ""
    text: str = Field(min_length=1)


class AnswerRequest(BaseModel):
    """Question + permitted passages only (gateway enforces the permit)."""

    query: str = Field(min_length=1)
    passages: list[str] = Field(min_length=1, max_length=10)


def _check_vector(vec: list[float], dim: int) -> None:
    """Fail loudly on wrong dim or non-finite values (spec section 6)."""
    if len(vec) != dim:
        raise HTTPException(500, f"expected dim {dim}, got {len(vec)}")
    if not all(math.isfinite(x) for x in vec):
        raise HTTPException(500, "non-finite value in embedding output")


def _encode(texts: list[str], kind: str) -> list[list[float]]:
    """Encode with the task-correct prompt (never unprompted)."""
    from trustlayer.embed.model import MAX_LENGTH

    model = MODELS.text_encoder()
    kwargs: dict[str, Any] = {
        "batch_size": 32,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if kind == "query":
        vectors = model.encode_query(texts, **kwargs)
    elif kind == "document":
        kwargs["processing_kwargs"] = {
            "text": {"max_length": MAX_LENGTH, "truncation": True}
        }
        vectors = model.encode_document(texts, **kwargs)
    else:
        kwargs["prompt_name"] = "Classification"
        kwargs["processing_kwargs"] = {
            "text": {"max_length": MAX_LENGTH, "truncation": True}
        }
        vectors = model.encode(texts, **kwargs)
    return [row.astype("float32").tolist() for row in vectors]


def _truncate(vec: list[float], dim: int) -> list[float]:
    """Truncate + L2-renormalize (spec section 6: slicing breaks unit length)."""
    if dim == DIM:
        return vec
    head = vec[:dim]
    norm = math.sqrt(sum(x * x for x in head))
    if norm == 0.0:
        raise HTTPException(500, "zero-norm vector after truncation")
    return [x / norm for x in head]


@app.exception_handler(Exception)
async def _request_id_errors(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: request id in the log line, no internals in the body."""
    if isinstance(exc, HTTPException):
        raise exc
    request_id = request.headers.get("x-request-id", "-")
    print(f"request_id={request_id} unhandled {type(exc).__name__}: {exc}")
    return JSONResponse({"detail": "internal error"}, status_code=500)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + per-model readiness (models load lazily on first use)."""
    from trustlayer.embed.config import MODEL_ID, REVISION

    return {
        "status": "ok",
        "model": MODEL_ID,
        "revision": REVISION,
        "embed_loaded": MODELS.text_model is not None,
        "embed_device": MODELS.text_device or None,
        "classify_loaded": MODELS.lr_weights is not None,
        "llm_loaded": MODELS.llm is not None,
        "laya_asset_present": LAYA_DEFAULT.exists(),
        "errors": dict(MODELS.errors),
    }


@app.post("/embed")
def embed(body: EmbedRequest) -> dict[str, Any]:
    """Encode texts with the task prompt; normalized vectors out."""
    t0 = time.monotonic()
    try:
        vectors = _encode(body.texts, body.kind)
    except Exception as exc:  # noqa: BLE001 — model load/encode failure
        raise HTTPException(503, f"embed unavailable: {exc}") from exc
    out = []
    for vec in vectors:
        _check_vector(vec, DIM)
        out.append(_truncate(vec, body.dim))
    total_ms = (time.monotonic() - t0) * 1000.0
    return {"vectors": out, "dim": body.dim, "latency_ms": total_ms}


@app.post("/classify")
def classify(body: ClassifyRequest) -> dict[str, Any]:
    """Classify raw text: Classification-prefix encode + LR head (ship row)."""
    t0 = time.monotonic()
    try:
        weights = MODELS.classifier()
    except Exception as exc:  # noqa: BLE001 — weights missing/corrupt
        raise HTTPException(503, f"classify unavailable: {exc}") from exc
    try:
        text = f"{body.title.strip() or 'none'}\n{body.text}"
        vectors = _encode([text], "classification")
    except Exception as exc:  # noqa: BLE001 — model load/encode failure
        raise HTTPException(503, f"embed unavailable: {exc}") from exc
    _check_vector(vectors[0], DIM)
    probs = lr_predict(weights, vectors[0])
    pred = max(LABELS, key=lambda name: probs[name])
    return {
        "label": pred,
        "probs": probs,
        "method": "lr-768cls",
        "latency_ms": (time.monotonic() - t0) * 1000.0,
    }


@app.post("/decide")
def decide(body: DecideRequest) -> dict[str, Any]:
    """Zero-shot sensitivity decision via Laya (Choice, strict parse)."""
    from trustlayer.decide.run import CRITERIA, INSTRUCTIONS, _build_context
    from trustlayer.decide.run import LABELS as DECIDE_LABELS

    if not LAYA_DEFAULT.exists():
        raise HTTPException(503, f"laya asset missing: {LAYA_DEFAULT}")
    try:
        session = MODELS.decide_session(LAYA_DEFAULT)
    except Exception as exc:  # noqa: BLE001 — mediapipe load failure
        raise HTTPException(503, f"decide unavailable: {exc}") from exc
    t0 = time.monotonic()
    try:
        context = _build_context(body.title, body.text)
        pred, probs = session.evaluate(context)
    except Exception as exc:  # noqa: BLE001 — evaluate/parse failure
        raise HTTPException(503, f"decide failed: {exc}") from exc
    return {
        "label": pred,
        "probs": probs,
        "method": "decide-laya",
        "labels": list(DECIDE_LABELS),
        "instructions": INSTRUCTIONS,
        "criteria": dict(CRITERIA),
        "latency_ms": (time.monotonic() - t0) * 1000.0,
    }


@app.post("/answer")
def answer(body: AnswerRequest) -> dict[str, Any]:
    """Answer a question grounded ONLY in the provided passages (RAG).

    The gateway passes permitted passages only — this endpoint cannot see
    the database, users, or ACLs. Greedy decoding, cited-passage format.
    """
    try:
        model, tokenizer, _token_ids, device = MODELS.answer_llm()
    except Exception as exc:  # noqa: BLE001 — gated weights/tokenizer
        raise HTTPException(503, f"answer unavailable: {exc}") from exc
    numbered = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(body.passages))
    prompt = (
        "Answer the question using ONLY the passages below. "
        "Cite every claim as [n]. If the passages do not answer it, say so.\n\n"
        f"Question: {body.query}\n\nPassages:\n{numbered}\n\nAnswer:"
    )
    t0 = time.monotonic()
    try:
        import torch

        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0][input_ids.shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    except Exception as exc:  # noqa: BLE001 — generation failure
        raise HTTPException(503, f"answer generation failed: {exc}") from exc
    return {"answer": text, "latency_ms": (time.monotonic() - t0) * 1000.0}
