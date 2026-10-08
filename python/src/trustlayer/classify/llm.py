"""Gemma 3 1B IT zero-shot sensitivity scorer.

CODE ONLY — the checkpoint (``google/gemma-3-1b-it``) is gated, so this
module is written and reviewed without running it. To run later: accept
the model licence on Hugging Face, ``hf auth login`` (or set
``HF_TOKEN``), install the ``embed`` extra (transformers + torch), then
``python -m trustlayer.classify.llm``. Importing this module never
touches the model: transformers/torch import lazily inside functions.

Method: fixed prompt template embedding the three label descriptions,
greedy decode (``do_sample=False`` is exact temperature-0; seed 42 is
set for any incidental stochasticity), label = first valid label token
in the output, probs = softmax over the three labels' first-token
logprobs, renormalized (real probabilities for ECE). Cost is $0 local;
per-row latency and total walltime are recorded in the output.

Label semantics match ``trustlayer.decide.run.CRITERIA`` tier-for-tier
(public = no PII/business content; internal = business content or
low-severity PII; confidential = high-severity PII or explicit markers;
both drafted from ``trustlayer.labels.rules`` + spec section 8.1).
Wording differs on purpose: the Choice criteria are 2-sentence
descriptive option texts per the MediaPipe guide, while these concise
phrases suit a prompt template.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import random
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trustlayer.labels.rules import Label

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

__all__ = [
    "FALLBACK_LABEL",
    "LABELS",
    "LABEL_DESCRIPTIONS",
    "METHOD",
    "MODEL_ID",
    "SEED",
    "ScoreResult",
    "build_prompt",
    "label_token_ids",
    "load_model",
    "main",
    "normalize_logprobs",
    "parse_label",
    "score_row",
]

ROOT = Path(__file__).resolve().parents[4]
EVAL_DEFAULT = ROOT / "data" / "classify" / "eval.jsonl"
OUT_DEFAULT = ROOT / "data" / "classify" / "results" / "llm-gemma3-1b.json"

MODEL_ID = "google/gemma-3-1b-it"
METHOD = "llm-gemma3-1b"
SEED = 42
LABELS: tuple[str, ...] = tuple(label.value for label in Label)
MAX_NEW_TOKENS = 16
MAX_INPUT_TOKENS = 2048

LABEL_DESCRIPTIONS: dict[str, str] = {
    Label.PUBLIC.value: (
        "safe to share broadly: no personal data and no internal business content"
    ),
    Label.INTERNAL.value: (
        "for employees only: internal business content (meetings, budgets, "
        "reports, strategy) or low-severity personal data such as names and "
        "email addresses; not for public release"
    ),
    Label.CONFIDENTIAL.value: (
        "restricted: high-severity personal data (SSN, card numbers, "
        "passports, passwords or API keys, account numbers) or explicit "
        "confidentiality markers (confidential, privileged, attorney, "
        "do-not-forward); needs an explicit grant, never shared broadly"
    ),
}

# Unparseable-output fallback: the middle class avoids both the over-sharing
# bias of public and the alarm bias of confidential. Benchmark-only — access
# is enforced by deterministic code, never by this label (spec section 9).
# Every fallback is counted in main()'s summary line.
FALLBACK_LABEL = Label.INTERNAL.value

_LABEL_RE = re.compile(r"\b(" + "|".join(LABELS) + r")\b", re.IGNORECASE)


@dataclass(frozen=True)
class ScoreResult:
    """One scored row: prediction, label probs, latency, parse flag."""

    pred: str
    probs: dict[str, float]
    latency_ms: float
    parsed: bool


def build_prompt(title: str, text: str) -> str:
    """Render the fixed zero-shot user message (chat markup applied later).

    Title and body are concatenated, never ``str.format``-ed, so braces
    in email content cannot break the template.
    """
    lines = [
        "You are a data-sensitivity classifier. Reply with exactly one word.",
        "",
        "Labels:",
        *(f"- {label}: {LABEL_DESCRIPTIONS[label]}" for label in LABELS),
        "",
        "Subject: " + (title.strip() or "(none)"),
        "Body:",
        text.strip(),
        "",
        "Label:",
    ]
    return "\n".join(lines)


def parse_label(generated: str) -> str | None:
    """Extract the first valid label token (case-insensitive), if any."""
    match = _LABEL_RE.search(generated)
    return match.group(1).lower() if match else None


def normalize_logprobs(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Renormalize per-label first-token logprobs into probabilities.

    Terms with non-finite logprobs contribute zero mass; all-empty mass
    falls back to uniform so callers always get a valid distribution.
    """
    mass = {
        label: math.exp(logprobs[label]) if math.isfinite(logprobs[label]) else 0.0
        for label in LABELS
    }
    total = sum(mass.values())
    if total <= 0.0:
        return {label: 1.0 / len(LABELS) for label in LABELS}
    return {label: value / total for label, value in mass.items()}


def label_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int]]:
    """Collect first-token id candidates per label.

    The model may emit the label bare (``public``), with a leading space
    (`` public``), or capitalized — each surface form can start with a
    different token id, so all four are collected and the max logprob
    wins at scoring time.
    """
    ids: dict[str, list[int]] = {}
    for label in LABELS:
        variants = (label, " " + label, label.capitalize(), " " + label.capitalize())
        first: list[int] = []
        for variant in variants:
            toks: list[int] = tokenizer.encode(variant, add_special_tokens=False)
            if toks and toks[0] not in first:
                first.append(toks[0])
        ids[label] = first
    return ids


def load_model(
    device: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Load the gated Gemma 3 1B IT checkpoint in fp32 (lazy heavy imports).

    Args:
        device: ``auto`` (cuda > mps > cpu), ``cpu``, ``cuda`` or ``mps``.

    Raises:
        ImportError: If transformers/torch are missing (the ``embed`` extra).
        OSError: If the gated checkpoint rejects the download (licence/token).

    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        msg = (
            "trustlayer.classify.llm needs transformers+torch "
            "(the `embed` extra): uv sync --extra embed"
        )
        raise ImportError(msg) from exc
    random.seed(SEED)  # noqa: S311 (determinism seed, not crypto)
    torch.manual_seed(SEED)
    resolved = _resolve_device(device, torch)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map=None
    ).to(resolved)
    model.eval()
    return model, tokenizer, resolved


def score_row(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Mapping[str, list[int]],
    prompt: str,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> ScoreResult:
    """Score one prompt with greedy decoding.

    Returns the first valid label token in the decoded text (or
    ``FALLBACK_LABEL`` when unparseable, with ``parsed=False``), the
    renormalized first-token probs, and the generate-call latency.
    """
    import torch

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    t0 = time.monotonic()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.monotonic() - t0) * 1000.0
    new_ids = out.sequences[0][input_ids.shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    pred = parse_label(text)
    return ScoreResult(
        pred=pred if pred is not None else FALLBACK_LABEL,
        probs=_first_token_probs(out, token_ids),
        latency_ms=latency_ms,
        parsed=pred is not None,
    )


def main(argv: list[str] | None = None) -> int:
    """Score eval.jsonl with Gemma 3 1B IT and write llm-gemma3-1b.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--limit", type=int, default=None, help="score the first N rows only"
    )
    args = parser.parse_args(argv)
    if not args.eval.exists():
        print(f"eval file not found: {args.eval}", file=sys.stderr)
        return 1
    rows = _read_eval(args.eval)
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tokenizer, device = load_model(args.device)
    token_ids = label_token_ids(tokenizer)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    n_fallback = 0
    for row in rows:
        prompt = build_prompt(row["title"], row["text"])
        res = score_row(
            model, tokenizer, token_ids, prompt, device, args.max_new_tokens
        )
        results.append(
            {
                "doc_id": row["doc_id"],
                "chunk_ord": row["chunk_ord"],
                "gold": row["gold"],
                "pred": res.pred,
                "probs": res.probs,
                "latency_ms": round(res.latency_ms, 2),
            }
        )
        n_fallback += not res.parsed
    walltime_s = time.monotonic() - started
    payload = {
        "config": {
            "method": METHOD,
            "model": MODEL_ID,
            "revision": _resolve_revision(model),
            "seed": SEED,
            "device": device,
            "date": dt.date.today().isoformat(),
            "versions": _versions(),
            "cost_usd_per_1k": 0.0,
            "walltime_s": round(walltime_s, 1),
        },
        "rows": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {len(results)} rows -> {args.out} "
        f"({walltime_s:.1f}s wall, {n_fallback} fallbacks)"
    )
    return 0


def _read_eval(path: Path) -> list[dict[str, Any]]:
    """Read eval rows; expect gold.jsonl fields plus a split tag."""
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            gold = row.get("gold_label", row.get("gold"))
            if gold not in LABELS:
                msg = f"{path}:{lineno}: bad gold label {gold!r}"
                raise ValueError(msg)
            rows.append(
                {
                    "doc_id": str(row["doc_id"]),
                    "chunk_ord": int(row["chunk_ord"]),
                    "title": str(row.get("title") or ""),
                    "text": str(row["text"]),
                    "gold": gold,
                }
            )
    return rows


def _resolve_device(device: str, torch: Any) -> str:
    """Map ``auto`` to cuda > mps > cpu; pass explicit choices through."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _first_token_probs(
    out: Any, token_ids: Mapping[str, list[int]]
) -> dict[str, float]:
    """Derive label logprobs from the first generation step's logits."""
    if not out.scores:
        return {label: 1.0 / len(LABELS) for label in LABELS}
    import torch

    logprobs = torch.log_softmax(out.scores[0][0], dim=-1)
    best = {
        label: max((float(logprobs[i]) for i in ids), default=float("-inf"))
        for label, ids in token_ids.items()
    }
    return normalize_logprobs(best)


def _resolve_revision(model: Any) -> str:
    """Resolve the checkpoint sha from config metadata (offline-safe)."""
    sha = getattr(getattr(model, "config", None), "_commit_hash", None)
    return sha if isinstance(sha, str) and sha else "unknown"


def _versions() -> dict[str, str]:
    """Record library versions (transformers/torch imported lazily)."""
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


if __name__ == "__main__":
    sys.exit(main())
