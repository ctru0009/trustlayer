"""Zero-shot sensitivity labels with MediaPipe Decision Maker.

Runs the Laya cross-encoder (``laya_s256.task``, float32) as a single 3-way
ChoiceQuestion over the frozen eval set, plus a float16 GLiNER NaN probe
(spec section 9 risk check — never a comparison row).

Usage::

    uv run --extra classify python -m trustlayer.decide.run
    uv run --extra classify python -m trustlayer.decide.run --probe

Probability semantics: every recorded probability comes from
``ChoiceResult.probabilities`` ONLY. ``ChoiceResult.confidence`` is a
margin/entropy-derived certainty score, not a class probability (a probe
showed 0.517 vs 0.00083 on the same row), so it is ignored everywhere and
ECE is computed from ``probabilities`` exclusively. Raw maps are
renormalized to sum to 1.0; any violation (missing keys, non-finite or
negative values, raw sum off by more than 1e-3) raises instead of writing.

Zero-shot: the Choice criteria below were fixed a priori from the weak-label
rule tiers and the spec ACL semantics, with no tuning on gold.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import sys
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
EVAL_DEFAULT = ROOT / "data" / "classify" / "eval.jsonl"
RESULTS_DEFAULT = ROOT / "data" / "classify" / "results" / "decide-laya.json"
PROBE_DEFAULT = ROOT / "data" / "classify" / "gliner-probe.json"
LAYA_DEFAULT = ROOT / "data" / "raw" / "decision" / "laya_s256.task"
GLINER_DEFAULT = ROOT / "data" / "raw" / "decision" / "gliner_s256.task"

LAYA_URL = (
    "https://storage.googleapis.com/mediapipe-models/decision_maker/laya/"
    "float32/laya_s256/latest/laya_s256.task"
)
GLINER_URL = (
    "https://storage.googleapis.com/mediapipe-models/decision_maker/gliner/"
    "float16/gliner_s256/latest/gliner_s256.task"
)

LABELS = ("public", "internal", "confidential")

INSTRUCTIONS = "Classify the email chunk by sensitivity: who may see it?"

# Fixed a priori from the weak-label rule tiers (rules.py) and spec section
# 8.1: public = no PII, no business content; internal = business content or
# low-severity PII; confidential = high-severity PII or explicit markers.
CRITERIA = {
    "public": (
        "Public information that is safe to share with anyone. Press releases, "
        "published announcements, and general messages with no business-sensitive "
        "content and no personal data."
    ),
    "internal": (
        "Internal business content meant for employees but not the public. "
        "Operations reports, budgets, forecasts, contracts, meeting notes, trading "
        "volumes, or routine work correspondence with no highly sensitive secrets."
    ),
    "confidential": (
        "Restricted content requiring explicit authorization to view. Government "
        "ID or account numbers, credentials, legal or attorney matters, or "
        "anything explicitly marked confidential, private, or do-not-forward."
    ),
}

NORMALIZE_PRIOR = True
MAX_NUM_TOKENS = 256
SEED = 42
RAW_SUM_TOL = 1e-3


def normalize_probs(raw: Mapping[str, float]) -> dict[str, float]:
    """Renormalize a Choice probability map to sum exactly 1.0.

    Fails loudly with ValueError on missing keys, non-finite or negative
    values, a zero total, or a raw total further than RAW_SUM_TOL from 1.0:
    a miscalibrated runtime must never be silently rescaled into a row.
    """
    missing = [label for label in LABELS if label not in raw]
    if missing:
        raise ValueError(f"probabilities missing keys: {missing}")
    probs = {label: float(raw[label]) for label in LABELS}
    for label, value in probs.items():
        if not math.isfinite(value):
            raise ValueError(f"non-finite probability for {label!r}: {value}")
        if value < 0.0:
            raise ValueError(f"negative probability for {label!r}: {value}")
    total = sum(probs.values())
    if total <= 0.0:
        raise ValueError("probabilities sum to zero (empty output)")
    if abs(total - 1.0) > RAW_SUM_TOL:
        raise ValueError(f"probabilities sum to {total}, expected ~1.0")
    if total != 1.0:
        probs = {label: value / total for label, value in probs.items()}
    if abs(sum(probs.values()) - 1.0) > 1e-9:
        raise ValueError("renormalized probabilities do not sum to 1.0")
    return probs


def _build_context(title: str, text: str) -> str:
    """Join subject + body into one decision context (fixed a priori format)."""
    return f"{title.strip()}\n{text.strip()}"


def _sha256(path: Path) -> str:
    """Streamed sha256 hex digest (assets are hundreds of MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    """Runtime versions via the metadata API (no heavy imports)."""
    try:
        mediapipe = importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        mediapipe = "missing"
    return {"python": platform.python_version(), "mediapipe": mediapipe}


def _load_eval_rows(path: Path) -> list[dict[str, Any]]:
    """Read eval.jsonl rows; tolerate gold/gold_label field aliases."""
    if not path.exists():
        raise ValueError(f"eval file {path} not found; run classify-data first")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold = row.get("gold", row.get("gold_label"))
            if gold not in LABELS:
                raise ValueError(f"{path}:{lineno}: bad gold label {gold!r}")
            rows.append(
                {
                    "doc_id": row["doc_id"],
                    "chunk_ord": row["chunk_ord"],
                    "title": row.get("title", ""),
                    "text": row["text"],
                    "gold": gold,
                }
            )
    return rows


def _build_question(decision_maker: Any) -> tuple[Any, dict[str, Any]]:
    """Build the 3-way sensitivity ChoiceQuestion; describe it for the config."""
    kwargs: dict[str, Any] = {
        "instructions": INSTRUCTIONS,
        "criteria": dict(CRITERIA),
        "normalize_prior": NORMALIZE_PRIOR,
    }
    question = decision_maker.ChoiceQuestion(**kwargs)
    try:
        params = {
            name: (
                repr(param.default)
                if param.default is not inspect.Parameter.empty
                else "required"
            )
            for name, param in inspect.signature(
                decision_maker.ChoiceQuestion
            ).parameters.items()
            if name != "self"
        }
    except (TypeError, ValueError):
        params = {"introspection": "unavailable"}
    meta: dict[str, Any] = {
        "instructions": INSTRUCTIONS,
        "criteria": dict(CRITERIA),
        "normalize_prior": NORMALIZE_PRIOR,
        "normalize_prior_why": (
            "Option specificity varies (public is broad, confidential is "
            "narrow); per the MediaPipe best-practices guide the null-context "
            "prior is normalized so the broad option cannot dominate."
        ),
        "scoring_mode": params.get("scoring_mode"),
        "scoring_mode_note": (
            "Installed-API default (passed implicitly); zero-decoding scoring "
            "normalizes per-option scores to probabilities in one pass."
        ),
        "temperature": params.get("temperature"),
        "temperature_note": (
            "Installed-API default (passed implicitly); zero-decoding scoring "
            "has no sampling step to temper."
        ),
        "probability_source": (
            "ChoiceResult.probabilities ONLY. ChoiceResult.confidence is a "
            "margin/entropy-derived certainty score, not a class probability, "
            "so ECE and every probability metric use probabilities exclusively."
        ),
        "api_params": params,
    }
    return question, meta


def _parse_result(result: Any) -> tuple[str, dict[str, float]]:
    """Extract (selected_key, normalized probs) from a ChoiceResult.

    Fails loudly unless selected_key is a known label AND equals the argmax
    of the recorded probabilities — the row must be self-consistent for ECE.
    """
    raw = result.probabilities
    if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
        raise ValueError("empty probabilities output")
    try:
        mapping = raw if isinstance(raw, dict) else dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unparseable probabilities payload: {raw!r}") from exc
    probs = normalize_probs(mapping)
    pred = result.selected_key
    if pred not in LABELS:
        raise ValueError(f"unknown selected_key {pred!r}")
    if pred != max(probs, key=probs.get):
        raise ValueError(f"selected_key {pred!r} disagrees with {probs}")
    return pred, probs


def _run_model(
    model_path: Path, contexts: list[str], *, strict: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open a maker, prewarm, evaluate each context; return (rows, info).

    Rows carry pred/probs/latency_ms plus an error field that is always None
    when strict is True (failures raise) and captures the message otherwise
    (GLiNER probe). Info carries model bytes, sha256, load/prewarm seconds,
    and the full question spec for the result config.
    """
    if not contexts:
        raise ValueError("no contexts to evaluate")
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.decision import decision_maker

    info: dict[str, Any] = {
        "model_bytes": model_path.stat().st_size,
        "sha256": _sha256(model_path),
    }
    t0 = time.perf_counter()
    options = decision_maker.DecisionMakerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
        max_num_tokens=MAX_NUM_TOKENS,
    )
    scored: list[dict[str, Any]] = []
    with decision_maker.DecisionMaker.create_from_options(options) as maker:
        info["load_seconds"] = round(time.perf_counter() - t0, 2)
        question, qmeta = _build_question(decision_maker)
        info.update(qmeta)
        t1 = time.perf_counter()
        maker.prewarm_choice(question)
        info["prewarm_seconds"] = round(time.perf_counter() - t1, 2)
        try:
            maker.evaluate_choice(contexts[0], question)
        except ValueError:
            if strict:
                raise
        for context in contexts:
            t2 = time.perf_counter()
            try:
                result = maker.evaluate_choice(context, question)
                latency_ms = (time.perf_counter() - t2) * 1000.0
                pred, probs = _parse_result(result)
                scored.append(
                    {
                        "pred": pred,
                        "probs": probs,
                        "latency_ms": latency_ms,
                        "error": None,
                    }
                )
            except ValueError as exc:
                if strict:
                    raise
                latency_ms = (time.perf_counter() - t2) * 1000.0
                scored.append(
                    {
                        "pred": None,
                        "probs": None,
                        "latency_ms": latency_ms,
                        "error": str(exc),
                    }
                )
    return scored, info


def _result_config(
    *,
    model_path: Path,
    model_url: str,
    info: dict[str, Any],
    n_rows: int,
    longest_context_chars: int,
) -> dict[str, Any]:
    """Assemble the Contract config block: bytes, load time, question spec."""
    return {
        "method": "decide-laya",
        "model": model_path.name,
        "model_url": model_url,
        "revision": info.pop("sha256"),
        "model_bytes": info.pop("model_bytes"),
        "load_seconds": info.pop("load_seconds"),
        "seed": SEED,
        "device": "cpu",
        "platform": f"{platform.system()} {platform.machine()}",
        "date": datetime.date.today().isoformat(),
        "versions": _versions(),
        "eval_rows": n_rows,
        "cost_per_1k_usd": 0.0,
        "cost_note": "local CPU inference; $0 per 1k docs plus walltime",
        "max_num_tokens": MAX_NUM_TOKENS,
        "truncation": (
            "none client-side (full title+text passed); the model truncates "
            f"past {MAX_NUM_TOKENS} tokens internally; longest eval context is "
            f"{longest_context_chars} chars (~{longest_context_chars // 4} "
            "tokens est.), so marginal overflow is possible on longest rows"
        ),
        **info,
    }


def main(argv: list[str] | None = None) -> int:
    """Score eval.jsonl with Laya (or --probe the fp16 GLiNER); exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT, help="eval input")
    parser.add_argument("--out", type=Path, default=RESULTS_DEFAULT, help="output")
    parser.add_argument("--model", type=Path, default=LAYA_DEFAULT, help=".task")
    parser.add_argument("--limit", type=int, default=None, help="first N rows only")
    parser.add_argument("--probe", action="store_true", help="run the GLiNER probe")
    args = parser.parse_args(argv)
    if args.probe:
        return probe(["--eval", str(args.eval)])
    rows = _load_eval_rows(args.eval)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no eval rows in {args.eval}")
    if args.limit is None and len(rows) != 200:
        print(f"warning: expected 200 eval rows, found {len(rows)}", file=sys.stderr)
    contexts = [_build_context(row["title"], row["text"]) for row in rows]
    longest = max(len(context) for context in contexts)
    scored, info = _run_model(args.model, contexts, strict=True)
    out_rows = [
        {
            "doc_id": row["doc_id"],
            "chunk_ord": row["chunk_ord"],
            "gold": row["gold"],
            "pred": res["pred"],
            "probs": res["probs"],
            "latency_ms": round(res["latency_ms"], 2),
        }
        for row, res in zip(rows, scored, strict=True)
    ]
    url = LAYA_URL if args.model.name == LAYA_DEFAULT.name else str(args.model)
    config = _result_config(
        model_path=args.model,
        model_url=url,
        info=info,
        n_rows=len(out_rows),
        longest_context_chars=longest,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "rows": out_rows}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    total_s = sum(row["latency_ms"] for row in out_rows) / 1000.0
    print(f"wrote {args.out} ({len(out_rows)} rows, {total_s:.1f}s eval)")
    return 0


def _ensure_asset(url: str, dest: Path) -> Path:
    """Download url to dest unless a non-empty file is there; return dest."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"using cached {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} ...")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed https host)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def _asset_summary(info: dict[str, Any]) -> dict[str, Any]:
    """Pick per-model file facts out of a _run_model info dict."""
    return {
        "revision": info["sha256"],
        "model_bytes": info["model_bytes"],
        "load_seconds": info["load_seconds"],
        "prewarm_seconds": info["prewarm_seconds"],
    }


def probe(argv: list[str] | None = None) -> int:
    """Score 20 rows with fp16 GLiNER vs Laya; write gliner-probe.json.

    Spec section 9 risk check: the listed float16 build may emit NaN or
    degraded outputs. This is NOT a comparison row — a load failure on macOS
    is recorded honestly in the probe file instead of raising.
    """
    parser = argparse.ArgumentParser(description=probe.__doc__)
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT, help="eval input")
    parser.add_argument("--out", type=Path, default=PROBE_DEFAULT, help="output")
    parser.add_argument("--laya-model", type=Path, default=LAYA_DEFAULT, help=".task")
    parser.add_argument("--gliner-model", type=Path, default=GLINER_DEFAULT, help="")
    parser.add_argument("--n", type=int, default=20, help="rows to probe")
    args = parser.parse_args(argv)
    rows = _load_eval_rows(args.eval)[: args.n]
    if not rows:
        raise ValueError(f"no eval rows in {args.eval}")
    _ensure_asset(GLINER_URL, args.gliner_model)
    contexts = [_build_context(row["title"], row["text"]) for row in rows]
    base_config: dict[str, Any] = {
        "probe": "gliner-f16-nan-check",
        "comparison_row": False,
        "note": (
            "Spec section 9 risk check on the listed float16 GLiNER build: "
            "NaN/empty-output screen plus selected_key agreement vs Laya. "
            "Never a comparison row; excluded from comparison.json."
        ),
        "rows_evaluated": len(rows),
        "row_selection": "first N rows of eval.jsonl (deterministic, a priori)",
        "seed": SEED,
        "device": "cpu",
        "platform": f"{platform.system()} {platform.machine()}",
        "date": datetime.date.today().isoformat(),
        "versions": _versions(),
        "max_num_tokens": MAX_NUM_TOKENS,
        "instructions": INSTRUCTIONS,
        "criteria": dict(CRITERIA),
        "normalize_prior": NORMALIZE_PRIOR,
    }
    laya_scored, laya_info = _run_model(args.laya_model, contexts, strict=True)
    try:
        gliner_scored, gliner_info = _run_model(
            args.gliner_model, contexts, strict=False
        )
    except Exception as exc:  # noqa: BLE001 (probe records load failures)
        base_config["laya"] = {"model": args.laya_model.name, "model_url": LAYA_URL}
        base_config["gliner"] = {
            "model": args.gliner_model.name,
            "model_url": GLINER_URL,
            "model_bytes": args.gliner_model.stat().st_size,
        }
        base_config["load_error"] = f"{type(exc).__name__}: {exc}"
        payload = {"config": base_config, "rows": [], "summary": {"n": 0}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out} (GLiNER failed to load: {exc})")
        return 0
    base_config["laya"] = {
        "model": args.laya_model.name,
        "model_url": LAYA_URL,
        **_asset_summary(laya_info),
    }
    base_config["gliner"] = {
        "model": args.gliner_model.name,
        "model_url": GLINER_URL,
        **_asset_summary(gliner_info),
    }
    probe_rows = []
    nan_rows = empty_rows = error_rows = agree = compared = 0
    for row, laya, gliner in zip(rows, laya_scored, gliner_scored, strict=True):
        if gliner["error"] is not None:
            error_rows += 1
            lowered = gliner["error"].lower()
            if "non-finite" in lowered:
                nan_rows += 1
            if "empty" in lowered or "sum to zero" in lowered:
                empty_rows += 1
            agreement = None
        else:
            compared += 1
            agreement = bool(gliner["pred"] == laya["pred"])
            agree += agreement
        probe_rows.append(
            {
                "doc_id": row["doc_id"],
                "chunk_ord": row["chunk_ord"],
                "laya_pred": laya["pred"],
                "laya_latency_ms": round(laya["latency_ms"], 2),
                "gliner_pred": gliner["pred"],
                "gliner_probs": gliner["probs"],
                "gliner_latency_ms": round(gliner["latency_ms"], 2),
                "gliner_error": gliner["error"],
                "agreement": agreement,
            }
        )
    summary = {
        "n": len(probe_rows),
        "gliner_errors": error_rows,
        "nan_rows": nan_rows,
        "empty_rows": empty_rows,
        "compared": compared,
        "agreement": agree,
        "agreement_rate": round(agree / compared, 4) if compared else None,
    }
    payload = {"config": base_config, "rows": probe_rows, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({summary['n']} rows, {error_rows} errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
