"""Phase 7 DistilBERT row: fine-tune on weak labels, score the gold set.

Needs the ``embed`` extra (torch + transformers); run with::

    UV_LINK_MODE=copy uv run --extra embed \
        python -m trustlayer.classify.bert --mode all

Pipeline: smoke (50 steps on a subset; refuses to continue unless the loss
is finite and decreasing) → full train (3 fixed epochs, model saved to
``data/classify/distilbert/``) → predict ``eval.jsonl`` (softmax probs +
per-row latency) → ``data/classify/results/distilbert.json``.

Setup: ``distilbert-base-uncased`` at pinned ``REVISION`` (HEAD resolved
2026-10-08; passed to every ``from_pretrained`` call and recorded in the
output config), input = ``title + text`` truncated to 512 tokens, lr 2e-5,
batch 16 (micro-batch 4 x 4 gradient accumulation), seed 42, fp32.

Imbalance (~3.5% confidential): class-weighted CrossEntropyLoss. Every weak
row is seen once per epoch with minority gradients up-weighted, instead of
a balanced sampler that would repeat the same confidential rows ~28x per
epoch and overfit to duplicates.

Fixed epochs are deliberate: a dev-weak F1 would measure rule-mimicry (how
well the model copies the weak labeller), not gold quality, and stopping on
the 200-row gold set would tune on the comparison metric. So no dev split,
no early stopping.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import date
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification

ROOT = Path(__file__).resolve().parents[4]
DATA_DEFAULT = ROOT / "data" / "classify"
MODEL_ID = "distilbert/distilbert-base-uncased"
MODEL_ALIAS = "distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
LABELS = ("public", "internal", "confidential")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
SEED = 42
MAX_TOKENS = 512
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 16
MICRO_BATCH = 4
SMOKE_STEPS = 50
SMOKE_ROWS = 2048


def build_input(title: str, text: str) -> str:
    """Join title + body; the tokenizer truncates to MAX_TOKENS."""
    title = (title or "").strip()
    text = (text or "").strip()
    return f"{title}\n{text}" if title else text


def set_seed(seed: int = SEED) -> None:
    """Seed python/torch RNGs (call before model init for a fixed head)."""
    random.seed(seed)  # noqa: S311 (training shuffle, not crypto)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """Map auto|cuda|mps|cpu to a device (auto prefers cuda, then mps)."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def class_weights(label_ids: list[int]) -> list[float]:
    """Inverse-frequency weights N/(K*n_c); an empty class gets 0."""
    total = len(label_ids)
    counts = [label_ids.count(i) for i in range(len(LABELS))]
    return [total / (len(LABELS) * c) if c else 0.0 for c in counts]


class WeakDataset(Dataset):
    """Weak-labelled rows; tokenized on the fly, padded in collate."""

    def __init__(self, rows: list[dict], tokenizer: AutoTokenizer) -> None:
        """Store inputs + label ids (missing title tolerated)."""
        self._inputs = [build_input(r.get("title", ""), r["text"]) for r in rows]
        self._labels = [LABEL_TO_ID[r["weak_label"]] for r in rows]
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        """Return the row count."""
        return len(self._inputs)

    def __getitem__(self, i: int) -> dict:
        """Return unpadded id/mask lists + label tensor for row i."""
        enc = self._tokenizer(self._inputs[i], truncation=True, max_length=MAX_TOKENS)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(self._labels[i], dtype=torch.long),
        }


def load_tokenizer() -> AutoTokenizer:
    """Load the pinned tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)


def load_model(device: torch.device) -> DistilBertForSequenceClassification:
    """Load the pinned checkpoint with a fresh 3-way head, fp32 on device."""
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_ID, revision=REVISION, num_labels=len(LABELS)
    )
    return model.to(device).float()


def _sync(device: torch.device) -> None:
    """Block until queued device work finishes (honest latency)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_loop(
    model: DistilBertForSequenceClassification,
    dataset: WeakDataset,
    tokenizer: AutoTokenizer,
    weights: list[float],
    device: torch.device,
    *,
    lr: float = LR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    micro_batch: int = MICRO_BATCH,
    seed: int = SEED,
    max_steps: int | None = None,
    log_every: int = 100,
) -> list[float]:
    """Run a manual AdamW loop at constant lr; return per-step losses.

    Manual loop instead of Trainer: avoids the ``accelerate`` dependency
    so the ``embed`` extra stays sufficient. A partial trailing
    accumulation window is dropped each epoch (rows reshuffle next epoch).
    """
    if batch_size % micro_batch:
        raise ValueError(f"batch {batch_size} not divisible by {micro_batch}")

    def collate(items: list[dict]) -> dict[str, torch.Tensor]:
        """Pad ids/masks to the batch max; stack labels."""
        padded = tokenizer.pad(
            [{k: v for k, v in item.items() if k != "labels"} for item in items],
            return_tensors="pt",
        )
        padded["labels"] = torch.stack([item["labels"] for item in items])
        return padded

    accum = batch_size // micro_batch
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_t)
    history: list[float] = []
    model.train()
    epoch = 0
    while True:
        gen = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=micro_batch,
            shuffle=True,
            generator=gen,
            collate_fn=collate,
        )
        start = len(history)
        opt.zero_grad()
        pending = 0
        for batch in loader:
            moved = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
            ).logits
            loss = loss_fn(logits, moved["labels"]) / accum
            loss.backward()
            pending += 1
            if pending < accum:
                continue
            opt.step()
            opt.zero_grad()
            pending = 0
            step_loss = loss.item() * accum
            if not math.isfinite(step_loss):
                raise ValueError(f"non-finite loss at step {len(history)}")
            history.append(step_loss)
            if len(history) % log_every == 0:
                print(
                    f"[bert] step {len(history)} loss {step_loss:.4f}",
                    flush=True,
                )
            if max_steps is not None and len(history) >= max_steps:
                return history
        epoch += 1
        if history[start:]:
            mean_loss = sum(history[start:]) / len(history[start:])
            print(f"[bert] epoch {epoch} mean loss {mean_loss:.4f}", flush=True)
        if max_steps is None and epoch >= epochs:
            return history


def run_smoke(
    tokenizer: AutoTokenizer,
    device: torch.device,
    rows: list[dict],
    *,
    seed: int = SEED,
    steps: int = SMOKE_STEPS,
) -> list[float]:
    """Train `steps` optimizer steps on a subset; must drop, else raise."""
    set_seed(seed)
    subset = rows[: min(SMOKE_ROWS, len(rows))]
    label_ids = [LABEL_TO_ID[r["weak_label"]] for r in subset]
    dataset = WeakDataset(subset, tokenizer)
    weights = class_weights(label_ids)
    model = load_model(device)
    print(
        f"[bert] smoke: {len(subset)} rows, {steps} steps, "
        f"weights {[round(w, 3) for w in weights]}",
        flush=True,
    )
    history = train_loop(
        model,
        dataset,
        tokenizer,
        weights,
        device,
        seed=seed,
        max_steps=steps,
        log_every=10,
    )
    early = sum(history[:10]) / 10
    late = sum(history[-10:]) / 10
    print(f"[bert] smoke loss first-10 {early:.4f} last-10 {late:.4f}")
    if late >= early:
        raise ValueError(f"smoke failed: loss {early:.4f} -> {late:.4f}")
    return history


def run_train(
    tokenizer: AutoTokenizer,
    device: torch.device,
    rows: list[dict],
    model_dir: Path,
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    micro_batch: int = MICRO_BATCH,
) -> dict:
    """Fine-tune on all weak rows; save model + stats; return the stats."""
    set_seed(seed)
    label_ids = [LABEL_TO_ID[r["weak_label"]] for r in rows]
    weights = class_weights(label_ids)
    counts = [label_ids.count(i) for i in range(len(LABELS))]
    dist = " ".join(f"{label}={counts[i]}" for i, label in enumerate(LABELS))
    print(
        f"[bert] train: {len(rows)} rows {dist} "
        f"weights {[round(w, 3) for w in weights]}",
        flush=True,
    )
    dataset = WeakDataset(rows, tokenizer)
    model = load_model(device)
    t0 = time.monotonic()
    history = train_loop(
        model,
        dataset,
        tokenizer,
        weights,
        device,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        micro_batch=micro_batch,
        seed=seed,
    )
    seconds = time.monotonic() - t0
    if not history:
        raise ValueError("no optimizer steps ran")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    stats = {
        "rows": len(rows),
        "label_counts": {label: counts[i] for i, label in enumerate(LABELS)},
        "class_weights": weights,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "steps": len(history),
        "first_loss": history[0],
        "last_loss": history[-1],
        "seconds": round(seconds, 1),
    }
    with (model_dir / "train-stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    print(
        f"[bert] saved {model_dir} ({len(history)} steps, {seconds:.0f}s)",
        flush=True,
    )
    return stats


def predict_rows(
    model: DistilBertForSequenceClassification,
    tokenizer: AutoTokenizer,
    rows: list[dict],
    device: torch.device,
) -> list[dict]:
    """Score eval rows one at a time; return Contract rows with latency."""
    model.eval()
    out: list[dict] = []
    with torch.no_grad():
        for row in rows:
            enc = tokenizer(
                build_input(row.get("title", ""), row["text"]),
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            _sync(device)
            t0 = time.perf_counter()
            logits = model(**enc).logits[0]
            _sync(device)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            probs = torch.softmax(logits.float(), dim=0).tolist()
            total = sum(probs)
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(f"non-finite probs for {row.get('doc_id')}")
            probs = [p / total for p in probs]
            pred = max(range(len(LABELS)), key=probs.__getitem__)
            out.append(
                {
                    "doc_id": row["doc_id"],
                    "chunk_ord": row["chunk_ord"],
                    "gold": row.get("gold_label", row.get("gold")),
                    "pred": LABELS[pred],
                    "probs": {label: probs[i] for i, label in enumerate(LABELS)},
                    "latency_ms": latency_ms,
                }
            )
    return out


def run_predict(
    tokenizer: AutoTokenizer,
    device: torch.device,
    eval_rows: list[dict],
    model_dir: Path,
) -> list[dict]:
    """Load the fine-tuned head from model_dir; score eval rows."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device).float()
    t0 = time.monotonic()
    rows = predict_rows(model, tokenizer, eval_rows, device)
    print(f"[bert] scored {len(rows)} rows in {time.monotonic() - t0:.1f}s")
    return rows


def build_config(device: torch.device, **extra: object) -> dict:
    """Build the Contract config block (model, seed, device, versions)."""
    return {
        "method": "distilbert",
        "model": MODEL_ID,
        "model_alias": MODEL_ALIAS,
        "revision": REVISION,
        "seed": SEED,
        "device": str(device),
        "date": date.today().isoformat(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        **extra,
    }


def write_results(path: Path, config: dict, rows: list[dict]) -> None:
    """Write a Contract predictions file (config + rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "rows": rows}, handle, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Parse args; run smoke/train/predict per --mode; return exit code."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on weak labels; score the gold set."
    )
    parser.add_argument(
        "--mode", choices=("smoke", "train", "predict", "all"), default="all"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--micro-batch", type=int, default=MICRO_BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke-steps", type=int, default=SMOKE_STEPS)
    args = parser.parse_args(argv)
    train_file = args.train_file or args.data_dir / "train.jsonl"
    eval_file = args.eval_file or args.data_dir / "eval.jsonl"
    model_dir = args.model_dir or args.data_dir / "distilbert"
    out_file = args.out or args.data_dir / "results" / "distilbert.json"
    device = resolve_device(args.device)
    print(
        f"[bert] model {MODEL_ID} (alias {MODEL_ALIAS}) rev {REVISION} "
        f"device {device} seed {args.seed}",
        flush=True,
    )
    tokenizer = load_tokenizer()
    train_rows: list[dict] = []
    if args.mode in ("smoke", "all"):
        train_rows = read_jsonl(train_file)
        run_smoke(
            tokenizer,
            device,
            train_rows,
            seed=args.seed,
            steps=args.smoke_steps,
        )
    if args.mode in ("train", "all"):
        train_rows = train_rows or read_jsonl(train_file)
        run_train(
            tokenizer,
            device,
            train_rows,
            model_dir,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            micro_batch=args.micro_batch,
        )
    if args.mode in ("predict", "all"):
        eval_rows = read_jsonl(eval_file)
        if len(eval_rows) != 200:
            print(f"[bert] WARNING: {len(eval_rows)} eval rows, want 200")
        rows = run_predict(tokenizer, device, eval_rows, model_dir)
        config = build_config(
            device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            imbalance="class-weighted CrossEntropyLoss",
        )
        write_results(out_file, config, rows)
        print(f"[bert] wrote {len(rows)} rows to {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
