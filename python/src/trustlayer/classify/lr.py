"""Logistic regression on Classification-prefix embeddings (Phase 7 LR rows).

Needs the ``classify`` extra (scikit-learn) plus ``embed`` for the PII
ablation encode. Trains on weak labels only — no gold peeking: 5-fold
stratified CV over a C grid picks C, then a final model trains on the
full weak set and predicts the gold eval rows.

Run from the repo root (after ``classify-data`` and ``classify-vectors``)::

    make classify-lr        # → data/classify/results/lr-768cls.json
    make classify-lr-pii    # → data/classify/results/lr-768cls+pii.json

The PII ablation maps synthetic PII fills to 3-class labels via the
severity tiers in ``trustlayer.labels.rules`` and mixes them into the
weak train set before the same CV → train → predict flow.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRAIN_DEFAULT = ROOT / "data" / "classify" / "train.jsonl"
EVAL_DEFAULT = ROOT / "data" / "classify" / "eval.jsonl"
VECTORS_DEFAULT = ROOT / "data" / "processed" / "embeddings-cls"
OUT_DEFAULT = ROOT / "data" / "classify" / "results" / "lr-768cls.json"
OUT_PII_DEFAULT = ROOT / "data" / "classify" / "results" / "lr-768cls+pii.json"
PII_CSV_DEFAULT = ROOT / "data" / "raw" / "pii" / "PII43k.csv"
METHOD_BASE = "lr-768cls"
METHOD_PII = "lr-768cls+pii"
LABELS = ("public", "internal", "confidential")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 42


def _load_vectors(vec_dir: Path) -> dict[tuple[str, int], list[float]]:
    """Load all cls chunk files into a (doc_id, chunk_ord) → vector map."""
    from trustlayer.embed.store import read_chunk_file

    vectors: dict[tuple[str, int], list[float]] = {}
    for path in sorted(vec_dir.glob("chunk-*.jsonl")):
        for row in read_chunk_file(path):
            vectors[(row.doc_id, row.chunk_ord)] = list(row.vector)
    return vectors


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _label_counts(labels: list[str]) -> dict[str, int]:
    """Count occurrences of each label in order."""
    return {name: sum(1 for lab in labels if lab == name) for name in LABELS}


def _encode_pii_texts(
    model: object, texts: list[str], batch_size: int
) -> list[list[float]]:
    """Encode PII fills with the Classification prompt; normalized fp32 out."""
    from trustlayer.embed.model import MAX_LENGTH

    vectors = model.encode(  # type: ignore[attr-defined]
        texts,
        prompt_name="Classification",
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        processing_kwargs={"text": {"max_length": MAX_LENGTH, "truncation": True}},
    )
    return [row.astype("float32").tolist() for row in vectors]


def _bio_entities(tags: list[str]) -> set[str]:
    """Distinct entity types from B- tags (B- marks entity starts)."""
    return {tag[2:] for tag in tags if tag.startswith("B-")}


def map_bio_to_label(tags: list[str]) -> str:
    """Map a BIO tag sequence to public/internal/confidential.

    Any confidential-tier entity wins; else any internal-tier entity;
    untagged rows are public. Unknown entity types raise — the mapping
    table must stay exhaustive over the CSV's entity set.
    """
    entities = _bio_entities(tags)
    unknown = {e for e in entities if e not in BIO_LABEL_MAP}
    if unknown:
        msg = f"unmapped PII entities: {sorted(unknown)}"
        raise ValueError(msg)
    if entities & _CONFIDENTIAL_ENTITIES:
        return "confidential"
    if entities:
        return "internal"
    return "public"


def load_pii_train(
    pii_csv: Path, encode_device: str, batch_size: int
) -> tuple[list[str], list[str], list[list[float]]]:
    """Map PII fills to 3-class labels and encode them (Classification prompt).

    Returns (texts, labels, vectors). NUL bytes are stripped before CSV
    parse per the SOURCES.md caveat; labels come from the explicit
    ``BIO_LABEL_MAP`` over the CSV's BIO tags (mirrors the severity
    tiers in ``trustlayer.labels.rules``: credential/financial entities
    → confidential, other entities → internal, untagged → public).
    """
    import ast
    import math

    from trustlayer.embed.model import DIM, load_text_model

    raw = pii_csv.read_bytes().replace(b"\x00", b"").decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    texts: list[str] = []
    labels: list[str] = []
    for row in reader:
        filled = row.get("Filled Template")
        if not filled:
            continue
        texts.append(filled)
        labels.append(map_bio_to_label(ast.literal_eval(row["Tokens"])))
    model = load_text_model(encode_device)
    vectors = _encode_pii_texts(model, texts, batch_size)
    for vec in vectors:
        if len(vec) != DIM or not all(math.isfinite(x) for x in vec):
            msg = "non-finite or wrong-dim PII vector"
            raise ValueError(msg)
    return texts, labels, vectors


_CONFIDENTIAL_ENTITIES = frozenset(
    {
        # High-severity: direct account/credential takeover or money movement.
        "PASSWORD",
        "PIN",
        "CREDITCARDNUMBER",
        "CREDITCARDCVV",
        "IBAN",
        "BIC",
        "ACCOUNTNUMBER",
        "MASKEDNUMBER",
        "BITCOINADDRESS",
        "ETHEREUMADDRESS",
        "LITECOINADDRESS",
    }
)

_INTERNAL_ENTITIES = frozenset(
    {
        # Identifying/contactable: personal data, not directly exfiltrative.
        "FULLNAME",
        "NAME",
        "FIRSTNAME",
        "LASTNAME",
        "DISPLAYNAME",
        "ACCOUNTNAME",
        "USERNAME",
        "EMAIL",
        "CITY",
        "STATE",
        "COUNTY",
        "STREET",
        "STREETADDRESS",
        "BUILDINGNUMBER",
        "SECONDARYADDRESS",
        "ZIPCODE",
        "NEARBYGPSCOORDINATE",
        "IP",
        "IPV4",
        "IPV6",
        "MAC",
        "URL",
        "USERAGENT",
        "AMOUNT",
        "CURRENCY",
        "CURRENCYCODE",
        "CURRENCYNAME",
        "CURRENCYSYMBOL",
        "CREDITCARDISSUER",
        "NUMBER",
        "GENDER",
        "SEX",
        "SEXTYPE",
        "JOBAREA",
        "JOBDESCRIPTOR",
        "JOBTITLE",
        "JOBTYPE",
        "ORDINALDIRECTION",
    }
)

# Explicit BIO entity → 3-class mapping (logged in the ablation config):
# high-severity credential/financial entities → confidential, any other
# tagged entity → internal, no B- tag → public.
BIO_LABEL_MAP: dict[str, str] = {
    **dict.fromkeys(sorted(_CONFIDENTIAL_ENTITIES), "confidential"),
    **dict.fromkeys(sorted(_INTERNAL_ENTITIES), "internal"),
}


def pick_c(
    x_train: object, y_train: list[str], seed: int = SEED
) -> tuple[float, dict[float, float]]:
    """5-fold stratified CV over C_GRID (macro F1); return best C + means."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    means: dict[float, float] = {}
    for c in C_GRID:
        scores = []
        for train_idx, val_idx in splitter.split(x_train, y_train):  # type: ignore[arg-type]
            clf = LogisticRegression(
                C=c,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
            )
            clf.fit(x_train[train_idx], [y_train[i] for i in train_idx])  # type: ignore[index]
            pred = clf.predict(x_train[val_idx])  # type: ignore[index]
            val_true = [y_train[i] for i in val_idx]
            scores.append(f1_score(val_true, pred, average="macro", zero_division=0))
        means[c] = sum(scores) / len(scores)
    best = max(C_GRID, key=lambda c: means[c])
    return best, means


def _versions() -> dict[str, str]:
    """Collect pinned versions for the result config."""
    import platform

    versions = {"python": platform.python_version()}
    for dist in ("scikit-learn", "numpy", "torch", "sentence-transformers"):
        try:
            from importlib.metadata import version

            versions[dist] = version(dist)
        except Exception:  # noqa: BLE001, S110 — missing extra, record blank
            versions[dist] = "unknown"
    return versions


def main(argv: list[str] | None = None) -> int:
    """CV-pick C on weak labels, train final LR, predict eval, write JSON."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from trustlayer.embed.model import DIM, MODEL_ID, REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--eval", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--vectors", type=Path, default=VECTORS_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--with-pii", action="store_true")
    parser.add_argument("--pii-csv", type=Path, default=PII_CSV_DEFAULT)
    parser.add_argument("--encode-device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    method = args.method or (METHOD_PII if args.with_pii else METHOD_BASE)
    out = args.out or (OUT_PII_DEFAULT if args.with_pii else OUT_DEFAULT)

    vectors = _load_vectors(args.vectors)
    print(f"loaded {len(vectors)} cls vectors from {args.vectors}", flush=True)

    train_rows = _read_jsonl(args.train)
    x_list = [vectors[(r["doc_id"], int(r["chunk_ord"]))] for r in train_rows]
    y_list = [str(r["weak_label"]) for r in train_rows]
    for lab in y_list:
        if lab not in LABELS:
            msg = f"unexpected train label {lab!r}"
            raise ValueError(msg)
    print(f"train rows={len(train_rows)} counts={_label_counts(y_list)}", flush=True)

    pii_info: dict = {}
    if args.with_pii:
        _, pii_labels, pii_vecs = load_pii_train(
            args.pii_csv, args.encode_device, args.batch_size
        )
        x_list.extend(pii_vecs)
        y_list.extend(pii_labels)
        pii_info = {
            "pii_rows": len(pii_labels),
            "pii_counts": _label_counts(pii_labels),
            "pii_bio_map": dict(sorted(BIO_LABEL_MAP.items())),
            "pii_note": (
                "ablated secondary row: synthetic fills mixed into weak train, "
                "no rebalancing; lr-768cls (no synthetic) is the primary row"
            ),
        }
        print(
            f"mixed {len(pii_labels)} PII rows counts={pii_info['pii_counts']}",
            flush=True,
        )

    x_train = np.asarray(x_list, dtype=np.float32)
    if x_train.shape[1] != DIM:
        msg = f"expected dim {DIM}, got {x_train.shape[1]}"
        raise ValueError(msg)

    best_c, cv_means = pick_c(x_train, y_list, seed=args.seed)
    print(
        f"C grid CV macro-F1: { {c: round(s, 4) for c, s in cv_means.items()} }",
        flush=True,
    )
    print(f"best C={best_c}", flush=True)

    clf = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=args.seed,
    )
    clf.fit(x_train, y_list)

    eval_rows = _read_jsonl(args.eval)
    x_eval = np.asarray(
        [vectors[(r["doc_id"], int(r["chunk_ord"]))] for r in eval_rows],
        dtype=np.float32,
    )
    class_index = {str(cls): i for i, cls in enumerate(clf.classes_)}

    clf.predict_proba(x_eval[: min(8, len(x_eval))])  # warm-up, untimed
    rows: list[dict] = []
    for row, vec in zip(eval_rows, x_eval, strict=True):
        gold = str(row.get("gold_label", row.get("gold")))
        t0 = time.perf_counter()
        proba = clf.predict_proba(vec.reshape(1, -1))[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs = {
            name: float(proba[class_index[name]]) if name in class_index else 0.0
            for name in LABELS
        }
        total = sum(probs.values())
        probs = {name: p / total for name, p in probs.items()}
        pred = max(LABELS, key=lambda name: probs[name])
        rows.append(
            {
                "doc_id": row["doc_id"],
                "chunk_ord": int(row["chunk_ord"]),
                "gold": gold,
                "pred": pred,
                "probs": probs,
                "latency_ms": latency_ms,
            }
        )

    config: dict = {
        "method": method,
        "model": f"{MODEL_ID} Classification-prefix + sklearn LogisticRegression",
        "revision": REVISION,
        "seed": args.seed,
        "device": "cpu",
        "date": date.today().isoformat(),
        "versions": _versions(),
        "C": best_c,
        "cv_macro_f1": {str(c): round(s, 4) for c, s in cv_means.items()},
        "train_rows": len(y_list),
        "train_counts": _label_counts(y_list),
        "eval_rows": len(rows),
    }
    if pii_info:
        config["encode_device"] = args.encode_device
        config.update(pii_info)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": config, "rows": rows}, indent=2) + "\n")
    print(f"wrote {out} ({len(rows)} rows, C={best_c})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
