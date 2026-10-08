"""Service tests: pure LR parity, route contracts, input validation.

Heavy models are stubbed — these tests assert the service contract
(shapes, methods, error codes), not model quality. A live smoke run
against real models happens at integration time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trustlayer.service.state import (
    DIM,
    LABELS,
    LazyModels,
    load_lr_weights,
    lr_predict,
    softmax,
)

WEIGHTS = Path("src/trustlayer/service/lr-weights.json")


def _fake_weights() -> dict[str, Any]:
    """Deterministic weights: strong public signal on x[0], else uniform."""
    coef = [[0.0] * DIM for _ in LABELS]
    coef[0][0] = 10.0
    return {
        "labels": list(LABELS),
        "coef": coef,
        "intercept": [0.0] * len(LABELS),
    }


def test_softmax_sums_to_one_and_stable() -> None:
    """Softmax normalizes; large logits do not overflow."""
    probs = softmax([1000.0, 1001.0, 999.0])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[1] > probs[0] > probs[2]


def test_lr_predict_matches_sklearn() -> None:
    """Pure-NumPy-free inference equals sklearn on the same weights."""
    pytest.importorskip("sklearn")
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    weights = _fake_weights()
    clf = LogisticRegression()
    clf.coef_ = np.asarray(weights["coef"])
    clf.intercept_ = np.asarray(weights["intercept"])
    clf.classes_ = np.asarray(weights["labels"])
    vec = [0.0] * DIM
    vec[0] = 1.0
    expected = clf.predict_proba(np.asarray([vec]))[0]
    got = lr_predict(weights, vec)
    for name, want in zip(LABELS, expected, strict=True):
        assert abs(got[name] - float(want)) < 1e-6


def test_lr_predict_rejects_wrong_dim() -> None:
    """Wrong-length vectors fail loudly, never silently truncate."""
    with pytest.raises(ValueError, match="expected dim"):
        lr_predict(_fake_weights(), [0.0] * 256)


def test_committed_weights_shape_and_labels() -> None:
    """The committed artifact loads and has the Phase 7 contract."""
    raw = json.loads(WEIGHTS.read_text())
    assert raw["C"] == 10.0
    assert raw["seed"] == 42
    weights = load_lr_weights(WEIGHTS)
    assert list(weights["labels"]) == list(LABELS)
    assert len(weights["coef"]) == 3
    assert all(len(row) == DIM for row in weights["coef"])


def test_committed_weights_reproduce_phase7_row() -> None:
    """Exported weights + cls vectors reproduce lr-768cls.json exactly."""
    pytest.importorskip("numpy")
    results = Path("../data/classify/results/lr-768cls.json")
    if not results.exists():
        pytest.skip("Phase 7 results not present (git-ignored data)")
    expected = json.loads(results.read_text())["rows"][:8]
    from trustlayer.embed.store import read_chunk_file

    vectors = {}
    for chunk in sorted(Path("../data/processed/embeddings-cls").glob("*.jsonl")):
        for row in read_chunk_file(chunk):
            vectors[(row.doc_id, row.chunk_ord)] = [float(x) for x in row.vector]
    weights = load_lr_weights(WEIGHTS)
    for row in expected:
        vec = vectors[(row["doc_id"], row["chunk_ord"])]
        probs = lr_predict(weights, vec)
        pred = max(LABELS, key=lambda name: probs[name])
        assert pred == row["pred"]
        for name in LABELS:
            assert abs(probs[name] - row["probs"][name]) < 1e-6


def _client() -> Any:
    """TestClient with stubbed models (no torch/mediapipe)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from trustlayer.service import app as app_module

    holders = LazyModels()
    holders.text_model = object()  # mark loaded; _encode is stubbed below
    holders.lr_weights = _fake_weights()
    app_module.MODELS = holders

    def fake_encode(texts: list[str], kind: str) -> list[list[float]]:
        assert kind in ("query", "document", "classification")
        vecs = []
        for text in texts:
            vec = [0.0] * DIM
            vec[0] = 1.0 if "public" in text else 0.5
            vec[1] = 0.5
            norm = sum(x * x for x in vec) ** 0.5
            vecs.append([x / norm for x in vec])
        return vecs

    app_module._encode = fake_encode  # type: ignore[method-assign]
    return TestClient(app_module.app)


def test_health_shape() -> None:
    """Health reports liveness + per-model readiness fields."""
    body = _client().get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "google/embeddinggemma-2"
    assert body["revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert isinstance(body["embed_loaded"], bool)
    assert "errors" in body


def test_embed_contract() -> None:
    """Embed returns normalized vectors at the requested dim."""
    resp = _client().post(
        "/embed", json={"texts": ["hello", "world"], "kind": "query", "dim": 256}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dim"] == 256
    assert len(body["vectors"]) == 2
    for vec in body["vectors"]:
        assert len(vec) == 256
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5


def test_embed_rejects_bad_kind_and_empty() -> None:
    """Unknown kind → 422; empty texts → 422 (never unprompted encode)."""
    client = _client()
    bad = client.post("/embed", json={"texts": ["x"], "kind": "raw"})
    assert bad.status_code == 422
    assert client.post("/embed", json={"texts": []}).status_code == 422


def test_classify_contract_and_method() -> None:
    """Classify returns the ship row's label + probs + method tag."""
    body = _client().post("/classify", json={"text": "a public memo"}).json()
    assert body["method"] == "lr-768cls"
    assert body["label"] == "public"
    assert abs(sum(body["probs"].values()) - 1.0) < 1e-9
    assert set(body["probs"]) == set(LABELS)


def test_classify_rejects_empty_text() -> None:
    """Empty text → 422, never classified."""
    assert _client().post("/classify", json={"text": ""}).status_code == 422
