"""Phase 9 Gradio demo: a thin client over the C# gateway.

Spec section 4: the demo UI owns nothing but presentation — no business
rules, no ACL logic, no model calls. Every tab below is one gateway
endpoint with its auth header. Run with ``make demo`` (needs ``make
stack`` up first).

No synthetic data of our own: users, roles, and documents live in the
backend. Login posts a demo username to /auth/login and holds the JWT
in session state. Error text comes from the gateway verbatim so a 404
never reveals whether a document exists (same guarantee as the API).
"""

from __future__ import annotations

import html
import os
from typing import Any

import gradio as gr
import httpx

GATEWAY = os.environ.get("TRUSTLAYER_GATEWAY", "http://localhost:8080")
USERS = ["alice", "bob", "carol", "admin"]
TIMEOUT = 30.0
ANSWER_TIMEOUT = 600.0
# Enron is real people's email (SOURCES.md: no individual message in any
# demo, screenshot, or fixture). Recorded artifacts are made with this on:
# hits keep title/score/label/modality/doc_id, document bodies are hidden.
REDACT = os.environ.get("TRUSTLAYER_REDACT_SNIPPETS") == "1"


def _post(
    token: str, path: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """POST one gateway endpoint; raise DemoError on any non-200."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = httpx.post(
        f"{GATEWAY}{path}", json=payload, headers=headers, timeout=timeout
    )
    if resp.status_code != 200:
        raise DemoError(f"{path} → {resp.status_code}: {resp.text[:300]}")
    return dict(resp.json())


class DemoError(Exception):
    """Gateway returned an error; show it verbatim in the UI."""


def login(username: str) -> tuple[str, str]:
    """Log in a demo user; return (jwt, status markdown)."""
    try:
        body = _post("", "/auth/login", {"username": username}, TIMEOUT)
    except (DemoError, httpx.HTTPError) as exc:
        return "", f"❌ login failed: {html.escape(str(exc))}"
    roles = ", ".join(str(r) for r in body.get("roles", []))
    return str(body["token"]), f"✅ **{html.escape(username)}** ({html.escape(roles)})"


def _render_hits(body: dict[str, Any]) -> str:
    """Render /ask results + latency as markdown."""
    parts = []
    results = body.get("results", [])
    if body.get("answer"):
        if REDACT:
            parts.append("### Answer\n\n_answer redacted for the recorded demo._\n")
        else:
            parts.append(f"### Answer\n\n{html.escape(str(body['answer']))}\n")
        titles = (str(c.get("title", "")) for c in body.get("citations", []))
        cites = [f"- {html.escape(t)}" for t in titles]
        parts.append("\n".join(cites) + "\n" if cites else "")
    elif not results:
        parts.append("_No visible results._\n")
    for hit in results:
        parts.append(
            f"### {html.escape(str(hit.get('title', '(untitled)')))}\n\n"
            f"score {hit.get('score', 0):.3f} · "
            f"`{html.escape(str(hit.get('label', '')))}` · "
            f"{html.escape(str(hit.get('modality', '')))} · "
            f"`{html.escape(str(hit.get('doc_id', '')))}`\n"
        )
        if REDACT:
            parts.append("_snippet redacted for the recorded demo._\n")
        else:
            raw = html.escape(str(hit.get("snippet", ""))[:600]).replace("\n", "\n> ")
            parts.append(f"> {raw}\n")
    lat = body.get("latency_ms", {})
    parts.append(
        f"\n---\n_latency ms: embed {lat.get('embed', 0):.0f} · "
        f"search {lat.get('search', 0):.0f} · llm {lat.get('llm', 0):.0f} · "
        f"total {lat.get('total', 0):.0f}_"
    )
    return "\n".join(parts)


def ask(token: str, query: str, mode: str, top_k: float) -> str:
    """Run one /ask call and render the response."""
    if not token:
        return "⚠️ Log in first (sidebar)."
    if not query.strip():
        return "⚠️ Type a question first."
    try:
        body = _post(
            token,
            "/ask",
            {"query": query, "mode": mode, "top_k": int(top_k)},
            ANSWER_TIMEOUT if mode == "answer" else TIMEOUT,
        )
    except (DemoError, httpx.HTTPError) as exc:
        return f"❌ {html.escape(str(exc))}"
    return _render_hits(body)


def classify(token: str, title: str, text: str) -> str:
    """Run one /classify call and render label + probabilities."""
    if not token:
        return "⚠️ Log in first (sidebar)."
    if not text.strip():
        return "⚠️ Paste some text first."
    try:
        body = _post(token, "/classify", {"title": title, "text": text}, ANSWER_TIMEOUT)
    except (DemoError, httpx.HTTPError) as exc:
        return f"❌ {html.escape(str(exc))}"
    probs = body.get("probs", {})
    labels = ("public", "internal", "confidential")
    rows = "\n".join(
        f"| {html.escape(lb)} | {probs.get(lb, 0.0):.3f} |" for lb in labels
    )
    return (
        f"### {html.escape(str(body.get('label', '')))}\n\n"
        f"| label | prob |\n|---|---|\n{rows}\n\n"
        f"_method: {html.escape(str(body.get('method', '')))}_"
    )


def fetch(token: str, doc_id: str) -> str:
    """Fetch one document; a 404 means missing OR invisible (by design)."""
    if not token:
        return "⚠️ Log in first (sidebar)."
    if not doc_id.strip():
        return "⚠️ Paste a doc_id first (copy one from an Ask hit)."
    try:
        resp = httpx.get(
            f"{GATEWAY}/documents/{doc_id.strip()}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return f"❌ {html.escape(str(exc))}"
    if resp.status_code == 404:
        msg = "missing, or not visible to you. The API answers identically."
        return f"🔒 **404** — {msg}"
    if resp.status_code != 200:
        return f"❌ /documents → {resp.status_code}: {html.escape(resp.text[:300])}"
    body = resp.json()
    header = (
        f"### {html.escape(str(body.get('title', '')))}\n\n"
        f"`{html.escape(str(body.get('label', '')))}` · "
        f"{html.escape(str(body.get('modality', '')))} · "
        f"{html.escape(str(body.get('lang', '')))}"
    )
    if REDACT:
        n = len(body.get("chunks", []))
        return f"{header}\n\n_{n} chunk(s) hidden for the recorded demo._"
    texts = (str(c.get("text", ""))[:2000] for c in body.get("chunks", []))
    chunks = "\n\n---\n\n".join(html.escape(t) for t in texts)
    return f"{header}\n\n{chunks}"


def build() -> gr.Blocks:
    """Build the demo app (pure layout; handlers above stay unit-testable)."""
    with gr.Blocks(title="Trust Layer demo") as demo:
        token = gr.State("")
        gr.Markdown("# Trust Layer — permission-aware search demo")
        gr.Markdown(
            "Thin client over the C# gateway (`:8080`). Log in as a demo user, "
            "ask questions, classify text, fetch documents. Switch users to see "
            "the same query return different results."
        )
        with gr.Row():
            user = gr.Dropdown(USERS, value="alice", label="Demo user")
            login_btn = gr.Button("Log in", variant="primary")
        status = gr.Markdown("_Not logged in._")
        login_btn.click(login, inputs=user, outputs=[token, status])

        with gr.Tabs():
            with gr.Tab("Ask"):
                query = gr.Textbox(label="Question", placeholder="vacation policy")
                with gr.Row():
                    mode = gr.Radio(["fast", "answer"], value="fast", label="Mode")
                    top_k = gr.Slider(1, 10, value=5, step=1, label="top_k")
                ask_btn = gr.Button("Search", variant="primary")
                hits = gr.Markdown()
                ask_btn.click(ask, inputs=[token, query, mode, top_k], outputs=hits)
            with gr.Tab("Classify"):
                title = gr.Textbox(label="Title (optional)")
                text = gr.Textbox(
                    label="Text", lines=6, placeholder="Paste an email or document…"
                )
                classify_btn = gr.Button("Classify", variant="primary")
                verdict = gr.Markdown()
                classify_btn.click(
                    classify, inputs=[token, title, text], outputs=verdict
                )
            with gr.Tab("Document"):
                doc_id = gr.Textbox(
                    label="doc_id", placeholder="train:… (copy from an Ask hit)"
                )
                fetch_btn = gr.Button("Fetch", variant="primary")
                document = gr.Markdown()
                fetch_btn.click(fetch, inputs=[token, doc_id], outputs=document)
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7860)
