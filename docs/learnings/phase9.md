# Phase 9 — Demo, deploy, README: lesson

The repo is now skimmable: Gradio demo over the gateway (`make demo`),
three real screenshots in the README, hosting decision recorded, and
the fresh-clone gate still green. Public HF Spaces hosting is deferred
per the roadmap cut list (item 5) — local compose + recorded screenshots
carry the demo.

## What was built

- **Demo** (`demo/app.py`, ~200 lines): login dropdown (alice/bob/carol/
  admin, JWT in `gr.State`), Ask tab (query + fast/answer + top_k →
  scored hits with label/modality/doc_id + latency breakdown), Classify
  tab (label + prob table + method), Document tab (fetch by doc_id; the
  404 copy states the guarantee explicitly). Handlers take plain values
  and return markdown — `build()` is layout only, so everything below
  the UI was smoke-tested without a browser first.
- **Thin-client discipline holds** (spec §4 must-not): the demo owns no
  rules. Same query as different users returns different results because
  the gateway's SQL says so; the UI just renders. Error text passes
  through verbatim so a 404 never reveals existence.
- **Pin + target**: `gradio==6.29.1` in a `demo` extra (comment convention
  as other extras), `make demo` adds `demo` + `service` extras,
  `TRUSTLAYER_GATEWAY` override for non-local gateways. `make lint` and
  the pinned `uv` in CI now cover `demo/` — the Makefile previously only
  linted `python/` + `data/scripts`.
- **README**: demo paragraph + two screenshots up top (ask, permission
  block), next-steps rewritten (demo/video done-planned, HF Spaces named
  as future). Classify screenshot kept in `docs/` for the Phase 10 video.
- **Screenshots are captures, not mockups**: driven through a real
  browser against the local stack (login → search → classify → carol
  fetches confidential doc → 404). Query text is synthetic; result
  snippets are Enron-derived, consistent with the no-individual-messages
  rule (titles/scores only in prose, snippets only inside the UI frame).

## Findings

- **Escape before decorating, not after**: the first `_render_hits` cut
  spliced `> ` quote markers into the raw snippet and then `html.escape`d
  the whole line — escaping the markers and breaking continuation lines
  (visible in the browser snapshot as literal `>` text). Escape first,
  then add markdown. The unit check (`'\n> line2 &lt;b&gt;' in out`)
  pins the order.
- **Unconditional auth header breaks login**: `_post` always sent
  `Authorization: Bearer {token}`; on the login call the token is empty
  and httpx rejects the bare `Bearer ` value (`Illegal header value`).
  Headers are now conditional on a non-empty token.
- **Accessibility snapshots drop plain-text Markdown output**: the carol
  404 fetch rendered correctly on screen but never appeared in the
  snapshot tree (headings/tables/lists appear; bare paragraphs don't).
  Pixels, not the tree, were the verification — screenshot showed
  `🔒 404 — missing, or not visible to you` exactly as coded.
- **Gradio state survives tab switches**: login once, Ask/Classify/
  Document all share the session token. No per-tab re-auth needed.
- **Hosting decision**: HF Spaces wants a self-contained app with a
  precomputed small index; this demo is a client of a three-container
  stack (Postgres + torch service + gateway). Shipping a Space means a
  second packaging (SQLite/pickle index + in-Space embed) or a publicly
  reachable gateway — neither is free, and the cut list explicitly blesses
  "recorded demo + local instructions" as the fallback. So: local compose
  is the documented path, screenshots are the recorded demo, Spaces stays
  a named next step. No HF credentials were available in this session
  either, which settled it.
- **Acceptance check**: spec §13 criteria 1–5 + 7 hold — leak test green
  (0/69,160 direct, 0/840 API), vectors finite + 768d, README has results
  + Pareto + ship-section, limitations present, `docs/learnings/` covers
  phases 1–9, fresh clone reproduces via CI's `stack` job. Criterion 6
  (video) is Phase 10.

## Tips and tricks

- Drive Gradio with `agent-browser`: `open` → `snapshot -i` → `click` /
  `fill` by `@eN` ref. Refs go stale after every render — re-snapshot
  after each click. Dropdowns need `click` on the combobox, then `click`
  on the `option` (direct `select` doesn't take).
- `demo.launch(server_name="127.0.0.1", ...)` keeps the demo off the
  network; Gradio prints the local URL which the harness `ready` regex
  keys on.
- `uv add --optional demo 'gradio==6.29.1'` writes the pin + lockfile in
  one step; verify with `uv run --extra demo --extra service` since the
  demo imports httpx from the service extra.

Spec pointers: §13, §15.
