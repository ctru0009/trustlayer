# Demo (Phase 9)

Gradio thin client over the C# gateway. No business rules here — every tab
is one gateway endpoint with an auth header.

```bash
make stack   # db + model service + gateway (:8080)
make demo    # Gradio UI on http://127.0.0.1:7860
```

Log in as `alice` (HR), `bob` (Finance), `carol` (Everyone), or `admin`.
Ask a question, switch users, ask again: the same query returns different
results because the ACL predicate lives in the gateway's SQL, not in this
UI. The Document tab shows the other half of the guarantee — fetch a
confidential `doc_id` as carol and you get the same 404 as a missing
document.

`app.py` handlers (`login`, `ask`, `classify`, `fetch`) take plain values
and return markdown, so they are testable without a browser:
`build()` is layout only. Point at a non-local gateway with
`TRUSTLAYER_GATEWAY=http://host:8080 make demo`.

Screenshots in `docs/demo-*.png` are real captures from this app against
the local stack.
