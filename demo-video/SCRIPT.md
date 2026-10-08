# Demo video script (~90s)

Goal: show all four roadmap beats — ask, cited answer, flagged item, block.
Recorded against the local stack (`make stack` + `make demo`), narrated by
captions baked into the video (no voiceover; captions survive mute).

## Cast

- Gateway on :8080, Gradio demo on :7860.
- alice (HR, Everyone) asks. carol (Everyone) gets blocked.
- Query: `vacation policy`. Classify input is synthetic HR text.
- Document fetch: `validation:e6cb07e7` (confidential, HR-only).

## Shots

| # | Seconds | Tab | Action | Caption |
|---|---------|-----|--------|---------|
| 1 | 0–8 | — | Title card: project name + one-line pitch | Trust Layer: private semantic search |
| 2 | 8–20 | Ask (fast) | Log in as alice, ask `vacation policy`, top_k 5 | Ask by meaning, not keywords |
| 3 | 20–32 | Ask | Hits scroll: point at label chips + scores | Every hit carries a sensitivity label |
| 4 | 32–48 | Ask (answer) | Switch to answer mode, same query, cited answer renders | Answer mode cites its sources — or declines |
| 5 | 48–60 | Classify | Classify synthetic text, prob table renders | New text gets labelled on arrival |
| 6 | 60–74 | Document | Log in as carol, fetch `validation:e6cb07e7` → 404 | Carol can't see it — same 404 as missing |
| 7 | 74–84 | Document | Log in as alice, same doc_id → renders | Alice can. Permissions live in SQL, not in the model |
| 8 | 84–90 | — | End card: repo URL + `make stack && make demo` | Run it yourself |

Total: ~90s. Each shot is one continuous screen capture; captions and
title/end cards are baked in with ffmpeg drawtext (no editor round-trip).

## Beats checklist (roadmap)

- [ ] Ask a question (shot 2)
- [ ] Cited answer (shot 4)
- [ ] Sensitive item flagged (shots 3 + 5)
- [ ] Permission block (shot 6, contrasted by shot 7)

## Build

See `demo-video/README.md`. Source captures stay local-only (git-ignored);
the committed artifacts are `trustlayer-demo.mp4` (≤10MB) + `demo.gif`
(≤5MB preview) + this script.
