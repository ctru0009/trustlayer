# Phase 10 — Demo video: lesson

The repo now has a 104-second walkthrough (`demo-video/trustlayer-demo.mp4`,
~670KB, no audio) linked from the README with an 8s GIF preview. All four
roadmap beats are in: ask, cited answer, flagged item, permission block
(contrasted carol-404 / alice-render on the same doc). Spec §13 criterion 6
is met; phases 1–10 are done.

## What was built

- **Script** (`demo-video/SCRIPT.md`): 8 shots, caption-narrated (no
  voiceover — captions survive mute and need no mic). Cast: alice asks,
  carol gets blocked, same confidential doc `train:cab24aed` for the
  contrast shot.
- **Captures**: `agent-browser record` → WebM per beat, all actions after
  `record start` (recording resets the page context — any pre-record login
  is wiped; this cost two answer-mode takes before the pattern clicked).
- **Cards** (`demo-video/cards.py`): PIL-rendered title/end cards + six
  caption bars. This machine's ffmpeg has no drawtext/subtitles filters,
  so all text is pre-rendered PNGs composited with plain overlay filters.
  System Arial via absolute `/System/Library/Fonts/Supplemental/` path —
  DejaVu is absent and PIL's bitmap fallback renders postage-stamp text.
- **Assembly** (`demo-video/assemble.sh`): deterministic — normalise to
  720p30, trim, caption overlay on the last 6s of each segment, concat,
  GIF preview. The 170s LLM generation wait is cut: keep the submit click
  (4s) + the render (14s). Reruns byte-differently (x264) but frame-identify.
- **Isolation holds** (spec §14): no package, no `trustlayer` imports,
  nothing in CI. `build/` intermediates are git-ignored; committed are the
  mp4 + gif + script + two build files.

## Findings

- **fframes was the wrong tool, quickly**: a code-first motion-graphics
  framework (Rust struct per video, SVG frames) — recreating the Gradio UI
  in SVG instead of showing real pixels. No Rust toolchain installed
  either. The roadmap's 2-hour budget would have gone to toolchain + API
  learning. Took the blessed fallback (screen recording) in ~15 minutes
  of recon. Right call: the video's value is real UI pixels, not animation.
- **Verify doc_ids against ground truth before recording**: the script
  named `validation:e6cb07e7` from a stale screenshot; alice 404s on it
  too (it's admin-visible, not HR). Direct handler check found
  `train:cab24aed` — alice renders, carol 404s — before re-recording.
  Two wasted takes from trusting a screenshot over the API.
- **Review the video as frames, not vibes**: `select='eq(n,…)'` tile grids
  at segment boundaries caught two mistimed trims (classify/block slices
  ending before their results rendered) that a wall-clock check would
  have missed.
- **Redaction is a demo-mode flag, not post-production**: a review blocker
  caught Enron body text in the first-cut screenshots/video — SOURCES.md
  forbids any individual message in committed artifacts, and a screenshot
  of the UI showing a snippet IS a screenshot showing a message (the
  Phase 9 lesson's "snippets only inside the UI frame" was wrong and is
  corrected). `TRUSTLAYER_REDACT_SNIPPETS=1` hides snippets, generated
  answers (paraphrases still derive from real mail), and document bodies
  behind notices; all committed captures were re-recorded with it on and
  frame-verified. Default stays off for local exploration.
- **Classify timing is the flaky beat**: the submit click needs a settled
  page + fresh refs (stale refs silently no-op), and the result renders
  ~15s later below the fold. Snapshot-gate the scroll on output nodes
  (`cell "internal"`), not on wall time. Short input text keeps input +
  table in one frame.
- **Review carryover fixed first**: CI lint + pre-commit hook now cover
  `../demo` (Phase 9 only extended `make lint` — the lesson claim was
  ahead of the code), and `httpx==0.28.1` is pinned in the demo extra to
  match service. The escape-order advisory was stale: `_render_hits`
  escapes first, quotes after (unit-pinned in Phase 9).
- **Review the video as frames, not vibes**: `select='eq(n,…)'` tile grids
  at segment boundaries caught two mistimed trims (classify/block slices
  ending before their results rendered) that a wall-clock check would
  have missed.

## Tips and tricks

- `agent-browser record start <path>` resets page state — snapshot AFTER
  starting, then act. Refs from before are dead.
- Gradio `scroll` tool works when JS `window.scrollTo` doesn't (virtual
  scroll container); verify with a screenshot, not scrollY.
- ffmpeg `-ss` before `-i` (input seeking) + `trim` filter compose well
  for cutting long waits out of captures.
- `sh -n script.sh` syntax-checks before running; shellcheck SC2086 on
  `$FF`/`$NORM` unquoted vars is intentional (word-splitting wanted).

Spec pointers: roadmap Phase 10, spec §13 (criterion 6), §14.
