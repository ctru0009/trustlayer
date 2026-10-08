# Demo video workspace (Phase 10)

Isolated per spec §14 (`linguist-documentation` in `.gitattributes`): no
Python package, no imports from `trustlayer/`, nothing here runs in CI.

## Artifacts

- `SCRIPT.md` — the 90-second shot list (ask, cited answer, flagged item, block).
- `trustlayer-demo.mp4` — final video, 104s, 720p30, ~670KB, no audio.
- `demo.gif` — 8s preview of the ask segment for the README.
- `cards.py` — PIL renderer for title/end cards + caption bars (this
  machine's ffmpeg has no drawtext/subtitles filters, so all text is
  pre-rendered PNGs composited with plain overlay filters).
- `assemble.sh` — deterministic build: normalise captures → trim →
  caption overlays → concat. Inputs are the `/tmp/cap-*.webm` browser
  captures listed in the script; outputs are the mp4 + gif above.
- `build/` — git-ignored intermediates (normalised segments, cards).

## Reproduce

```bash
make stack && make demo   # gateway :8080 + Gradio :7860
# record the /tmp/cap-*.webm captures per SCRIPT.md (agent-browser record)
cd python && uv run --extra embed python ../demo-video/cards.py
sh demo-video/assemble.sh
```
