#!/usr/bin/env python3
"""Download the MediaPipe Decision Maker .task assets (Laya + GLiNER probe).

Usage:
    uv run python data/scripts/download_decision.py [--out DIR] [--gliner-only]

Fetches the pinned-bucket builds (see data/SOURCES.md) with plain urllib —
no extra dependencies. Writes to data/raw/decision/ by default
(git-ignored). Laya s256 float32 is the Phase 7 decision row; the GLiNER
s256 float16 build is a NaN-probe asset only, never a comparison row.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models/decision_maker"
ASSETS = {
    "laya_s256.task": f"{BASE}/laya/float32/laya_s256/latest/laya_s256.task",
    "gliner_s256.task": f"{BASE}/gliner/float16/gliner_s256/latest/gliner_s256.task",
}

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "raw" / "decision"


def download(out: Path, gliner_only: bool = False) -> list[Path]:
    """Fetch the asset(s) into out/; return the written paths."""
    out.mkdir(parents=True, exist_ok=True)
    names = ["gliner_s256.task"] if gliner_only else list(ASSETS)
    written = []
    for name in names:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"kept {dest} ({dest.stat().st_size / 1e6:.0f} MB, cached)")
        else:
            urllib.request.urlretrieve(ASSETS[name], dest)  # noqa: S310
            print(f"wrote {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        written.append(dest)
    return written


def main(argv: list[str] | None = None) -> int:
    """Parse args and download; return process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir")
    parser.add_argument(
        "--gliner-only", action="store_true", help="fetch only the probe asset"
    )
    args = parser.parse_args(argv)
    download(args.out, args.gliner_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
