#!/usr/bin/env python3
"""Download the AESLC dev corpus (Yale-LILY/aeslc) as Parquet.

Usage:
    uv run python data/scripts/download_aeslc.py [--out DIR]

Fetches the three upstream Parquet files at the pinned revision (see
data/SOURCES.md) with plain urllib — no extra dependencies. Writes to
data/raw/aeslc/ by default (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REVISION = "2305f2e63b68056f9b9037a3805c8c196e0d5581"
BASE = f"https://huggingface.co/datasets/Yale-LILY/aeslc/resolve/{REVISION}/data"
SPLITS = ("train", "validation", "test")

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "raw" / "aeslc"


def download(out: Path) -> list[Path]:
    """Fetch the three split files into out/; return the written paths."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for split in SPLITS:
        name = f"{split}-00000-of-00001.parquet"
        dest = out / name
        urllib.request.urlretrieve(f"{BASE}/{name}", dest)  # noqa: S310
        written.append(dest)
        print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return written


def main(argv: list[str] | None = None) -> int:
    """Parse args and download; return process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir")
    args = parser.parse_args(argv)
    download(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
