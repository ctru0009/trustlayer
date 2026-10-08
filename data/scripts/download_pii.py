#!/usr/bin/env python3
"""Download the ai4privacy PII-43k corpus (English, labelled, synthetic).

Usage:
    uv run python data/scripts/download_pii.py [--out DIR]

Fetches the single upstream CSV at the pinned revision (see
data/SOURCES.md) with plain urllib — no extra dependencies. Writes to
data/raw/pii/ by default (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REVISION = "7d380930dc1567a61ca861a1b4de433b9601c8dc"
BASE = f"https://huggingface.co/datasets/ai4privacy/pii-masking-43k/resolve/{REVISION}"
FILES = ("PII43k.csv",)

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "raw" / "pii"


def download(out: Path) -> list[Path]:
    """Fetch the CSV file into out/; return the written paths."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in FILES:
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
