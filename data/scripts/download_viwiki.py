"""Download a Vietnamese Wikipedia sample (Phase 6 cross-lingual rows).

Fetches ``--rows`` articles from the ``20231101.vi`` config of
``wikimedia/wikipedia`` and writes ``data/raw/viwiki/sample.parquet``
(title, text). Small on purpose: cross-lingual rows need hundreds of
docs, not the full dump.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "data" / "raw" / "viwiki" / "sample.parquet"


def main(argv: list[str] | None = None) -> int:
    """Stream the vi split, keep the first N non-trivial articles."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.vi",
        split="train",
        streaming=True,
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    )
    titles: list[str] = []
    texts: list[str] = []
    for row in ds:
        text = (row["text"] or "").strip()
        if len(text) < 500:
            continue
        titles.append(row["title"])
        texts.append(text[:2000])
        if len(titles) >= args.rows:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"title": titles, "text": texts}), args.out)
    print(f"wrote {args.out.relative_to(ROOT)} ({len(titles)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
