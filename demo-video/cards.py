"""Build title/end cards + caption bars for the Phase 10 demo video.

This ffmpeg build has no drawtext/subtitles filters, so all text is
rendered with PIL (pillow, in the embed extra) and composited with
plain ffmpeg image overlays. Run from the repo root:

    cd python && uv run --extra embed python ../demo-video/cards.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (18, 18, 22)
ORANGE = (230, 110, 30)
WHITE = (235, 235, 240)
MUTED = (150, 150, 160)
BAR = (10, 10, 14)

OUT = Path(__file__).resolve().parent / "build"


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Use macOS system fonts (DejaVu is absent); fall back to default."""
    for name in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _card(title: str, subtitle: str, path: Path) -> None:
    """Render one full-frame 1280x720 title/end card."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title, f_sub = _font(64), _font(32)
    d.text((W / 2, H / 2 - 60), title, font=f_title, fill=ORANGE, anchor="mm")
    d.text((W / 2, H / 2 + 40), subtitle, font=f_sub, fill=WHITE, anchor="mm")
    img.save(path)


def _caption(text: str, path: Path) -> None:
    """Render one bottom caption bar (1280x64, opaque) for -filter_complex overlay."""
    img = Image.new("RGB", (W, 64), BAR)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 32), text, font=_font(28), fill=WHITE, anchor="mm")
    img.save(path)


CAPTIONS = [
    ("cap1.png", "Ask by meaning, not keywords"),
    ("cap2.png", "Every hit carries a sensitivity label"),
    ("cap3.png", "Answer mode cites its sources \u2014 or declines"),
    ("cap4.png", "New text gets labelled on arrival"),
    ("cap5.png", "Carol can't see it \u2014 same 404 as missing"),
    ("cap6.png", "Alice can. Permissions live in SQL, not in the model"),
]


def main() -> None:
    """Write all cards + captions into demo-video/build/."""
    OUT.mkdir(exist_ok=True)
    _card("Trust Layer", "private semantic search", OUT / "title.png")
    _card("Run it yourself", "make stack && make demo", OUT / "end.png")
    for name, text in CAPTIONS:
        _caption(text, OUT / name)
    print(f"wrote {2 + len(CAPTIONS)} images to {OUT}")


if __name__ == "__main__":
    main()
