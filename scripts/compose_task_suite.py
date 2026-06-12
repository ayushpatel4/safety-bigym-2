#!/usr/bin/env python
"""Compose the v17 task_suite figure: keep the original saucepan panel,
swap in the regenerated dishwasher_close / drawers_open_all panels.

Bar style is reverse-engineered from the v16 strip (overlay rows 314-359,
black @ alpha 0.75; title DejaVu Sans Bold white at (10, ~322); second
line DejaVu Sans Bold amber (252,208,119) at (10, ~343)). Run with
--qa to emit a side-by-side of the recreated vs original dishwasher bar
(texts are identical, so any styling drift is visible directly).

Usage (from safety_bigym/)::

    ./venv/bin/python scripts/compose_task_suite.py \
        --dish results/figs/task_suite_remake/dishwasher_close/rerender_ep_d2_s0_k16.png \
        --dish-sep 0.22 \
        --drawers results/figs/task_suite_remake/drawers_open_all/rerender_ep_d2_s1_k41_az090.png \
        --drawers-sep 0.30 \
        --out results/figs/task_suite_remake/task_suite_v17.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = REPO_ROOT / "FYP_v16" / "task_suite.png"
FONT = (REPO_ROOT / "venv/lib/python3.12/site-packages/matplotlib/"
        "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")

PANEL_W, PANEL_H = 480, 360
SEP_W = 8
BAR_TOP = 314
BAR_ALPHA = 0.75
AMBER = (252, 208, 119)
WHITE = (255, 255, 255)


def _fit_font(text: str, target_w: int, lo: int = 8, hi: int = 28) -> ImageFont.FreeTypeFont:
    best = None
    for size in range(lo, hi):
        f = ImageFont.truetype(str(FONT), size)
        w = f.getbbox(text)[2] - f.getbbox(text)[0]
        if best is None or abs(w - target_w) < best[0]:
            best = (abs(w - target_w), f, w)
    return best[1]


def _bar_with_top_anchors(panel, title, regime, sep_m, title_font, sub_font,
                          title_y=318, sub_y=340):
    # Anchors measured from the v16 strip: title glyph tops at y=322,
    # second-line glyph tops at y=343, both starting at x=10.
    panel = panel.convert("RGB")
    a = np.asarray(panel).astype(np.float64)
    a[BAR_TOP:] *= (1.0 - BAR_ALPHA)
    panel = Image.fromarray(a.round().astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    draw.text((10, title_y), title, font=title_font, fill=WHITE)
    draw.text((10, sub_y), f"{regime}  ·  separation here: {sep_m:.2f} m",
              font=sub_font, fill=AMBER)
    return panel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dish", type=Path, required=True)
    p.add_argument("--dish-sep", type=float, required=True)
    p.add_argument("--drawers", type=Path, required=True)
    p.add_argument("--drawers-sep", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--qa", action="store_true",
                   help="Also write a recreated-vs-original dishwasher bar diff.")
    args = p.parse_args()

    orig = Image.open(ORIGINAL).convert("RGB")
    assert orig.size == (3 * PANEL_W + 2 * SEP_W, PANEL_H), orig.size

    # Fit fonts against measured v16 metrics (title 185 px wide for
    # 'dishwasher_close'; second line 430 px for its full string).
    title_font = _fit_font("dishwasher_close", 185)
    sub_font = _fit_font(
        "intermittent co-location  ·  separation here: 0.22 m", 430)

    saucepan = orig.crop((0, 0, PANEL_W, PANEL_H))

    def load_panel(path: Path) -> Image.Image:
        im = Image.open(path).convert("RGB")
        return im.resize((PANEL_W, PANEL_H), Image.LANCZOS)

    dish = _bar_with_top_anchors(
        load_panel(args.dish), "dishwasher_close",
        "intermittent co-location", args.dish_sep, title_font, sub_font)
    drawers = _bar_with_top_anchors(
        load_panel(args.drawers), "drawers_open_all",
        "intermittent co-location", args.drawers_sep, title_font, sub_font)

    strip = Image.new("RGB", orig.size, "white")
    strip.paste(saucepan, (0, 0))
    strip.paste(dish, (PANEL_W + SEP_W, 0))
    strip.paste(drawers, (2 * (PANEL_W + SEP_W), 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(args.out)
    print(f"wrote {args.out}")

    if args.qa:
        ob = orig.crop((PANEL_W + SEP_W, BAR_TOP - 6,
                        PANEL_W + SEP_W + PANEL_W, PANEL_H))
        nb = dish.crop((0, BAR_TOP - 6, PANEL_W, PANEL_H))
        qa = Image.new("RGB", (PANEL_W, 2 * (PANEL_H - BAR_TOP + 6) + 4), "red")
        qa.paste(ob, (0, 0))
        qa.paste(nb, (0, PANEL_H - BAR_TOP + 6 + 4))
        qa = qa.resize((PANEL_W * 2, qa.height * 2), Image.NEAREST)
        qa_path = args.out.with_name("bar_qa.png")
        qa.save(qa_path)
        print(f"wrote {qa_path} (top: v16 original, bottom: recreated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
