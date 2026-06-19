#!/usr/bin/env python3
"""Generate coloring-book orange fill-layers for the gameV2 props.

For each prop PNG (line art on opaque white paper) we emit a sibling `<name>-fill.png`
whose painted pixels are flat orange and whose pencil-line + background pixels are fully
transparent. The game overlays this layer and reveals it on player approach, so the drawing
"colours in" between the lines while the original strokes still show through on top.

  whole  — paint every fillable-paper pixel inside the cutout (tree / haystack / cellar).
  region — flood-fill outward from seed points, stopping at the pencil lines, so only the
           enclosed feature regions get painted (barn: silo dome, hayloft window, cupola,
           silo dome). Seeds are (u, v, r): UV coords with v top->down (matching the
           texture's invertY=true) plus a radius. The line mask is dilated a few px to seal
           sketchy gaps, AND each flood is clipped to a disc of radius r*height around its
           seed so a leak through an open stroke can't spread across the whole drawing.

Run:  pip install pillow numpy  &&  python3 gameV2/tools/gen_fills.py
Sketchy strokes with real gaps may still leak; refine those fills by hand afterwards.
"""

import os
from collections import deque

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "sceneAssets"))

ORANGE = (255, 84, 0)     # matches the shader's BABYLON.Color3(1.0, 0.33, 0.0)
LUMA_T = 135              # L < LUMA_T (and opaque) => pencil line (barrier); else paper
ALPHA_T = 10             # source alpha above this counts as "inside the paper cutout"
LINE_DILATE = 4           # px to thicken the line mask before flooding (seals small gaps)
DISC_SCALE = 1.6          # region flood is clipped to a disc of (r * height * DISC_SCALE)

# ALL prop fills are now HAND-PAINTED region masks by the designer (barn-silo, tree-tractor,
# haystack, route-cellar). The game uses only each PNG's ALPHA as a mask and tints it with
# uGlowColor, so paint the regions in any colour — only where you paint (opaque) matters.
# PROPS is intentionally EMPTY so this script never overwrites a hand-painted fill.
# To auto-generate a NEW prop later, add it here:
#   "name": {"mode": "whole"}                         # fill the entire cutout, OR
#   "name": {"mode": "region", "seeds": [(u, v, r)]}  # flood regions, disc-clipped (v top->down)
PROPS = {}


def luminance(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def dilate(mask, radius):
    """Binary dilation by a square structuring element (pure numpy, no scipy)."""
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[:-1, :] |= out[1:, :]
        grown[1:, :] |= out[:-1, :]
        grown[:, :-1] |= out[:, 1:]
        grown[:, 1:] |= out[:, :-1]
        out = grown
    return out


def flood(paper, seeds):
    """BFS over fillable `paper` pixels from each seed, clipped to a per-seed disc."""
    h, w = paper.shape
    filled = np.zeros_like(paper)
    for (u, v, r) in seeds:
        px, py = int(round(u * (w - 1))), int(round(v * (h - 1)))
        if not (0 <= px < w and 0 <= py < h):
            continue
        # if the seed lands on a line, nudge to the nearest fillable pixel within a small box
        if not paper[py, px]:
            found = False
            for rr in range(1, 25):
                y0, y1 = max(0, py - rr), min(h, py + rr + 1)
                x0, x1 = max(0, px - rr), min(w, px + rr + 1)
                ys, xs = np.where(paper[y0:y1, x0:x1])
                if len(ys):
                    py, px = y0 + ys[0], x0 + xs[0]
                    found = True
                    break
            if not found:
                continue
        rad = r * h * DISC_SCALE          # clip radius in pixels (height-relative, like the shader)
        rad2 = rad * rad
        cy, cx = py, px
        q = deque([(py, px)])
        filled[py, px] = True
        while q:
            y, x = q.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= ny < h and 0 <= nx < w and paper[ny, nx] and not filled[ny, nx]
                        and (ny - cy) ** 2 + (nx - cx) ** 2 <= rad2):
                    filled[ny, nx] = True
                    q.append((ny, nx))
    return filled


def build(name, cfg):
    src_path = os.path.join(ASSETS, name + ".png")
    im = Image.open(src_path).convert("RGBA")
    arr = np.asarray(im)
    rgb, a = arr[..., :3].astype(np.float32), arr[..., 3]

    inside = a > ALPHA_T                      # inside the paper cutout
    L = luminance(rgb)
    line = inside & (L < LUMA_T)              # dark pencil strokes
    paper = inside & ~line                    # fillable white paper

    if cfg["mode"] == "region":
        barrier = dilate(line, LINE_DILATE) | ~inside   # lines + outside both block the flood
        fillable = paper & ~barrier
        filled = flood(fillable, cfg["seeds"])
    else:
        filled = inside                       # whole-prop: the entire cutout (paper AND lines),
                                              # so the orange covers the strokes — no black on top

    # The game tints by uGlowColor and uses only this PNG's ALPHA as the mask, so RGB is just
    # for eyeballing the file; alpha is what matters.
    out = np.zeros_like(arr)
    out[..., 0], out[..., 1], out[..., 2] = ORANGE
    out[..., 3] = np.where(filled, 255, 0).astype(np.uint8)

    dst_path = os.path.join(ASSETS, name + "-fill.png")
    Image.fromarray(out, "RGBA").save(dst_path)
    pct = 100.0 * filled.sum() / inside.sum() if inside.sum() else 0.0
    print(f"  {name:14s} {cfg['mode']:6s} -> {os.path.basename(dst_path)}  "
          f"filled {int(filled.sum()):>9,} px ({pct:5.1f}% of cutout)")


if __name__ == "__main__":
    print("Generating fill-layers into", ASSETS)
    for name, cfg in PROPS.items():
        build(name, cfg)
    print("Done.")
