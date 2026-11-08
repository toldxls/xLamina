"""The xLamina icon: a stack of five thin slabs (the frames) with the name on
every one, as a focus stack sees it, on a transparent background. The middle
slab is in focus: a crisp purple xLamina. Above it the name is red and out of
focus, blurrier on the top slab than on the second; below it blue, blurrier on
the bottom slab than on the fourth. One light neutral slab colour carries all
three inks. Writes the master SVGs; rasterise with ImageMagick (librsvg renders
the blur filters):
  python tools/icon.py gui/icon
  magick -background none -density 384 gui/icon/xlamina.svg -resize 256x256 gui/icon/xlamina-256.png      (24..256: the name)
  magick -background none -density 384 gui/icon/xlamina_xl.svg -resize 16x16 gui/icon/xlamina-16.png      (16 alone: "xL")
  magick gui/icon/xlamina-256.png ... gui/icon/xlamina-16.png gui/icon/xlamina.ico
Every size from 24 px up carries the name on all five slabs, blurred red above and blue below, sharp purple in
the middle: unreadable at 48 px but it is the look Travis wants on the desktop (2026-11-08 evening; a 48 px "xL"
tried that morning came out cut off). Only the 16 px tile, where the layers are a smear, has a big upright "xL".
After regenerating, touch gui/xlamina.rc and cli/stackcli.rc: Ninja does not see .ico changes.
If the desktop still shows an old icon after a rebuild, Explorer's icon cache is stale: ie4uinit.exe -show."""
import math
import os
import sys

SLAB, SLAB_EDGE = "#dcdcdc", "#a6a6a6"          # one neutral slab colour: red, blue and purple all read on it
RED, BLUE, PURPLE = "#d6262c", "#2457d8", "#7a2fc9"
W, D, O = 84, 11, 26                             # half-width, half-depth, perspective offset of a slab's top face
SHEAR = math.degrees(math.atan(O / (2 * D)))
FONT = 'font-family="Segoe UI, Arial, sans-serif" font-weight="600" font-size="31" letter-spacing="0.5"'
FONT_XL = 'font-family="Segoe UI, Arial, sans-serif" font-weight="700" font-size="104"'   # "xL" at 16 px, upright, inside the tile
SIGMA_END, SIGMA_MID = 3.0, 1.8                  # blur of the end slabs' text and of the two next to the middle
SCALE = 1.38                                     # the stack fills the 256 px tile (no background square any more)


def face(y, fill):
    return f'<polygon points="{-W},{y-D} {W-O},{y-D} {W},{y+D} {-W+O},{y+D}" fill="{fill}"/>'


def front(y, th, fill):
    return f'<polygon points="{-W+O},{y+D} {W},{y+D} {W},{y+D+th} {-W+O},{y+D+th}" fill="{fill}"/>'


def text(y, fill, label, font=FONT, squash=0.78, skew=True):
    """the label lying on the slab (skewed into its perspective); skew=False: upright, as a sign on the stack"""
    tf = f'translate(0,{y:.1f}) skewX({SHEAR:.2f}) scale(1,{squash})' if skew else f'translate(0,{y:.1f}) scale(1,{squash})'
    return (f'<text transform="{tf}" x="0" y="0" dy="0.36em" '
            f'text-anchor="middle" {font} fill="{fill}">{label}</text>')


def blurred(body, sigma):
    if sigma < 0.05:
        return body
    return f'<g filter="url(#b{sigma:.1f})">{body}</g>'


def ink(k):
    """k: slab index, negative = above the middle (higher on screen)."""
    return PURPLE if k == 0 else RED if k < 0 else BLUE


def sigma(k):
    return 0.0 if k == 0 else SIGMA_MID if abs(k) == 1 else SIGMA_END


def icon(label="xLamina", n=5, pitch=36, th=8, scale=SCALE, font=FONT, squash=0.78, middle_only=False, skew=True, lift=0):
    """n slabs, `pitch` px apart, `th` px thick; the middle one in focus; transparent background.
    label: the text on the slabs ("" = none); middle_only: the text on the in-focus slab alone, painted over the
    whole stack (a sign on it, not cut by the slabs above) for the 16 px tile."""
    half = n // 2
    sigmas = sorted({round(sigma(k), 1) for k in range(-half, half + 1)} - {0.0})
    defs = "<defs>" + "".join(
        f'<filter id="b{s:.1f}" x="-30%" y="-80%" width="160%" height="260%"><feGaussianBlur stdDeviation="{s}"/></filter>'
        for s in sigmas) + "</defs>"
    # the stack spans y = -half*pitch - D .. half*pitch + D + th; centre it in the tile, then scale about the centre
    cy = (th) / 2
    g = f'<g transform="translate(128,128) scale({scale}) translate(0,{-cy:.1f})">'
    body = ""
    for k in range(half, -half - 1, -1):   # bottom to top, so each slab paints over the one below
        y = k * pitch
        body += face(y, SLAB) + front(y, th, SLAB_EDGE)
        if label and not middle_only:
            body += blurred(text(y - lift, ink(k), label, font, squash, skew), sigma(k))
    if label and middle_only:   # last: on top of every slab
        body += text(-lift, ink(0), label, font, squash, skew)
    return defs + g + body + "</g>"


def svg(body, w, h):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{body}</svg>'


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
open(f"{out}/xlamina.svg", "w", encoding="utf-8").write(svg(icon("xLamina"), 256, 256))
# 16 px alone: "xL" upright over the bare stack, centred, small enough to stay inside the tile
open(f"{out}/xlamina_xl.svg", "w", encoding="utf-8").write(svg(icon("xL", n=5, pitch=34, th=10, font=FONT_XL, squash=1.0, middle_only=True, skew=False, lift=4), 256, 256))
