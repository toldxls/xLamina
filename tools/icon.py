"""The xLamina icon: a stack of five thin slabs (the frames) with the name on
every one, as a focus stack sees it, on a transparent background. The middle
slab is in focus: a crisp purple xLamina. Above it the name is red and out of
focus, blurrier on the top slab than on the second; below it blue, blurrier on
the bottom slab than on the fourth. One light neutral slab colour carries all
three inks. Writes the master SVGs; rasterise with ImageMagick (librsvg renders
the blur filters):
  python tools/icon.py gui/icon
  magick -background none -density 384 gui/icon/xlamina.svg -resize 256x256 gui/icon/xlamina-256.png      (64..256: the name)
  magick -background none -density 384 gui/icon/xlamina_xl.svg -resize 48x48 gui/icon/xlamina-48.png      (32, 48: "xL")
  magick -background none -density 384 gui/icon/xlamina_small.svg -resize 24x24 gui/icon/xlamina-24.png   (16, 24: no text)
  magick gui/icon/xlamina-256.png ... gui/icon/xlamina-16.png gui/icon/xlamina.ico
The full name is unreadable at 48 px (the desktop's "medium icons" size), so 32 and 48 carry the short "xL"
(Travis, 2026-11-08); 16 and 24 are the bare stack.
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
FONT_XL = 'font-family="Segoe UI, Arial, sans-serif" font-weight="700" font-size="136"'   # "xL" at 32 / 48 px, upright
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
    label: the text on the slabs ("" = none); middle_only: the text on the in-focus slab alone (the short label,
    where the blurred copies would only muddy a 48 px tile)."""
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
        if label and (k == 0 or not middle_only):
            body += blurred(text(y - lift, ink(k), label, font, squash, skew), sigma(k))
    return defs + g + body + "</g>"


def svg(body, w, h):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{body}</svg>'


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
open(f"{out}/xlamina.svg", "w", encoding="utf-8").write(svg(icon("xLamina"), 256, 256))
open(f"{out}/xlamina_small.svg", "w", encoding="utf-8").write(svg(icon("", n=5, pitch=34, th=10), 256, 256))
# "xL" on the middle slab only, taller than the slab so it reads at 48 px; the other slabs stay bare
open(f"{out}/xlamina_xl.svg", "w", encoding="utf-8").write(svg(icon("xL", n=5, pitch=34, th=10, font=FONT_XL, squash=1.0, middle_only=True, skew=False, lift=14), 256, 256))
