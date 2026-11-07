"""The xLamina icon: a stack of five thin slabs (the frames) with the name on
every one, as a focus stack sees it. The middle slab is in focus: a crisp purple
xLamina. Above it the name is red and out of focus, blurrier on the top slab
than on the second; below it blue, blurrier on the bottom slab than on the
fourth. One light neutral slab colour carries all three inks. Writes the master
SVGs and a preview sheet; rasterise with ImageMagick (librsvg renders the blur
filters):
  python tools/icon.py gui/icon
  magick -background none -density 384 gui/icon/xlamina.svg -resize 256x256 gui/icon/xlamina-256.png   (64..256)
  magick -background none -density 384 gui/icon/xlamina_small.svg -resize 48x48 gui/icon/xlamina-48.png  (16..48, no text)
  magick gui/icon/xlamina-256.png ... gui/icon/xlamina-16.png gui/icon/xlamina.ico
Sizes <= 48 px drop the text (unreadable there; 48 is the desktop's "medium icons" size).
After regenerating, touch gui/xlamina.rc and cli/stackcli.rc: Ninja does not see .ico changes.
If the desktop still shows an old icon after a rebuild, Explorer's icon cache is stale: ie4uinit.exe -show."""
import math
import os
import sys

BG = "#262626"
SLAB, SLAB_EDGE = "#dcdcdc", "#a6a6a6"          # one neutral slab colour: red, blue and purple all read on it
RED, BLUE, PURPLE = "#d6262c", "#2457d8", "#7a2fc9"
W, D, O = 84, 11, 26                             # half-width, half-depth, perspective offset of a slab's top face
SHEAR = math.degrees(math.atan(O / (2 * D)))
FONT = 'font-family="Segoe UI, Arial, sans-serif" font-weight="600" font-size="31" letter-spacing="0.5"'
SIGMA_END, SIGMA_MID = 3.0, 1.8                  # blur of the end slabs' text and of the two next to the middle


def face(y, fill):
    return f'<polygon points="{-W},{y-D} {W-O},{y-D} {W},{y+D} {-W+O},{y+D}" fill="{fill}"/>'


def front(y, th, fill):
    return f'<polygon points="{-W+O},{y+D} {W},{y+D} {W},{y+D+th} {-W+O},{y+D+th}" fill="{fill}"/>'


def text(y, fill, label):
    return (f'<text transform="translate(0,{y:.1f}) skewX({SHEAR:.2f}) scale(1,0.78)" x="0" y="0" dy="0.36em" '
            f'text-anchor="middle" {FONT} fill="{fill}">{label}</text>')


def blurred(body, sigma):
    if sigma < 0.05:
        return body
    return f'<g filter="url(#b{sigma:.1f})">{body}</g>'


def ink(k):
    """k: slab index, negative = above the middle (higher on screen)."""
    return PURPLE if k == 0 else RED if k < 0 else BLUE


def sigma(k):
    return 0.0 if k == 0 else SIGMA_MID if abs(k) == 1 else SIGMA_END


def icon(label=True, rounded=True, n=5, pitch=36, th=8):
    """n slabs, `pitch` px apart, `th` px thick; the middle one in focus."""
    half = n // 2
    bg = f'<rect width="256" height="256" rx="{40 if rounded else 0}" fill="{BG}"/>'
    sigmas = sorted({round(sigma(k), 1) for k in range(-half, half + 1)} - {0.0})
    defs = "<defs>" + "".join(
        f'<filter id="b{s:.1f}" x="-30%" y="-80%" width="160%" height="260%"><feGaussianBlur stdDeviation="{s}"/></filter>'
        for s in sigmas) + "</defs>"
    g = f'<g transform="translate(128,{128 + (th + 2 * D) // 2})">'
    body = ""
    for k in range(half, -half - 1, -1):   # bottom to top, so each slab paints over the one below
        y = k * pitch
        body += face(y, SLAB) + front(y, th, SLAB_EDGE)
        if label:
            body += blurred(text(y, ink(k), "xLamina"), sigma(k))
    return bg + defs + g + body + "</g>"


def svg(body, w, h):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{body}</svg>'


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
open(f"{out}/xlamina.svg", "w", encoding="utf-8").write(svg(icon(True), 256, 256))
open(f"{out}/xlamina_small.svg", "w", encoding="utf-8").write(svg(icon(False, n=5, pitch=34, th=10), 256, 256))
# sheet: 256 with text, then 128 / 64 / 48 with text, 32 / 24 / 16 without
sheet = '<rect width="760" height="330" fill="#1c1c1c"/>'
sheet += f'<g transform="translate(30,30)">{icon(True)}</g>'
x = 320
for s in (128, 64, 48):
    sheet += f'<g transform="translate({x},{30 + 128 - s}) scale({s/256})">{icon(True)}</g>'
    x += s + 16
x = 320
for s in (32, 24, 16):
    sheet += f'<g transform="translate({x},{200 + 32 - s}) scale({s/256})">{icon(False, n=5, pitch=34, th=10)}</g>'
    x += s + 16
sheet += '<text x="320" y="300" font-family="Segoe UI, Arial" font-size="14" fill="#bbb">128 / 64 / 48 with the name; 32 / 24 / 16 without</text>'
open(f"{out}/sheet.svg", "w", encoding="utf-8").write(svg(sheet, 760, 330))
