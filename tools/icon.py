"""The xLamina icon: a stack of thin slabs (the frames), the middle one in
focus in tyuyamunite yellow, with the name standing through the stack as a 3D
column. The column's cross-section sits on every slab: sharp on the middle slab
and on the top and bottom ones (the frames focused on the top and the bottom of
the column), out of focus in between, and the column's sides in the gaps blur
the same way. Writes the master SVGs and a preview sheet; rasterise with
ImageMagick (librsvg renders the blur filters):
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

Y, YE, G, GE, BG, INK = "#f0d027", "#b89b12", "#5a5a5a", "#3a3a3a", "#262626", "#262626"
COL, COL_DIM = "#a88a14", "#7c6610"   # the column's sides (dark gold)
W, D, O = 84, 11, 26                  # half-width, half-depth, perspective offset of a slab's top face
SHEAR = math.degrees(math.atan(O / (2 * D)))
FONT = 'font-family="Segoe UI, Arial, sans-serif" font-weight="600" font-size="31" letter-spacing="0.5"'


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


def icon(label=True, rounded=True, n=5, pitch=36, th=8):
    """n slabs, `pitch` px apart, `th` px thick; the middle one in focus."""
    half = n // 2
    bg = f'<rect width="256" height="256" rx="{40 if rounded else 0}" fill="{BG}"/>'
    # focus: sharp on the middle slab and on the end slabs, out of focus in between
    def sigma(h):   # h in slab units, signed
        return 2.0 * math.sin(math.pi * min(abs(h), half) / half) if half else 0.0
    sigmas = sorted({round(sigma(k + f), 1) for k in range(-half, half + 1) for f in (0.0, 0.5)} - {0.0})
    defs = "<defs>" + "".join(
        f'<filter id="b{s:.1f}" x="-30%" y="-60%" width="160%" height="220%"><feGaussianBlur stdDeviation="{s}"/></filter>'
        for s in sigmas) + "</defs>"
    g = f'<g transform="translate(128,{128 + (th + 2 * D) // 2})">'
    body = ""
    # bottom to top: slab, then the column's side in the gap above it, then the next slab over it
    for k in range(half, -half - 1, -1):
        y = k * pitch
        mid = k == 0
        body += face(y, Y if mid else G) + front(y, th, YE if mid else GE)
        if label:
            if k > -half:   # the column's side between this slab and the one above, laminated every 3 px
                seg = "".join(text(y - j, COL if (j // 3) % 2 else COL_DIM, "xLamina") for j in range(1, pitch - D))
                body += blurred(seg, sigma(k - 0.5))
            # the name on the slab, over the column's base: ink on the slab in focus, yellow on the grey ones,
            # faded where it is out of focus
            s = sigma(k)
            fill = INK if mid else Y
            op = "" if s < 0.05 else ' opacity="0.75"'
            body += blurred(f"<g{op}>{text(y, fill, 'xLamina')}</g>", s)
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
