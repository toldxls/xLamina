"""The xLamina icon: three horizontal slabs stacked vertically, the middle one
(the one in focus) tyuyamunite yellow with the name on it. Writes the master
SVGs and a preview sheet; rasterise with ImageMagick:
  python tools/icon.py gui/icon
  magick -background none -density 384 gui/icon/xlamina.svg -resize 256x256 gui/icon/xlamina-256.png   (48..256)
  magick -background none -density 384 gui/icon/xlamina_small.svg -resize 32x32 gui/icon/xlamina-32.png  (16..32, no text)
  magick gui/icon/xlamina-256.png ... gui/icon/xlamina-16.png gui/icon/xlamina.ico
Sizes <= 32 px drop the text (unreadable there)."""
import sys, os

Y, YE, G, GE, BG, INK = "#f0d027", "#b89b12", "#555555", "#383838", "#262626", "#262626"
W, D, O, TH = 84, 15, 30, 16          # half-width, half-depth, perspective offset, thickness

def slab(y, fill, edge, label=None):
    top = f'<polygon points="{-W},{y-D} {W-O},{y-D} {W},{y+D} {-W+O},{y+D}" fill="{fill}"/>'
    front = f'<polygon points="{-W+O},{y+D} {W},{y+D} {W},{y+D+TH} {-W+O},{y+D+TH}" fill="{edge}"/>'
    txt = ""
    if label:
        shear = O / (2 * D)   # the side edges' slope
        txt = (f'<text transform="translate(0,{y}) skewX({__import__("math").degrees(__import__("math").atan(shear)):.2f}) scale(1,0.78)" '
               f'x="0" y="0" dy="0.36em" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-weight="600" '
               f'font-size="31" fill="{INK}" letter-spacing="0.5">{label}</text>')
    return top + front + txt

def icon(text=True, rounded=True):
    bg = f'<rect width="256" height="256" rx="{40 if rounded else 0}" fill="{BG}"/>'
    g = '<g transform="translate(128,122)">'
    g += slab(-66, G, GE)
    g += slab(0, Y, YE, "xLamina" if text else None)
    g += slab(66, G, GE)
    return bg + g + "</g>"

def svg(body, w, h):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{body}</svg>'

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
open(f"{out}/xlamina.svg", "w", encoding="utf-8").write(svg(icon(True), 256, 256))
open(f"{out}/xlamina_small.svg", "w", encoding="utf-8").write(svg(icon(False), 256, 256))
# sheet: 256 with text, then 128 / 64 / 48 with text, 32 / 24 / 16 without
sheet = '<rect width="760" height="330" fill="#1c1c1c"/>'
sheet += f'<g transform="translate(30,30)">{icon(True)}</g>'
x = 320
for s in (128, 64, 48):
    sheet += f'<g transform="translate({x},{30 + 128 - s}) scale({s/256})">{icon(True)}</g>'
    x += s + 16
x = 320
for s in (32, 24, 16):
    sheet += f'<g transform="translate({x},{200 + 32 - s}) scale({s/256})">{icon(False)}</g>'
    x += s + 16
sheet += '<text x="320" y="300" font-family="Segoe UI, Arial" font-size="14" fill="#bbb">128 / 64 / 48 with the name; 32 / 24 / 16 without</text>'
open(f"{out}/sheet.svg", "w", encoding="utf-8").write(svg(sheet, 760, 330))
