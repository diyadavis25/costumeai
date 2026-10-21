"""Procedurally drawn demo garments (so the app works with zero uploads)."""
import io

from PIL import Image, ImageDraw

from .config import PALETTE

SHAPES = {
    "top": [(130, 70), (185, 55), (210, 85), (235, 55), (290, 70), (375, 150), (335, 190),
            (300, 160), (300, 370), (120, 370), (120, 160), (85, 190), (45, 150)],
    "bottom": [(130, 50), (290, 50), (310, 370), (225, 370), (210, 150), (195, 370), (110, 370)],
    "shoes": [(50, 250), (120, 170), (190, 190), (260, 230), (370, 250), (385, 305), (50, 305)],
}


def _shade(rgb, f):
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def make_garment(slot, color, pattern=False, size=420):
    rgb = PALETTE[color] if isinstance(color, str) else tuple(color)
    s = size / 420.0
    pts = [(x * s, y * s) for x, y in SHAPES[slot]]
    img = Image.new("RGB", (size, size), (238, 238, 243))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    fill = Image.new("RGB", (size, size), rgb)
    if pattern:
        d = ImageDraw.Draw(fill)
        stripe = _shade(rgb, 0.45) if sum(rgb) > 300 else _shade(rgb, 2.2) if sum(rgb) > 60 else (90, 90, 90)
        for x in range(0, size, int(22 * s)):
            d.rectangle([x, 0, x + int(8 * s), size], fill=stripe)
    img.paste(fill, mask=mask)
    ImageDraw.Draw(img).line(pts + [pts[0]], fill=_shade(rgb, 0.8), width=max(1, int(2 * s)))
    return img


def garment_bytes(slot, color, pattern=False):
    buf = io.BytesIO()
    make_garment(slot, color, pattern).save(buf, format="PNG")
    return buf.getvalue()
