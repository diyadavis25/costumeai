"""Colour-theory stylist: personalised, outfit-specific advice (replaces the old fixed tips).

Uses the real colours and busyness of the three pieces, names the colour-wheel relationship
(monochromatic / analogous / complementary / split-complementary / triadic / clash), and
suggests concrete colours for a better outfit. Same outfit -> same advice; different outfit ->
different advice.
"""
import zlib

from .color_features import rgb_to_hsv
from .config import NEUTRAL_COLORS, PALETTE, SLOTS, hex_color

CHROMATIC = [c for c in PALETTE if c not in NEUTRAL_COLORS]
HUE = {c: rgb_to_hsv(PALETTE[c])[0] for c in PALETTE}
NOUN = {"top": "top", "bottom": "bottom", "shoes": "shoes"}


# ------------------------------------------------------------------ helpers
def _pick(key, options, salt=""):
    return options[zlib.crc32((key + salt).encode()) % len(options)]


def _sw(name):
    return (f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
            f'background:{hex_color(PALETTE[name])};border:1px solid #0003;margin:0 3px 0 2px;'
            f'vertical-align:middle"></span>')


def _c(name):
    return f"{_sw(name)}<b>{name}</b>"


def _hsv(item):
    return rgb_to_hsv(item["rgb"])


def _neutral(item):
    return item["color"] in NEUTRAL_COLORS


def _hdiff(h1, h2):
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def _scheme(d):
    if d <= 15:
        return "monochromatic"
    if d <= 45:
        return "analogous"
    if d <= 105:
        return "clash"
    if d <= 140:
        return "triadic"
    if d <= 165:
        return "split-complementary"
    return "complementary"


def _nearest(hue, exclude=()):
    best = [c for c in CHROMATIC if c not in exclude]
    return min(best, key=lambda c: _hdiff(HUE[c], hue % 360))


def _neutrals(v, exclude=(), key=""):
    pool = ["white", "beige", "grey"] if v < 0.5 else ["navy", "black", "brown", "grey"]
    pool = [n for n in pool if n not in exclude] or pool
    first = _pick(key, pool)
    rest = [n for n in pool if n != first] or pool
    return first, _pick(key, rest, "b")


def _temp(item):
    if _neutral(item):
        return None
    h = _hsv(item)[0]
    if h < 65 or h >= 330:
        return "warm"
    if 90 <= h < 290:
        return "cool"
    return None


# ------------------------------------------------------------------ advice parts
def _diagnosis(items, chrom, key):
    if not chrom:
        names = ", ".join(items[s]["color"] for s in SLOTS)
        a = _pick(key, CHROMATIC)
        b = _pick(key, [c for c in CHROMATIC if c != a], "x")
        return (f"🎨 <b>All-neutral outfit</b> ({names}): clean and safe, but nothing draws the eye. "
                f"Add one accent of about 10%, for example {_c(a)} or {_c(b)}.")
    if len(chrom) == 1:
        s = chrom[0]
        others = [items[o]["color"] for o in SLOTS if o != s]
        return (f"🎨 <b>Neutral base + one colour:</b> your {_c(items[s]['color'])} {NOUN[s]} is the star while "
                f"{others[0]} and {others[1]} stay quiet. This is the 60-30-10 rule at its easiest, and it's hard to get wrong.")
    if len(chrom) == 2:
        a, b = chrom
        return _pair_sentence(items, a, b, key)
    # three chromatic pieces
    cols = [items[s]["color"] for s in SLOTS]
    pairs = [(SLOTS[0], SLOTS[1]), (SLOTS[0], SLOTS[2]), (SLOTS[1], SLOTS[2])]
    a, b = min(pairs, key=lambda p: _hdiff(_hsv(items[p[0]])[0], _hsv(items[p[1]])[0]))
    return (f"🎨 <b>Three colours at once</b> ({_c(cols[0])}, {_c(cols[1])}, {_c(cols[2])}). That's a lot of voices. "
            f"Let one colour dominate, keep one supporting and one small accent, or turn one piece neutral. "
            f"Your closest pair is the {NOUN[a]} and {NOUN[b]}: {_scheme(_hdiff(_hsv(items[a])[0], _hsv(items[b])[0]))}.")


def _pair_sentence(items, a, b, key):
    ca, cb = items[a]["color"], items[b]["color"]
    sch = _scheme(_hdiff(_hsv(items[a])[0], _hsv(items[b])[0]))
    names = f"{_c(ca)} {NOUN[a]} and {_c(cb)} {NOUN[b]}"
    text = {
        "monochromatic": [f"{names} sit on the same spot of the colour wheel (<b>monochromatic</b>): a calm, tone-on-tone look. "
                          f"Make one piece lighter and one darker so they don't blur together.",
                          f"{names} are the same colour family (<b>monochromatic</b>). Elegant, as long as the shades differ in depth."],
        "analogous": [f"{names} are neighbours on the colour wheel (<b>analogous</b>): harmonious and easy on the eye.",
                      f"{names} sit side by side on the wheel (<b>analogous</b>), a soft, natural blend."],
        "complementary": [f"{names} are opposites on the colour wheel (<b>complementary</b>): high energy. "
                          f"Let one be the star and tone the other down so they don't fight.",
                          f"{names} are <b>complementary</b> colours, so they make each other pop. Keep the third piece neutral."],
        "split-complementary": [f"{names} are almost opposite on the wheel (<b>split-complementary</b>): lively but still balanced.",
                                f"{names} form a <b>split-complementary</b> pair, with contrast without harshness."],
        "triadic": [f"{names} form a <b>triadic-style</b> pairing. It works best when one colour is muted or darker.",
                    f"{names} are spread wide on the wheel (<b>triadic</b>). Keep one soft so the look stays controlled."],
        "clash": [f"{names} are a mid-distance pair on the wheel, the combination most likely to <b>clash</b>.",
                  f"{names} sit in the awkward middle of the colour wheel, so they <b>compete</b> instead of complementing."],
    }[sch]
    return "🎨 " + _pick(key, text, a + b)


def _fix(items, chrom, pairs, key):
    """Specific colour swap for the weakest pair."""
    names = {("top", "bottom"): pairs[0], ("top", "shoes"): pairs[1], ("bottom", "shoes"): pairs[2]}
    (a, b), worst = min(names.items(), key=lambda kv: kv[1])
    if a not in chrom or b not in chrom:
        return None
    change, keep = (b, a) if b != "top" else (a, b)       # prefer changing bottom/shoes over the top
    kh = _hsv(items[keep])[0]
    own = {items[change]["color"], items[keep]["color"]}
    comp = _nearest(kh + 180, own)
    ana = _nearest(kh + 30, own | {comp})
    n1, n2 = _neutrals(_hsv(items[keep])[2], key=key)
    return (f"🔧 <b>Quick fix:</b> swap the {NOUN[change]} ({items[change]['color']}) so it works with your "
            f"{items[keep]['color']} {NOUN[keep]}. Try {_c(comp)} (complementary), {_c(ana)} (analogous), "
            f"or go neutral with {_c(n1)} or {_c(n2)}.")


def _finishing(items, chrom, key):
    opts = []
    if chrom:
        s = max(chrom, key=lambda x: _hsv(items[x])[1])
        c = items[s]["color"]
        if s != "shoes":
            opts.append(f"✨ <b>Finishing touch:</b> echo your {_c(c)} in a small piece (belt, bag, watch or socks) "
                        f"to tie the outfit together.")
        else:
            opts.append(f"✨ <b>Finishing touch:</b> your {_c(c)} shoes are the accent. Repeat that colour once more "
                        f"(a bag or a scarf) so they look planned.")
    temps = [_temp(items[s]) for s in chrom if _temp(items[s])]
    if len(set(temps)) == 2:
        opts.append("🌡️ <b>Warm meets cool:</b> mixing temperatures adds contrast. Keep one temperature dominant "
                    "and the other as the accent.")
    elif temps and temps.count("warm") == len(temps):
        n = _pick(key, ["navy", "grey"])
        opts.append(f"🌡️ <b>All warm tones:</b> cosy and friendly. A cool neutral like {_c(n)} adds depth.")
    elif temps and temps.count("cool") == len(temps):
        n = _pick(key, ["beige", "brown"])
        opts.append(f"🌡️ <b>All cool tones:</b> fresh and calm. A warm neutral like {_c(n)} stops it feeling cold.")
    opts.append("📐 <b>60-30-10 rule:</b> about 60% one main colour, 30% a second, 10% an accent. "
                "Check your outfit against it.")
    return _pick(key, opts, "f")


def _balance(items, key):
    busy = {s: float(items[s].get("busyness", 0.2)) for s in SLOTS}
    patterned = [s for s in SLOTS if busy[s] > 0.55]
    vt, vb = _hsv(items["top"])[2], _hsv(items["bottom"])[2]
    if len(patterned) >= 2:
        a, b = patterned[:2]
        return (f"🧵 <b>Too much pattern:</b> the {NOUN[a]} and {NOUN[b]} both look busy. "
                f"Let one piece lead and keep the other solid.")
    if len(patterned) == 1:
        s = patterned[0]
        others = " and ".join(NOUN[o] for o in SLOTS if o != s)
        return (f"🧵 <b>Pattern piece:</b> your {NOUN[s]} carries the detail, so keep the {others} solid. "
                f"A good trick: pick a colour from the print for the plain pieces.")
    if abs(vt - vb) < 0.12 and not (items["top"]["color"] in NEUTRAL_COLORS and items["bottom"]["color"] in NEUTRAL_COLORS and vt < 0.3):
        light_dark = "a lighter top with a darker bottom" if vt >= vb else "a darker top with a lighter bottom"
        return (f"🌗 <b>Needs contrast:</b> the top and bottom are almost the same brightness, so the outfit looks flat. "
                f"Try {light_dark}.")
    return None


def _formulas(items, chrom, key):
    if not chrom:
        return None
    anchor = max(chrom, key=lambda s: _hsv(items[s])[1])
    h, _, v = _hsv(items[anchor])
    own = items[anchor]["color"]
    others = [s for s in SLOTS if s != anchor]
    comp = _nearest(h + 180, {own})
    ana = _nearest(h - 30, {own, comp}) if _pick(key, [0, 1], "d") else _nearest(h + 30, {own, comp})
    n1, n2 = _neutrals(v, key=key)
    lines = [
        f"• <b>Complementary:</b> {NOUN[others[0]]} in {_c(comp)}, {NOUN[others[1]]} in {_c(n1)}",
        f"• <b>Analogous:</b> {NOUN[others[1]]} in {_c(ana)}, {NOUN[others[0]]} in {_c(n2)}",
        f"• <b>Neutral:</b> {NOUN[others[0]]} in {_c(n1)}, {NOUN[others[1]]} in {_c(n2)}",
    ]
    return f"💡 <b>Colour-theory looks around your {_c(own)} {NOUN[anchor]}:</b><br>" + "<br>".join(lines)


# ------------------------------------------------------------------ public API
def style_advice(items, pairs, balance, score):
    """Return a list of HTML tip strings (max 4) tailored to this exact outfit."""
    key = "|".join(f"{items[s]['color']}{round(float(items[s].get('busyness', 0.2)), 1)}" for s in SLOTS)
    chrom = [s for s in SLOTS if not _neutral(items[s])]
    tips = [_diagnosis(items, chrom, key)]
    weak = min(pairs) < 0.62
    extra = _fix(items, chrom, pairs, key) if weak else None
    if extra is None and score < 65 and len(chrom) >= 2:
        sats = sorted(chrom, key=lambda x: -_hsv(items[x])[1])[:2]
        extra = (f"🔧 <b>Tone it down:</b> the colours follow colour theory, but the {_c(items[sats[0]]['color'])} "
                 f"{NOUN[sats[0]]} and {_c(items[sats[1]]['color'])} {NOUN[sats[1]]} are both strong. "
                 f"Soften one with a muted or darker shade, or make the third piece neutral.")
    tips.append(extra or _finishing(items, chrom, key))
    bal = _balance(items, key)
    if bal:
        tips.append(bal)
    f = _formulas(items, chrom, key)
    if f and len(tips) < 4:
        tips.append(f)
    return tips