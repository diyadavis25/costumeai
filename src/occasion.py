"""Occasion-aware analysis (classroom, office/interview, party, casual) + uniform detection.

IMPORTANT (be honest in the UI): the app only sees colour, brightness and how busy/patterned each
piece is. It cannot see garment type, fit, fabric, logos, ties or your institution's dress code.
Uniform detection is a heuristic ("looks like a formal shirt + dark trousers/skirt + plain shoes")
and the user can always confirm or override it.
"""
import zlib

from .color_features import rgb_to_hsv
from .config import NEUTRAL_COLORS, SLOTS
from .stylist import _c, _nearest, _pick

OCCASIONS = ["Classroom / College", "Office / Interview", "Party / Night out", "Casual outing"]

OFFICE_LOUD = {"yellow", "orange", "pink", "red"}
SAFE = {"top": ["white", "blue", "grey", "beige"], "bottom": ["navy", "black", "grey", "brown"],
        "shoes": ["black", "brown", "navy"]}
FORMAL_SHOES = {"black", "brown", "navy", "grey"}
UNIFORM_TOPS = {"white", "blue", "grey", "beige", "navy"}
UNIFORM_BOTTOMS = {"navy", "black", "grey"}
UNIFORM_SHOES = {"black", "brown", "white"}
BOLD = {"red", "pink", "purple", "blue", "green", "yellow", "orange"}
DAYTIME = {"beige", "olive", "brown", "grey"}


def _metrics(items):
    out = {}
    for s in SLOTS:
        it = items[s]
        h, sat, v = rgb_to_hsv(it["rgb"])
        out[s] = {"color": it["color"], "sat": sat, "val": v, "busy": float(it.get("busyness", 0.2)),
                  "neutral": it["color"] in NEUTRAL_COLORS}
    return out


def _alts(slot, current, key):
    pool = [c for c in SAFE[slot] if c != current] or SAFE[slot]
    a = _pick(key, pool, slot)
    b = _pick(key, [c for c in pool if c != a] or pool, slot + "b")
    return a, b


def _label(fit):
    return "Great fit" if fit >= 80 else "Good fit" if fit >= 65 else "Borderline" if fit >= 50 else "Not suitable"


# ---------------------------------------------------------------- uniform detection
def looks_like_uniform(items):
    """Heuristic only. Returns (bool, short reason)."""
    M = _metrics(items)
    top_ok = M["top"]["color"] in UNIFORM_TOPS and M["top"]["busy"] < 0.30
    bot_ok = M["bottom"]["color"] in UNIFORM_BOTTOMS and M["bottom"]["busy"] < 0.30
    shoe_ok = M["shoes"]["color"] in UNIFORM_SHOES
    if top_ok and bot_ok and shoe_ok:
        return True, (f"plain {M['top']['color']} top, plain {M['bottom']['color']} bottom and "
                      f"{M['shoes']['color']} shoes match a typical uniform pattern")
    return False, ""


# ---------------------------------------------------------------- per-occasion rules
def _uniform(items, M, key):
    fit, checks, tips = 92, [], []
    if M["shoes"]["color"] in UNIFORM_SHOES:
        checks.append(("good", f"{M['shoes']['color'].title()} shoes suit most uniform codes."))
    else:
        fit -= 22
        checks.append(("bad", f"{M['shoes']['color'].title()} shoes break most uniform rules. Black or brown leather (or plain white if allowed) is safe."))
    for s in ("top", "bottom"):
        if M[s]["busy"] < 0.30:
            checks.append(("good", f"The {s} is plain and solid, as uniforms usually are."))
        else:
            fit -= 10
            checks.append(("warn", f"The {s} looks patterned or textured. Check that it matches the official uniform."))
    tips.append("🏫 <b>Uniform mode:</b> uniform colours are set by your institution, so colour matching doesn't apply here. "
                "What you control is the finish.")
    tips.append("✅ <b>Uniform checklist:</b> shirt ironed and tucked in, belt matching the shoes (black with black, brown with brown), "
                "plain socks, clean polished shoes, ID card visible.")
    return fit, checks, tips


def _office(items, M, key):
    fit, checks, tips = 90, [], []
    chrom = [s for s in SLOTS if not M[s]["neutral"]]
    if len(chrom) <= 1:
        checks.append(("good", "Restrained colours (at most one non-neutral piece) read as professional."))
    elif len(chrom) == 2:
        fit -= 10
        checks.append(("warn", "Two coloured pieces. For a formal look, keep at least one piece neutral."))
    else:
        fit -= 25
        checks.append(("bad", "Three coloured pieces look casual and busy for an office or interview."))
    loud = [s for s in ("top", "bottom") if M[s]["color"] in OFFICE_LOUD]
    for i, s in enumerate(loud):
        fit -= 15 if i == 0 else 10
        a, b = _alts(s, M[s]["color"], key)
        checks.append(("bad", f"A {M[s]['color']} {s} is loud for this setting."))
        tips.append(f"👔 <b>Swap the {s}:</b> {_c(a)} or {_c(b)} is a safer, more professional choice than {M[s]['color']}.")
    for s in ("top", "bottom"):
        if M[s]["busy"] > 0.45:
            fit -= 12
            checks.append(("warn", f"The {s} looks patterned. Solid colours (or a fine stripe) are safer for interviews."))
    if M["shoes"]["busy"] > 0.45:
        fit -= 8
        checks.append(("warn", "The shoes look very detailed. Plain leather reads more formal."))
    sc = M["shoes"]["color"]
    if sc in FORMAL_SHOES:
        checks.append(("good", f"{sc.title()} shoes suit a formal setting."))
    elif sc == "white":
        fit -= 10
        checks.append(("warn", "White sneakers read casual. Black or brown leather shoes look more formal."))
        a, b = _alts("shoes", sc, key)
        tips.append(f"👞 <b>Shoes:</b> go for {_c(a)} or {_c(b)} leather.")
    else:
        fit -= 15
        checks.append(("bad", f"{sc.title()} shoes are too bold for a formal setting."))
        a, b = _alts("shoes", sc, key)
        tips.append(f"👞 <b>Shoes:</b> switch to {_c(a)} or {_c(b)}.")
    if M["bottom"]["color"] == "blue":
        fit -= 6
        checks.append(("warn", "A mid-blue bottom can look like jeans. Navy or grey trousers are safer for formal settings."))
    if M["top"]["color"] == "black" and M["bottom"]["color"] == "black":
        fit -= 5
        checks.append(("warn", "All-black is formal but severe. A white or light blue shirt softens it."))
    if not tips:
        tips.append("💼 <b>Office palette that always works:</b> white or light-blue shirt, navy / grey / black trousers, "
                    "black or brown leather shoes.")
    return fit, checks, tips


def _classroom(items, M, key):
    fit, checks, tips = 88, [], []
    chrom = [s for s in SLOTS if not M[s]["neutral"]]
    if len(chrom) >= 3:
        fit -= 12
        checks.append(("warn", "Three colours at once. Fine for a college day, but one hero colour looks more put together."))
    else:
        checks.append(("good", "Easy, wearable colours for a long day of classes."))
    patt = [s for s in SLOTS if M[s]["busy"] > 0.55]
    if len(patt) >= 2:
        fit -= 10
        checks.append(("warn", "Two patterned pieces. Let one lead and keep the other solid."))
    if sum(M[s]["color"] in OFFICE_LOUD and M[s]["sat"] > 0.7 for s in SLOTS) >= 2:
        fit -= 8
        checks.append(("warn", "Two very bright pieces can feel more party than classroom."))
    if M["shoes"]["color"] in ("white", "grey", "black", "brown", "navy", "beige"):
        checks.append(("good", "Neutral shoes are practical and go with everything."))
    if M["top"]["color"] == "black" and M["bottom"]["color"] == "black" and M["shoes"]["color"] == "black":
        fit -= 4
        checks.append(("warn", "Head-to-toe black is smart but can feel heavy in warm classrooms."))
    tips.append("🎒 <b>Classroom rule of thumb:</b> comfortable, clean and put together. One colour as the hero plus neutrals "
                "always works. Light fabrics are better in warm weather.")
    if len(chrom) == 0:
        a = _pick(key, ["blue", "green", "red", "purple"])
        tips.append(f"✨ <b>Add life:</b> an all-neutral outfit is safe. A {_c(a)} top or bag adds personality.")
    return fit, checks, tips


def _party(items, M, key):
    fit, checks, tips = 72, [], []
    chrom = [s for s in SLOTS if not M[s]["neutral"]]
    bold = [s for s in chrom if M[s]["color"] in BOLD]
    black = [s for s in SLOTS if M[s]["color"] == "black"]
    if bold and len(chrom) <= 2:
        fit += 15
        checks.append(("good", f"A bold {M[bold[0]]['color']} {bold[0]} gives the outfit a party energy."))
    elif len(chrom) >= 3:
        fit -= 8
        checks.append(("warn", "Three colours is a lot. Pick one hero piece and let the others support it."))
    if black and chrom:
        fit += 8
        checks.append(("good", "Black plus colour is a classic evening combination."))
    if not chrom:
        if len(black) >= 2:
            fit -= 2
            checks.append(("warn", "Mostly black is a classic party look, but it needs an accent (shoes, bag or jewellery) to stand out."))
            a = _pick(key, ["red", "pink", "purple"])
            tips.append(f"✨ <b>Accent:</b> add {_c(a)} shoes or a bag. A single pop of colour makes an all-dark outfit look planned.")
        else:
            fit -= 22
            checks.append(("bad", "Soft neutrals look more daytime than party. Add a bold colour or go darker."))
            a = _pick(key, ["red", "purple", "pink", "blue"])
            tips.append(f"🎉 <b>Level it up:</b> try a {_c(a)} top, or a black bottom with a bright top.")
    main = [s for s in ("top", "bottom") if M[s]["color"] in DAYTIME]
    if len(main) == 2:
        fit -= 10
        checks.append(("warn", "Muted earth tones (beige, olive, brown, grey) read as daytime colours."))
    if abs(M["top"]["val"] - M["bottom"]["val"]) < 0.12 and not chrom:
        fit -= 6
        checks.append(("warn", "Top and bottom have the same brightness, so the outfit looks flat under party lights."))
    if M["shoes"]["color"] in ("black", "white", "grey", "navy", "brown", "beige") or M["shoes"]["color"] in BOLD:
        checks.append(("good", "Shoes work for a party. Keep them clean and comfortable enough to last the night."))
    if not tips:
        ref = bold[0] if bold else (chrom[0] if chrom else None)
        if ref:
            comp = _nearest(rgb_to_hsv(items[ref]["rgb"])[0] + 180, {M[ref]["color"]})
            tips.append(f"🎉 <b>Party boost:</b> a small {_c(comp)} accent (bag, jewellery or shoes) is complementary to your "
                        f"{M[ref]['color']} {ref} and makes the whole look pop.")
    return fit, checks, tips


def _casual(items, M, key):
    fit, checks, tips = 88, [], []
    chrom = [s for s in SLOTS if not M[s]["neutral"]]
    if len(chrom) >= 3:
        fit -= 10
        checks.append(("warn", "Three colours is playful, but it works best when one is clearly the star."))
    else:
        checks.append(("good", "Relaxed and wearable. Casual outings are forgiving."))
    if sum(M[s]["busy"] > 0.55 for s in SLOTS) >= 2:
        fit -= 8
        checks.append(("warn", "Two patterned pieces compete. Keep one solid."))
    if sum(M[s]["sat"] > 0.7 and not M[s]["neutral"] for s in SLOTS) >= 2:
        fit -= 6
        checks.append(("warn", "Two very saturated pieces can look loud. Mute one."))
    tips.append("🌤️ <b>Casual tip:</b> comfort first. Matching the shoes to one colour in your outfit makes a relaxed look feel intentional.")
    return fit, checks, tips


RULES = {"Classroom / College": _classroom, "Office / Interview": _office,
         "Party / Night out": _party, "Casual outing": _casual}


def analyze(items, occasion, uniform, compat_score):
    """Return the occasion analysis for the current outfit."""
    M = _metrics(items)
    key = "|".join(f"{M[s]['color']}{round(M[s]['busy'], 1)}" for s in SLOTS) + occasion
    if uniform:
        fit, checks, tips = _uniform(items, M, key)
        overall, name = fit, "Uniform / dress code"
    else:
        fit, checks, tips = RULES[occasion](items, M, key)
        overall, name = None, occasion
    fit = int(max(0, min(100, round(fit))))
    if overall is None:
        overall = int(round(0.55 * compat_score + 0.45 * fit))
    else:
        overall = fit
    return {"occasion": name, "uniform": uniform, "fit": fit, "label": _label(fit), "overall": overall,
            "checks": checks, "tips": tips,
            "note": ("Colour-match score is shown for information only: uniform colours are fixed."
                     if uniform else
                     f"Overall = 55% colour match + 45% occasion fit for {name}.")}