"""Member 3 - recommendation engine (alternatives + wardrobe ranking)."""
import numpy as np

from .compatibility import get_model, items_matrix
from .config import PALETTE, SLOTS, SLOT_NOUN, score_label


def recommend_alternatives(top, bottom, shoes, model=None, per_slot=3) -> dict:
    """Try every palette colour in each slot and rank the ones that raise the score."""
    model = model or get_model()
    items = {"top": top, "bottom": bottom, "shoes": shoes}
    base = float(model.proba_items([top], [bottom], [shoes])[0])
    out = {}
    for slot in SLOTS:
        cands = []
        for cname, rgb in PALETTE.items():
            if cname == items[slot]["color"]:
                continue
            alt = dict(items[slot])
            alt.update(color=cname, rgb=rgb)
            cands.append(alt)
        trip = {s: [items[s]] * len(cands) for s in SLOTS}
        trip[slot] = cands
        probs = model.proba_items(trip["top"], trip["bottom"], trip["shoes"])
        ranked = np.argsort(-probs)[:per_slot]
        out[slot] = [{"color": cands[i]["color"], "rgb": cands[i]["rgb"],
                      "item_name": f"{cands[i]['color'].title()} {SLOT_NOUN[slot]}",
                      "score": int(round(probs[i] * 100)),
                      "delta": int(round((probs[i] - base) * 100))} for i in ranked]
    best_slot = max(SLOTS, key=lambda s: out[s][0]["score"])
    return {"current_score": int(round(base * 100)), "slots": out, "best_slot": best_slot}


def rank_wardrobe(wardrobe: dict, model=None, top_n=5, diverse=False):
    """Score every top x bottom x shoes combination; return (best list, all scores)."""
    model = model or get_model()
    lists = [wardrobe[s] for s in SLOTS]
    if not all(lists):
        return [], np.array([])
    mats = [items_matrix(l) for l in lists]
    ti, bi, si = (a.ravel() for a in np.meshgrid(*[np.arange(len(l)) for l in lists], indexing="ij"))
    probs = model.proba_matrices(mats[0][ti], mats[1][bi], mats[2][si])
    order = np.argsort(-probs)
    chosen = []
    for k in order:
        trio = (int(ti[k]), int(bi[k]), int(si[k]))
        if diverse and any(sum(a == b for a, b in zip(trio, c["idx"])) >= 2 for c in chosen):
            continue
        score = int(round(probs[k] * 100))
        chosen.append({"idx": trio, "score": score, "label": score_label(score), "probability": float(probs[k])})
        if len(chosen) >= top_n:
            break
    return chosen, probs * 100
