"""Member 2 deliverable: predict_compatibility(top, bottom, shoes)."""
import numpy as np

from .config import COMPAT_MODEL_PATH, SLOTS, SLOT_TITLE, score_label
from .feature_engineering import item_matrix, outfit_features
from .style_rules import color_harmony, pair_scores, visual_balance


def item_row(item) -> np.ndarray:
    return item_matrix([item["rgb"]], [item.get("busyness", 0.2)])


def items_matrix(items) -> np.ndarray:
    return item_matrix([i["rgb"] for i in items], [i.get("busyness", 0.2) for i in items])


class CompatibilityModel:
    """Loads the trained model (auto-trains if missing / incompatible)."""

    def __init__(self, path=COMPAT_MODEL_PATH):
        import joblib
        self.name = "unknown"
        try:
            bundle = joblib.load(path)
        except Exception:
            from .model_training import train_and_save
            train_and_save(verbose=False)
            bundle = joblib.load(path)
        self.model, self.name = bundle["model"], bundle["name"]

    def proba_matrices(self, T, B, S) -> np.ndarray:
        return self.model.predict_proba(outfit_features(T, B, S))[:, 1]

    def proba_items(self, tops, bottoms, shoes) -> np.ndarray:
        return self.proba_matrices(items_matrix(tops), items_matrix(bottoms), items_matrix(shoes))


_MODEL = None


def get_model() -> CompatibilityModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = CompatibilityModel()
    return _MODEL


def _cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _tips(items, pairs, balance):
    tips = []
    names = {("top", "bottom"): pairs[0], ("top", "shoes"): pairs[1], ("bottom", "shoes"): pairs[2]}
    (a, b), worst = min(names.items(), key=lambda kv: kv[1])
    if worst < 0.6:
        tips.append(f"{SLOT_TITLE[a]} ({items[a]['color']}) and {SLOT_TITLE[b].lower()} "
                    f"({items[b]['color']}) compete. Try a neutral such as black, white, grey or beige for one of them.")
    if balance < 0.6:
        tips.append("The look feels busy or flat. Keep at most one patterned piece and add light/dark contrast between top and bottom.")
    if not tips:
        tips.append("Colours sit well together. Keep accessories simple to let the outfit shine.")
    return tips


def predict_compatibility(top_features, bottom_features, shoe_features, model=None) -> dict:
    """Score an outfit. Pass the dicts returned by analyze_clothing().

    Returns {"score": 87, "label": "Excellent Match", ...breakdown...}.
    """
    items = {"top": top_features, "bottom": bottom_features, "shoes": shoe_features}
    for s, it in items.items():
        if not isinstance(it, dict) or "rgb" not in it:
            raise TypeError(f"{s}: pass the dict returned by analyze_clothing() (needs 'rgb', 'busyness', 'features').")
    model = model or get_model()
    T, B, S = (item_row(items[s]) for s in SLOTS)
    prob = float(model.proba_matrices(T, B, S)[0])
    score = int(round(prob * 100))

    harmony = float(color_harmony(T, B, S)[0])
    balance = float(visual_balance(T, B, S)[0])
    emb = [it.get("features") for it in items.values()]
    sim = None
    if all(e is not None and len(e) for e in emb):
        sim = float(np.clip(np.mean([_cos(emb[0], emb[1]), _cos(emb[0], emb[2]), _cos(emb[1], emb[2])]), 0, 1))
    cats = [items[s].get("category_source") == "classifier" and items[s].get("category") == s for s in SLOTS]
    from_clf = [items[s].get("category_source") == "classifier" for s in SLOTS]
    cat_match = float(np.mean(cats)) if any(from_clf) else None

    pairs = [float(p[0]) for p in pair_scores(T, B, S)]
    return {
        "score": score,
        "label": score_label(score),
        "probability": prob,
        "model": model.name,
        "breakdown": {
            "Color Harmony": round(harmony * 100),
            "Visual Balance": round(balance * 100),
            "Visual Similarity": None if sim is None else round(sim * 100),
            "Category Match": None if cat_match is None else round(cat_match * 100),
        },
        "pair_harmony": {"top-bottom": pairs[0], "top-shoes": pairs[1], "bottom-shoes": pairs[2]},
        "tips": _tips(items, pairs, balance),
    }
