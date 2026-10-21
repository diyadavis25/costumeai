"""Member 2 - build the labelled outfit-pair dataset (outfit_pairs.csv).

IMPORTANT: with no public colour-only outfit-label set available offline, labels are
generated from the transparent rules in style_rules.py plus Gaussian label noise.
The ML models therefore learn an approximation of these rules. Replace/extend
data/outfit_pairs.csv with real human-labelled outfits (or the in-app thumbs-up/down
feedback in data/user_feedback.csv) and retrain for a real-world model.
"""
import numpy as np
import pandas as pd

from .color_features import name_color
from .config import OUTFIT_CSV, PALETTE, RANDOM_SEED, SLOTS
from .feature_engineering import item_matrix
from .style_rules import latent_compatibility

NAMES = list(PALETTE)
THRESHOLD = 0.77
NOISE_SD = 0.06


def _sample_items(rng, n):
    neutral_w = np.array([0.16 if c in ("black", "white", "grey", "beige", "navy", "brown") else 0.0 for c in NAMES])
    chroma_w = np.array([0.0 if c in ("black", "white", "grey", "beige", "navy", "brown") else 1.0 for c in NAMES])
    w = neutral_w / neutral_w.sum() * 0.5 + chroma_w / chroma_w.sum() * 0.5
    idx = rng.choice(len(NAMES), size=n, p=w)
    rgb = np.array([PALETTE[NAMES[i]] for i in idx], dtype=float) + rng.normal(0, 12, (n, 3))
    # 12% fully random colours for variety
    rnd = rng.random(n) < 0.12
    rgb[rnd] = rng.integers(0, 256, (rnd.sum(), 3))
    rgb = np.clip(rgb, 0, 255).round().astype(int)
    patterned = rng.random(n) < 0.2
    busy = np.where(patterned, rng.beta(4, 2, n), rng.beta(1.2, 6, n))
    return rgb, np.clip(busy, 0, 1)


def generate_outfit_pairs(n=6000, seed=RANDOM_SEED, path=OUTFIT_CSV) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols, mats = {}, []
    for s in SLOTS:
        rgb, busy = _sample_items(rng, n)
        cols[f"{s}_color"] = [name_color(c) for c in rgb]
        for i, ch in enumerate("rgb"):
            cols[f"{s}_{ch}"] = rgb[:, i]
        cols[f"{s}_busy"] = busy.round(3)
        mats.append(item_matrix(rgb, busy))
    latent = latent_compatibility(*mats) + rng.normal(0, NOISE_SD, n)
    cols["label"] = (latent >= THRESHOLD).astype(int)
    df = pd.DataFrame(cols)
    df = df[[f"{s}_{k}" for s in SLOTS for k in ("color", "r", "g", "b", "busy")] + ["label"]]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df
