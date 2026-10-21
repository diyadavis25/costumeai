"""Transparent fashion rules (colour harmony + visual balance).

Used (a) to generate the synthetic training labels, and (b) to explain predictions.
All functions work on item matrices from feature_engineering.item_matrix
(columns: sin_h, cos_h, sat, val, neutral, busy). Application-defined heuristics,
not universal fashion standards.
"""
import numpy as np


def hue_diff_deg(a, b):
    c = np.clip(a[:, 1] * b[:, 1] + a[:, 0] * b[:, 0], -1.0, 1.0)
    return np.degrees(np.arccos(c))


def pair_harmony(a, b):
    d = hue_diff_deg(a, b)
    chroma = np.select([d <= 30, d <= 60, d < 105, d < 150], [0.88, 0.62, 0.28, 0.50], default=0.72)
    sat_prod = a[:, 2] * b[:, 2]
    chroma = chroma + (1 - sat_prod) * (0.80 - chroma) * 0.6   # muted colours clash less
    neutral_either = (a[:, 4] + b[:, 4]) > 0
    return np.clip(np.where(neutral_either, 0.86, chroma), 0, 1)


def pair_scores(T, B, S):
    return pair_harmony(T, B), pair_harmony(T, S), pair_harmony(B, S)


def color_harmony(T, B, S):
    tb, ts, bs = pair_scores(T, B, S)
    n_chroma = 3 - (T[:, 4] + B[:, 4] + S[:, 4])
    h = (1.2 * tb + 0.8 * ts + 1.0 * bs) / 3.0 - 0.12 * (n_chroma >= 3)
    return np.clip(h, 0, 1)


def visual_balance(T, B, S):
    busy = T[:, 5] + B[:, 5] + S[:, 5]
    contrast = np.abs(T[:, 3] - B[:, 3])
    bal = 0.70 + 0.25 * np.minimum(contrast, 0.5) / 0.5
    bal = bal - 0.5 * np.clip(busy - 0.8, 0, None) - 0.10 * (S[:, 5] > 0.6)
    return np.clip(bal, 0, 1)


def latent_compatibility(T, B, S):
    return 0.7 * color_harmony(T, B, S) + 0.3 * visual_balance(T, B, S)
