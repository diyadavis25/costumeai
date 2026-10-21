"""Member 2 - combine item descriptors into one outfit feature vector."""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .color_features import name_color, rgb_to_hsv
from .config import NEUTRAL_COLORS, RANDOM_SEED, SLOTS

ITEM_COLS = ["sin_h", "cos_h", "sat", "val", "neutral", "busy"]
PAIR_COLS = ["hue_diff", "val_contrast", "sat_product", "both_neutral"]
FEATURE_NAMES = (
    [f"{s}_{c}" for s in SLOTS for c in ITEM_COLS]
    + [f"{p}_{c}" for p in ("top_bottom", "top_shoes", "bottom_shoes") for c in PAIR_COLS]
    + ["n_chromatic", "total_busy", "total_sat"]
)


@lru_cache(maxsize=None)
def _color_row(r, g, b):
    h, s, v = rgb_to_hsv((r, g, b))
    rad = np.radians(h)
    neutral = float(name_color((r, g, b)) in NEUTRAL_COLORS)
    return (float(np.sin(rad)), float(np.cos(rad)), s, v, neutral)


def item_matrix(rgbs, busy) -> np.ndarray:
    """(n,3) RGB + (n,) busyness -> (n,6) item matrix."""
    rows = [_color_row(int(r), int(g), int(b)) for r, g, b in rgbs]
    return np.column_stack([np.array(rows, dtype=np.float32), np.asarray(busy, dtype=np.float32)])


def _pair_block(a, b):
    c = np.clip(a[:, 1] * b[:, 1] + a[:, 0] * b[:, 0], -1, 1)
    return np.column_stack([np.degrees(np.arccos(c)) / 180.0, np.abs(a[:, 3] - b[:, 3]),
                            a[:, 2] * b[:, 2], a[:, 4] * b[:, 4]])


def outfit_features(T, B, S) -> np.ndarray:
    chroma = 3 - (T[:, 4] + B[:, 4] + S[:, 4])
    extras = np.column_stack([chroma, T[:, 5] + B[:, 5] + S[:, 5], T[:, 2] + B[:, 2] + S[:, 2]])
    return np.hstack([T, B, S, _pair_block(T, B), _pair_block(T, S), _pair_block(B, S), extras]).astype(np.float32)


def matrices_from_dataframe(df: pd.DataFrame):
    mats = []
    for s in SLOTS:
        rgb = df[[f"{s}_r", f"{s}_g", f"{s}_b"]].to_numpy()
        mats.append(item_matrix(rgb, df[f"{s}_busy"].to_numpy()))
    return mats


def features_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    return outfit_features(*matrices_from_dataframe(df))


def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Stratified 70 / 15 / 15 train / validation / test split."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train, val, test
