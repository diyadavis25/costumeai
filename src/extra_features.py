"""Extra engineered features, applied INSIDE the saved model pipeline.

Because the transform lives in the pipeline, the app keeps feeding the original 33 features
and nothing else in the app needs to change.
"""
import numpy as np

from .feature_engineering import FEATURE_NAMES

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_PAIRS = ("top_bottom", "top_shoes", "bottom_shoes")


def add_extra_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    col = lambda n: X[:, _IDX[n]]
    hue = np.column_stack([col(f"{p}_hue_diff") for p in _PAIRS])
    val = np.column_stack([col(f"{p}_val_contrast") for p in _PAIRS])
    sat = np.column_stack([col(f"{s}_sat") for s in ("top", "bottom", "shoes")])
    busy = np.column_stack([col(f"{s}_busy") for s in ("top", "bottom", "shoes")])
    neu = np.column_stack([col(f"{s}_neutral") for s in ("top", "bottom", "shoes")])
    pair_neu = np.column_stack([col(f"{p}_both_neutral") for p in _PAIRS])
    chroma = 1.0 - neu
    # near-complementary (hue_diff ~ 0.5) and near-analogous (hue_diff ~ 0) closeness
    comp = 1.0 - np.abs(hue - 0.5) * 2.0
    analog = 1.0 - hue
    extra = np.column_stack([
        hue.min(1), hue.max(1), hue.mean(1),
        val.min(1), val.max(1), val.mean(1),
        sat.max(1), sat.mean(1), sat.std(1),
        busy.max(1), busy.std(1), (busy > 0.5).sum(1),
        neu.sum(1), pair_neu.sum(1),
        comp.max(1), analog.max(1),
        # chromatic pieces clashing: both coloured AND mid hue distance
        (chroma[:, 0] * chroma[:, 1] * (1 - np.abs(hue[:, 0] - 0.33) * 3).clip(0, 1)),
        # sat of each chromatic piece weighted by busyness
        (sat * busy * chroma).sum(1),
        # top vs bottom brightness contrast times neutral count (classic safe look)
        val[:, 0] * neu.sum(1),
    ])
    return np.hstack([X, extra]).astype(np.float32)