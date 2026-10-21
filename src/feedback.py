"""Store in-app thumbs-up/down as extra labelled rows (used by train_compatibility)."""
import pandas as pd

from .config import FEEDBACK_CSV, SLOTS


def save_feedback(items: dict, liked: bool) -> int:
    row = {}
    for s in SLOTS:
        it = items[s]
        r, g, b = it["rgb"]
        row.update({f"{s}_color": it["color"], f"{s}_r": int(r), f"{s}_g": int(g), f"{s}_b": int(b),
                    f"{s}_busy": float(it.get("busyness", 0.2))})
    row["label"] = int(liked)
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(FEEDBACK_CSV, mode="a", header=not FEEDBACK_CSV.exists(), index=False)
    return feedback_count()


def feedback_count() -> int:
    if not FEEDBACK_CSV.exists():
        return 0
    return max(0, sum(1 for _ in open(FEEDBACK_CSV)) - 1)
