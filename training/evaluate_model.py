"""Member 2: python -m training.evaluate_model
Re-evaluates the saved model on the held-out test split and writes report PNGs to models/reports/."""
import json

import joblib

from src import reporting
from src.config import COMPAT_MODEL_PATH, METRICS_PATH, REPORTS_DIR
from src.feature_engineering import features_from_dataframe, split_data
from src.model_training import _scores, load_training_frame

if __name__ == "__main__":
    metrics = json.loads(METRICS_PATH.read_text())
    bundle = joblib.load(COMPAT_MODEL_PATH)
    df, _ = load_training_frame()
    _, _, test = split_data(df)
    m, _ = _scores(bundle["model"], features_from_dataframe(test), test["label"].to_numpy())
    print(f"Selected model: {bundle['name']}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:<10} {m[k]:.4f}")
    print("  confusion  ", m["confusion_matrix"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in (("confusion_matrix", reporting.confusion_fig(metrics)),
                      ("model_comparison", reporting.comparison_fig(metrics)),
                      ("roc_curves", reporting.roc_fig(metrics)),
                      ("feature_importance", reporting.importance_fig(metrics))):
        fig.savefig(REPORTS_DIR / f"{name}.png", dpi=160)
    print("Saved plots to", REPORTS_DIR)
