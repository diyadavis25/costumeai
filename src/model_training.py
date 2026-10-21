"""Member 2 - train / compare / select the compatibility model and write metrics."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (COMPAT_MODEL_PATH, FEEDBACK_CSV, METRICS_PATH, MODELS_DIR, OUTFIT_CSV,
                     RANDOM_SEED)
from .feature_engineering import FEATURE_NAMES, features_from_dataframe, split_data
from .synthetic_data import generate_outfit_pairs


def build_models(seed=RANDOM_SEED):
    models = {
        "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=3000))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                n_jobs=-1, random_state=seed),
    }
    xgb_available = True
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                          subsample=0.9, colsample_bytree=0.9,
                                          eval_metric="logloss", random_state=seed, n_jobs=-1)
    except Exception:
        xgb_available = False
        models["Gradient Boosting (XGBoost not installed)"] = HistGradientBoostingClassifier(random_state=seed)
    return models, xgb_available


def _scores(model, X, y):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }, p


def load_training_frame(n_rows=6000, regenerate=False):
    if regenerate or not OUTFIT_CSV.exists():
        generate_outfit_pairs(n_rows)
    df = pd.read_csv(OUTFIT_CSV)
    n_feedback = 0
    if FEEDBACK_CSV.exists():
        fb = pd.read_csv(FEEDBACK_CSV)
        n_feedback = len(fb)
        df = pd.concat([df, fb.reindex(columns=df.columns)], ignore_index=True)
    return df, n_feedback


def train_and_save(n_rows=6000, regenerate=False, verbose=True):
    t0 = time.time()
    df, n_feedback = load_training_frame(n_rows, regenerate)
    train, val, test = split_data(df)
    Xtr, ytr = features_from_dataframe(train), train["label"].to_numpy()
    Xva, yva = features_from_dataframe(val), val["label"].to_numpy()
    Xte, yte = features_from_dataframe(test), test["label"].to_numpy()

    models, xgb_ok = build_models()
    report, fitted = {}, {}
    base_pred_rate = float(max(ytr.mean(), 1 - ytr.mean()))
    for name, m in models.items():
        m.fit(Xtr, ytr)
        val_m, _ = _scores(m, Xva, yva)
        test_m, p_te = _scores(m, Xte, yte)
        fpr, tpr, _ = roc_curve(yte, p_te)
        keep = np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int)
        report[name] = {"validation": val_m, "test": test_m,
                        "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}}
        fitted[name] = m
        if verbose:
            print(f"{name:<45} val F1={val_m['f1']:.3f}  test acc={test_m['accuracy']:.3f}  AUC={test_m['roc_auc']:.3f}")

    best = max(report, key=lambda k: (report[k]["validation"]["f1"], report[k]["validation"]["roc_auc"]))
    best_model = fitted[best]
    pi = permutation_importance(best_model, Xva, yva, n_repeats=5, random_state=RANDOM_SEED, scoring="f1")
    order = np.argsort(-pi.importances_mean)[:12]
    importance = [{"feature": FEATURE_NAMES[i], "importance": float(pi.importances_mean[i])} for i in order]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "name": best, "feature_names": FEATURE_NAMES,
                 "n_train": int(len(train))}, COMPAT_MODEL_PATH)
    metrics = {
        "selected_model": best,
        "selection_rule": "highest validation F1 (ties broken by validation ROC-AUC)",
        "models": report,
        "baseline_majority_accuracy": base_pred_rate,
        "feature_importance": importance,
        "dataset": {"rows_total": int(len(df)), "train": int(len(train)), "validation": int(len(val)),
                    "test": int(len(test)), "positive_rate": float(df["label"].mean()),
                    "user_feedback_rows": int(n_feedback),
                    "label_source": "rule-based synthetic labels + noise (+ user feedback if present)"},
        "xgboost_available": xgb_ok,
        "n_features": len(FEATURE_NAMES),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"Selected: {best}  -> saved {COMPAT_MODEL_PATH.name}, {METRICS_PATH.name}")
    return metrics
