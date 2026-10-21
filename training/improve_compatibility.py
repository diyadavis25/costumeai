"""Improved compatibility training.

    python -m training.improve_compatibility                 # tune + ensemble + calibrate, save model
    python -m training.improve_compatibility --quick         # faster, smaller search
    python -m training.improve_compatibility --no-save       # experiment without overwriting the shipped model
"""
import argparse
import json
import shutil
import time
import warnings

import joblib
import numpy as np
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import COMPAT_MODEL_PATH, METRICS_PATH, MODELS_DIR, RANDOM_SEED
from src.extra_features import add_extra_features
from src.feature_engineering import FEATURE_NAMES, features_from_dataframe, split_data
from src.model_training import _scores, load_training_frame

warnings.filterwarnings("ignore")
SEED = RANDOM_SEED


def wrap(clf, scale=False):
    """Pipeline: original 33 features -> extra features -> (scaler) -> classifier."""
    steps = [("extra", FunctionTransformer(add_extra_features))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def candidate_models(quick):
    n = 8 if quick else 20
    c = {
        "Logistic Regression": (wrap(LogisticRegression(max_iter=5000), scale=True),
                                {"clf__C": uniform(0.05, 20)}, n),
        "Random Forest": (wrap(RandomForestClassifier(n_jobs=-1, random_state=SEED)),
                          {"clf__n_estimators": randint(200, 600), "clf__max_depth": [None, 10, 16, 24],
                           "clf__min_samples_leaf": randint(1, 6), "clf__max_features": ["sqrt", 0.4, 0.6]}, n),
        "Extra Trees": (wrap(ExtraTreesClassifier(n_jobs=-1, random_state=SEED)),
                        {"clf__n_estimators": randint(200, 600), "clf__max_depth": [None, 12, 20],
                         "clf__min_samples_leaf": randint(1, 5), "clf__max_features": ["sqrt", 0.5, 0.7]}, n),
        "Gradient Boosting": (wrap(HistGradientBoostingClassifier(random_state=SEED)),
                              {"clf__learning_rate": uniform(0.02, 0.15), "clf__max_depth": [3, 4, 6, None],
                               "clf__max_leaf_nodes": randint(15, 64), "clf__l2_regularization": uniform(0, 2),
                               "clf__min_samples_leaf": randint(10, 60)}, n),
        "Neural Net (MLP)": (wrap(MLPClassifier(max_iter=600, early_stopping=True, random_state=SEED), scale=True),
                             {"clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                              "clf__alpha": uniform(1e-5, 1e-2), "clf__learning_rate_init": uniform(5e-4, 5e-3)}, n),
    }
    try:
        from xgboost import XGBClassifier
        c["XGBoost"] = (wrap(XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=SEED)),
                        {"clf__n_estimators": randint(200, 600), "clf__max_depth": randint(3, 8),
                         "clf__learning_rate": uniform(0.02, 0.15), "clf__subsample": uniform(0.7, 0.3),
                         "clf__colsample_bytree": uniform(0.6, 0.4)}, n)
    except Exception:
        print("xgboost not installed -> skipped (pip install xgboost)")
    try:
        from lightgbm import LGBMClassifier
        c["LightGBM"] = (wrap(LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)),
                         {"clf__n_estimators": randint(200, 600), "clf__num_leaves": randint(15, 64),
                          "clf__learning_rate": uniform(0.02, 0.15), "clf__subsample": uniform(0.7, 0.3),
                          "clf__colsample_bytree": uniform(0.6, 0.4)}, n)
    except Exception:
        print("lightgbm not installed -> skipped (pip install lightgbm)")
    return c


def noise_ceiling():
    """Best accuracy any model could reach on the synthetic data (rule labels vs noisy labels)."""
    try:
        import pandas as pd
        from src.config import OUTFIT_CSV
        from src.feature_engineering import matrices_from_dataframe
        from src.style_rules import latent_compatibility
        from src.synthetic_data import THRESHOLD
        df = pd.read_csv(OUTFIT_CSV)
        pred = (latent_compatibility(*matrices_from_dataframe(df)) >= THRESHOLD).astype(int)
        return float((pred == df["label"].to_numpy()).mean())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--cv", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()

    df, n_fb = load_training_frame()
    train, val, test = split_data(df)
    Xtr, ytr = features_from_dataframe(train), train["label"].to_numpy()
    Xva, yva = features_from_dataframe(val), val["label"].to_numpy()
    Xte, yte = features_from_dataframe(test), test["label"].to_numpy()
    print(f"rows: train {len(train)}  val {len(val)}  test {len(test)}  (user feedback rows: {n_fb})")

    skf = StratifiedKFold(a.cv, shuffle=True, random_state=SEED)
    tuned, report = {}, {}
    for name, (pipe, space, n_iter) in candidate_models(a.quick).items():
        s = time.time()
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="f1", cv=skf,
                                    random_state=SEED, n_jobs=1, refit=True)
        search.fit(Xtr, ytr)
        tuned[name] = search.best_estimator_
        print(f"{name:<22} cv F1={search.best_score_:.3f}  ({time.time() - s:.0f}s)  {search.best_params_}")

    # soft-voting ensemble of the 3 best tuned models (by CV F1 on train only), then calibrate
    ranked = sorted(tuned, key=lambda k: -cross_val_score(tuned[k], Xtr, ytr, cv=3, scoring="f1").mean())
    top3 = ranked[:3]
    ens = VotingClassifier([(n, tuned[n]) for n in top3], voting="soft", n_jobs=1)
    ens.fit(Xtr, ytr)
    tuned[f"Ensemble ({' + '.join(top3)})"] = ens
    cal = CalibratedClassifierCV(VotingClassifier([(n, tuned[n]) for n in top3], voting="soft"),
                                 method="isotonic", cv=5)
    cal.fit(Xtr, ytr)
    tuned["Calibrated Ensemble"] = cal

    base_rate = float(max(ytr.mean(), 1 - ytr.mean()))
    for name, m in tuned.items():
        v, _ = _scores(m, Xva, yva)
        t, p = _scores(m, Xte, yte)
        t["brier"] = float(brier_score_loss(yte, p))
        fpr, tpr, _ = roc_curve(yte, p)
        keep = np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int)
        report[name] = {"validation": v, "test": t,
                        "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}}
        print(f"{name[:46]:<46} val F1={v['f1']:.3f}  test acc={t['accuracy']:.3f}  "
              f"F1={t['f1']:.3f}  AUC={t['roc_auc']:.3f}  Brier={t['brier']:.3f}")

    best = max(report, key=lambda k: (report[k]["validation"]["f1"], report[k]["validation"]["roc_auc"]))
    model = tuned[best]
    print(f"\nSelected (highest validation F1): {best}")

    # honest generalisation estimate: CV over train+val for the winner (test stays untouched)
    Xall, yall = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
    cv_f1 = cross_val_score(model, Xall, yall, cv=a.cv, scoring="f1")
    cv_acc = cross_val_score(model, Xall, yall, cv=a.cv, scoring="accuracy")
    print(f"{a.cv}-fold CV  accuracy {cv_acc.mean():.3f} +- {cv_acc.std():.3f}   F1 {cv_f1.mean():.3f} +- {cv_f1.std():.3f}")
    ceiling = noise_ceiling()
    if ceiling:
        print(f"Noise ceiling on the synthetic labels: ~{ceiling:.3f} accuracy (no model can beat this honestly)")

    pi = permutation_importance(model, Xva, yva, n_repeats=5, random_state=SEED, scoring="f1")
    order = np.argsort(-pi.importances_mean)[:12]
    importance = [{"feature": FEATURE_NAMES[i], "importance": float(pi.importances_mean[i])} for i in order]

    if a.no_save:
        print("--no-save: model not written.")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if COMPAT_MODEL_PATH.exists():
        shutil.copy(COMPAT_MODEL_PATH, COMPAT_MODEL_PATH.with_suffix(".pkl.bak"))
    if METRICS_PATH.exists():
        shutil.copy(METRICS_PATH, METRICS_PATH.with_suffix(".json.bak"))
    joblib.dump({"model": model, "name": best, "feature_names": FEATURE_NAMES, "n_train": int(len(train))},
                COMPAT_MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "selected_model": best,
        "selection_rule": "highest validation F1 (ties broken by validation ROC-AUC); tuned with 5-fold CV on train",
        "models": report,
        "baseline_majority_accuracy": base_rate,
        "feature_importance": importance,
        "cross_validation": {"folds": a.cv, "accuracy_mean": float(cv_acc.mean()), "accuracy_std": float(cv_acc.std()),
                             "f1_mean": float(cv_f1.mean()), "f1_std": float(cv_f1.std())},
        "noise_ceiling_accuracy": ceiling,
        "dataset": {"rows_total": int(len(df)), "train": int(len(train)), "validation": int(len(val)),
                    "test": int(len(test)), "positive_rate": float(df["label"].mean()),
                    "user_feedback_rows": int(n_fb),
                    "label_source": "rule-based synthetic labels + noise (+ user feedback if present)"},
        "xgboost_available": "XGBoost" in tuned,
        "n_features": len(FEATURE_NAMES),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"Saved {COMPAT_MODEL_PATH.name} (old one backed up as .pkl.bak) and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()