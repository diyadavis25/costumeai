# Team guide

| Member | Owns | Files | Say to judges |
|---|---|---|---|
| 1 CV | dataset prep, ResNet18 classifier, K-Means colour, 512-d features, `analyze_clothing()` | `training/prepare_dataset.py`, `training/train_classifier.py`, `src/preprocessing.py`, `src/color_features.py`, `src/feature_extraction.py`, `src/clothing_classifier.py`, `src/clothing_analysis.py` | "I built the vision pipeline: category, dominant colour and CNN features from a photo, using transfer learning and OpenCV." |
| 2 ML | outfit dataset, feature engineering, 3-model comparison, evaluation | `src/synthetic_data.py`, `src/feature_engineering.py`, `src/style_rules.py`, `src/model_training.py`, `src/compatibility.py`, `training/train_compatibility.py`, `training/evaluate_model.py` | "I compared Logistic Regression, Random Forest and boosting and chose by validation F1. Labels are rule-based, which I can explain and how to replace them." |
| 3 Product | recommendations, wardrobe ranking, dashboard | `src/recommendation.py`, `src/feedback.py`, `app/` | "I built the engine that tests alternatives and ranks wardrobe combinations, and the Streamlit app." |

## Interfaces
- M1 -> M2: dict from `analyze_clothing()` with `category, color, rgb, busyness, features[512], confidence`.
- M2 -> M3: `predict_compatibility(top, bottom, shoes)` -> `{score, label, breakdown, tips}`; batch API `CompatibilityModel.proba_items(...)`.
- M3 -> M2: candidate outfits scored through the same model, sorted and displayed.

## Questions judges may ask
- *Where do labels come from?* Rule-based synthetic + noise; swap in real labels or app votes.
- *Why does the score change when I fix the colour?* The model scores colour/contrast/busyness features, so correcting the colour changes them.
- *Why scores like 0-100?* Predicted probability x 100; bands are our own thresholds.
