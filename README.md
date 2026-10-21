# Outfit AI - Outfit Compatibility Smart Mirror

Upload or capture your **top, bottom and shoes**. The app reads colour and visual features, scores how well the
outfit works (0-100), explains the score, suggests what to change, and ranks the best outfits from a whole wardrobe.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py                                # or ./run_app.sh  /  run_app.bat
```

The trained compatibility model (`models/compatibility_model.pkl`) and metrics ship in the zip. To rebuild:
`python build_all.py`. Camera capture works on `localhost`; over a network it needs HTTPS (browser rule).

## App pages
| Page | What it does |
|---|---|
| Home | Hero, demo outfit button, system status |
| Analyze Outfit | Per slot: **Upload**, **Camera**, or **Sample**; run analysis |
| AI Analysis | Detected colour/category per piece, score ring, breakdown bars, colour correction, 👍/👎 feedback |
| Recommendations | Best alternative colour per slot with score gain, one-click Apply |
| My Wardrobe | Multi-upload / camera / sample wardrobe; scores every top x bottom x shoes combination, shows the best N |
| Model Performance | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC, model comparison, feature importance |
| About | Team roles, pipeline, limitations |

## Architecture
```
image -> preprocessing -> [ResNet18 512-d features | K-Means colour | category] -> analyze_clothing()   (Member 1)
      -> feature engineering -> Logistic Regression / Random Forest / XGBoost -> predict_compatibility() (Member 2)
      -> alternatives + wardrobe ranking + Streamlit dashboard                                          (Member 3)
```
Key functions: `analyze_clothing(image)`, `predict_compatibility(top, bottom, shoes)`,
`recommend_alternatives(...)`, `rank_wardrobe(...)`. Pass the dicts from `analyze_clothing` straight into `predict_compatibility`.

## Results (held-out test set, 900 outfits)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.812 | 0.789 | 0.841 | 0.814 | 0.899 |
| Random Forest | 0.834 | 0.798 | 0.886 | 0.840 | 0.911 |
| Gradient Boosting (XGBoost not installed) | 0.831 | 0.814 | 0.848 | 0.831 | 0.915 |

Selection rule: highest validation F1. Majority-class baseline accuracy: ~51%.
Reproduce with `python -m training.evaluate_model`.

## Read this before presenting (honest limitations)
1. **Compatibility labels are synthetic.** `data/outfit_pairs.csv` (6,000 outfits) is generated from transparent
   colour-harmony / visual-balance rules (`src/style_rules.py`) plus label noise. The models learn that rule set; the metrics measure
   that, **not** agreement with human taste. For a stronger claim, add real labelled outfits to the CSV, or use the in-app 👍/👎 votes
   (saved to `data/user_feedback.csv` and merged automatically by the next training run).
2. **XGBoost wasn't installed in the build environment**, so the shipped comparison used scikit-learn gradient boosting in its place
   (labelled in the output). `pip install xgboost` then `python -m training.train_compatibility` runs real XGBoost.
3. **The ResNet18 classifier is not pre-trained** (no PyTorch/GPU/dataset here). Until you train it, the category comes from the slot
   the user uploaded into and embeddings use a handcrafted 512-d descriptor. Steps: `python -m training.prepare_dataset fashion-mnist`
   then `python -m training.train_classifier --epochs 5`. Once `models/clothing_classifier.pth` exists, the app uses ResNet18 automatically.
   Fashion-MNIST is only for initial testing (28x28 grey); use DeepFashion2 + Fashionpedia crops (converters included, untested against the real downloads) for real photos.
4. The Streamlit UI was smoke-tested with a stub (all pages execute) but not rendered in a real browser here; expect to tweak CSS.
5. Score bands (80 / 65 / 50) are application-defined. "Visual Similarity" is embedding cosine similarity and is informational only.

## Repo map
`src/` core logic | `training/` dataset + training scripts | `app/` Streamlit UI | `models/` model, metrics, report PNGs |
`data/` outfit_pairs.csv | `tests/` pytest | `docs/` team guide | `presentation/` talking points | `notebooks/` demo
