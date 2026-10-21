`outfit_pairs.csv` - 6,000 synthetic labelled outfits (rule-based labels + noise). Columns: `<slot>_color,_r,_g,_b,_busy` for top/bottom/shoes and `label` (1 compatible).
`user_feedback.csv` - created by the app's 👍/👎 buttons; merged into training automatically.
`clothing/` - created by `training/prepare_dataset.py` (train/validation/test folders).
