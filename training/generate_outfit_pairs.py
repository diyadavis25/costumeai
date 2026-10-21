"""Member 2: python -m training.generate_outfit_pairs --rows 6000"""
import argparse

from src.synthetic_data import generate_outfit_pairs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    a = ap.parse_args()
    df = generate_outfit_pairs(a.rows)
    print(f"Wrote data/outfit_pairs.csv  rows={len(df)}  compatible={df['label'].mean():.1%}")
