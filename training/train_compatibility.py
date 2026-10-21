"""Member 2: python -m training.train_compatibility [--regenerate --rows 6000]"""
import argparse

from src.model_training import train_and_save

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--regenerate", action="store_true", help="rebuild outfit_pairs.csv first")
    a = ap.parse_args()
    train_and_save(a.rows, a.regenerate)
