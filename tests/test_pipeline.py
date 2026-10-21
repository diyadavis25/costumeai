"""Run:  python -m pytest -q   (or)   python tests/test_pipeline.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clothing_analysis import analyze_clothing
from src.color_features import name_color
from src.compatibility import predict_compatibility
from src.config import PALETTE, SLOTS
from src.recommendation import rank_wardrobe, recommend_alternatives
from src.sample_garments import garment_bytes


def test_color_naming_roundtrip():
    for name, rgb in PALETTE.items():
        assert name_color(rgb) == name


def test_color_extraction_on_garments():
    for slot in SLOTS:
        for name in PALETTE:
            assert analyze_clothing(garment_bytes(slot, name), slot)["color"] == name


def test_pattern_is_busier():
    plain = analyze_clothing(garment_bytes("top", "blue"), "top")["busyness"]
    striped = analyze_clothing(garment_bytes("top", "blue", True), "top")["busyness"]
    assert striped > plain + 0.3


def test_output_contract():
    a = analyze_clothing(garment_bytes("top", "white"), "top")
    assert {"category", "color", "features", "confidence"} <= set(a)
    assert len(a["features"]) == 512


def _outfit(t, b, s):
    return [analyze_clothing(garment_bytes(sl, c), sl) for sl, c in zip(SLOTS, (t, b, s))]


def test_compatibility_prefers_neutral_shoes():
    good = predict_compatibility(*_outfit("white", "blue", "white"))
    bad = predict_compatibility(*_outfit("red", "green", "orange"))
    assert good["score"] > bad["score"]
    assert good["label"] in {"Excellent Match", "Good Match", "Moderate Match", "Low Match"}


def test_recommendations_and_wardrobe():
    t, b, s = _outfit("white", "blue", "red")
    rec = recommend_alternatives(t, b, s)
    assert rec["slots"]["shoes"][0]["score"] >= rec["current_score"]
    wardrobe = {sl: [analyze_clothing(garment_bytes(sl, c), sl) for c in ("white", "black", "beige")] for sl in SLOTS}
    best, scores = rank_wardrobe(wardrobe, top_n=5)
    assert len(scores) == 27 and best[0]["score"] >= best[-1]["score"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
