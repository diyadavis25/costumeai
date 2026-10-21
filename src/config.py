"""Central configuration: paths, palette, categories, score bands."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = MODELS_DIR / "reports"

CLASSIFIER_PATH = MODELS_DIR / "clothing_classifier.pth"
COMPAT_MODEL_PATH = MODELS_DIR / "compatibility_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
OUTFIT_CSV = DATA_DIR / "outfit_pairs.csv"
FEEDBACK_CSV = DATA_DIR / "user_feedback.csv"

RANDOM_SEED = 42
EMBED_DIM = 512

CATEGORIES = ["top", "bottom", "dress", "outerwear", "shoes", "accessory"]
SLOTS = ("top", "bottom", "shoes")
SLOT_TITLE = {"top": "Top", "bottom": "Bottom", "shoes": "Shoes"}
SLOT_NOUN = {"top": "Shirt", "bottom": "Pants", "shoes": "Shoes"}

# Canonical RGB for each named colour (used for naming, demos and alternatives)
PALETTE = {
    "black": (20, 20, 20), "white": (245, 245, 245), "grey": (128, 128, 128),
    "beige": (222, 205, 170), "brown": (101, 67, 33), "navy": (25, 35, 90),
    "blue": (40, 90, 200), "red": (200, 30, 40), "orange": (240, 130, 30),
    "yellow": (245, 215, 50), "green": (40, 150, 70), "olive": (110, 115, 40),
    "purple": (120, 50, 160), "pink": (245, 150, 185),
}
NEUTRAL_COLORS = {"black", "white", "grey", "beige", "navy", "brown"}

# Application-defined bands (not universal fashion standards)
SCORE_BANDS = [(80, "Excellent Match"), (65, "Good Match"), (50, "Moderate Match"), (0, "Low Match")]


def score_label(score: float) -> str:
    for lo, name in SCORE_BANDS:
        if score >= lo:
            return name
    return "Low Match"


def hex_color(rgb) -> str:
    r, g, b = (int(x) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"
