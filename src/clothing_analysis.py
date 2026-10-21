"""Member 1 deliverable: analyze_clothing(image)."""
from .clothing_classifier import get_classifier
from .color_features import color_info
from .feature_extraction import get_extractor
from .preprocessing import load_image


def analyze_clothing(image, slot_hint=None) -> dict:
    """Analyse one garment image.

    Returns {"category","color","rgb","hex","palette","busyness","features","confidence",...}.
    If no trained classifier is available, the category falls back to `slot_hint`
    (the slot the user uploaded into) and `category_source` says so.
    """
    img = load_image(image)
    info = color_info(img)
    extractor = get_extractor()
    emb = extractor.embed(img)

    pred = get_classifier().predict(img)
    if pred is not None:
        category, confidence, probs = pred
        source = "classifier"
    else:
        category, confidence, probs, source = (slot_hint or "unknown"), None, {}, "slot-hint"

    return {
        "category": category,
        "category_source": source,
        "confidence": None if confidence is None else round(confidence, 4),
        "class_probs": probs,
        "color": info["color"],
        "rgb": info["rgb"],
        "hex": info["hex"],
        "palette": info["palette"],
        "busyness": info["busyness"],
        "features": emb.tolist(),
        "embedding_backend": extractor.backend,
    }
