"""Member 1 - dominant colour extraction (OpenCV K-Means) and colour naming."""
import colorsys

import cv2
import numpy as np

from .config import hex_color


def rgb_to_hsv(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s, v


def name_color(rgb) -> str:
    """Map an RGB triple to one of the palette colour names (rule based)."""
    h, s, v = rgb_to_hsv(rgb)
    if v < 0.20:
        return "black"
    if s < 0.13:
        return "white" if v > 0.86 else "grey"
    if 20 <= h <= 55 and s < 0.40 and v >= 0.70:
        return "beige"
    if 10 <= h <= 50 and s >= 0.35 and v < 0.55:
        return "brown"
    if 200 <= h <= 260 and v < 0.42:
        return "navy"
    if 50 <= h <= 95 and v < 0.60:
        return "olive"
    if h >= 345 or h < 15:
        return "pink" if (s < 0.60 and v >= 0.75) else "red"
    if h < 40:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 255:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def foreground_pixels(arr: np.ndarray):
    """Return (pixels, mask). Removes a uniform background when one is detected,
    otherwise falls back to the central region of the image."""
    h, w, _ = arr.shape
    bw = max(2, int(0.06 * min(h, w)))
    border = np.concatenate([
        arr[:bw].reshape(-1, 3), arr[-bw:].reshape(-1, 3),
        arr[:, :bw].reshape(-1, 3), arr[:, -bw:].reshape(-1, 3)]).astype(np.float32)
    if border.std(axis=0).mean() < 22:
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(arr.astype(np.float32) - bg, axis=2)
        mask = dist > 28
        if mask.mean() > 0.05:
            return arr[mask], mask
    ch, cw = int(h * 0.25), int(w * 0.25)
    mask = np.zeros((h, w), bool)
    mask[ch:h - ch, cw:w - cw] = True
    return arr[mask], mask


def dominant_colors(pixels: np.ndarray, k: int = 3):
    """K-Means (OpenCV) on foreground pixels -> list of {rgb, fraction}, largest first."""
    if len(pixels) == 0:
        return [{"rgb": (128, 128, 128), "fraction": 1.0}]
    rng = np.random.default_rng(0)
    if len(pixels) > 5000:
        pixels = pixels[rng.choice(len(pixels), 5000, replace=False)]
    data = np.float32(pixels)
    k = max(1, min(k, len(np.unique(pixels, axis=0))))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    return [{"rgb": tuple(int(round(c)) for c in centers[i]),
             "fraction": float(counts[i] / counts.sum())} for i in order]


def busyness(arr: np.ndarray, mask: np.ndarray) -> float:
    """0..1 'visual busyness' (patterns/prints) = interior edge density."""
    gray = cv2.GaussianBlur(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150) > 0
    inner = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if inner.sum() < 50:
        return 0.0
    return float(np.clip(edges[inner].mean() / 0.10, 0.0, 1.0))


def color_info(img) -> dict:
    """Full colour analysis of a PIL image."""
    arr = np.asarray(img.convert("RGB"))
    pixels, mask = foreground_pixels(arr)
    clusters = dominant_colors(pixels, 3)
    main = clusters[0]["rgb"]
    palette = []
    for c in clusters:
        palette.append({"name": name_color(c["rgb"]), "rgb": c["rgb"],
                        "hex": hex_color(c["rgb"]), "fraction": round(c["fraction"], 3)})
    return {"color": name_color(main), "rgb": main, "hex": hex_color(main),
            "palette": palette, "busyness": round(busyness(arr, mask), 3)}
