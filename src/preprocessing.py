"""Member 1 - image loading and preprocessing."""
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(src, max_side: int = 768) -> Image.Image:
    """Load a path / bytes / file-like / PIL image into an RGB PIL image.

    Transparent PNG cut-outs are composited on white. EXIF rotation is applied.
    """
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    elif hasattr(src, "getvalue"):
        img = Image.open(io.BytesIO(src.getvalue()))
    elif hasattr(src, "read"):
        img = Image.open(io.BytesIO(src.read()))
    else:
        img = Image.open(Path(src))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def preprocess_for_cnn(img: Image.Image, size: int = 224) -> np.ndarray:
    """PIL -> normalised float32 CHW array (ImageNet statistics)."""
    arr = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).astype(np.float32)


def image_to_bytes(img: Image.Image, max_side: int = 360) -> bytes:
    """Small PNG thumbnail for storing in session state."""
    t = img.copy()
    t.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    t.save(buf, format="PNG")
    return buf.getvalue()
