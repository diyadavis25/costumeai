"""Member 1 - 512-d visual embeddings.

Primary: ResNet18 with the final classification layer removed (PyTorch).
Fallback (no PyTorch / no weights): handcrafted 512-d descriptor (HSV histogram +
spatial colour grid + gradient-orientation grid). `backend` tells which one is used.
"""
from functools import lru_cache

import cv2
import numpy as np

from .config import CLASSIFIER_PATH, EMBED_DIM
from .preprocessing import preprocess_for_cnn


def handcrafted_embedding(img) -> np.ndarray:
    arr = np.asarray(img.convert("RGB").resize((128, 128)))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)  # 256
    grid = []
    for i in range(4):
        for j in range(4):
            cell = hsv[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32].reshape(-1, 3).mean(0)
            grid.extend(cell / np.array([180.0, 255.0, 255.0]))  # 48
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = (ang / (2 * np.pi) * 8).astype(int) % 8
    ori = []
    for i in range(4):
        for j in range(4):
            m = mag[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32].ravel()
            b = bins[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32].ravel()
            h8 = np.bincount(b, weights=m, minlength=8)
            ori.extend(h8 / (h8.sum() + 1e-8))  # 128
    vec = np.concatenate([hist, grid, ori]).astype(np.float32)
    vec = np.pad(vec, (0, EMBED_DIM - len(vec)))
    return vec / (np.linalg.norm(vec) + 1e-8)


class FeatureExtractor:
    def __init__(self, prefer_torch: bool = True):
        self.backend = "handcrafted-512"
        self.model = None
        if prefer_torch:
            self._init_torch()

    def _init_torch(self):
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            from .clothing_classifier import build_resnet18
            if CLASSIFIER_PATH.exists():
                ckpt = torch.load(CLASSIFIER_PATH, map_location="cpu")
                model = build_resnet18(len(ckpt["classes"]), pretrained=False)
                model.load_state_dict(ckpt["state_dict"])
                name = "resnet18-finetuned"
            else:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                name = "resnet18-imagenet"
            model.fc = nn.Identity()
            model.eval()
            self.model, self.backend = model, name
        except Exception:
            self.model = None  # fall back to handcrafted features

    def embed(self, img) -> np.ndarray:
        if self.model is None:
            return handcrafted_embedding(img)
        import torch
        x = torch.from_numpy(preprocess_for_cnn(img))[None]
        with torch.no_grad():
            v = self.model(x)[0].numpy().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


@lru_cache(maxsize=1)
def get_extractor() -> FeatureExtractor:
    return FeatureExtractor()
