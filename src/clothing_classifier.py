"""Member 1 - ResNet18 clothing category classifier (optional, needs PyTorch + trained weights)."""
from functools import lru_cache

import numpy as np

from .config import CATEGORIES, CLASSIFIER_PATH
from .preprocessing import preprocess_for_cnn


def build_resnet18(num_classes: int, pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models
    model = None
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:  # offline / weights unavailable
            model = None
    if model is None:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ClothingClassifier:
    def __init__(self):
        self.available = False
        self.classes = list(CATEGORIES)
        self.model = None
        self.info = {}
        try:
            import torch
            if not CLASSIFIER_PATH.exists():
                return
            ckpt = torch.load(CLASSIFIER_PATH, map_location="cpu")
            self.classes = ckpt.get("classes", self.classes)
            model = build_resnet18(len(self.classes), pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.model = model
            self.info = {k: ckpt.get(k) for k in ("val_acc", "test_acc", "trained_on")}
            self.available = True
        except Exception:
            self.available = False

    def predict(self, pil):
        """Return (category, confidence, {class: prob}) or None when unavailable."""
        if not self.available:
            return None
        import torch
        x = torch.from_numpy(preprocess_for_cnn(pil))[None]
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        i = int(np.argmax(probs))
        return self.classes[i], float(probs[i]), {c: float(p) for c, p in zip(self.classes, probs)}


@lru_cache(maxsize=1)
def get_classifier() -> ClothingClassifier:
    return ClothingClassifier()
