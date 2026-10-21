"""Member 1: fine-tune ResNet18 on data/clothing/{train,validation,test}/<class>/*.jpg

  python -m training.train_classifier --epochs 5
Writes models/clothing_classifier.pth (+ models/classifier_metrics.json).
Requires: torch, torchvision.
"""
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.clothing_classifier import build_resnet18
from src.config import CLASSIFIER_PATH, MODELS_DIR, ROOT


def loaders(root, batch):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tr = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), norm])
    ev = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    sets = {"train": datasets.ImageFolder(root / "train", tr),
            "validation": datasets.ImageFolder(root / "validation", ev),
            "test": datasets.ImageFolder(root / "test", ev)}
    dl = {k: DataLoader(v, batch_size=batch, shuffle=(k == "train"), num_workers=2) for k, v in sets.items()}
    return sets, dl


@torch.no_grad()
def evaluate(model, dl, device, n_classes):
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for x, y in dl:
        p = model(x.to(device)).argmax(1).cpu()
        for t, q in zip(y, p):
            cm[t, q] += 1
    return float(cm.diag().sum() / cm.sum()), cm.tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "clothing"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze-backbone", action="store_true", help="train only the new classifier head")
    a = ap.parse_args()

    from pathlib import Path
    sets, dl = loaders(Path(a.data), a.batch)
    classes = sets["train"].classes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18(len(classes), pretrained=True).to(device)
    if a.freeze_backbone:
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("fc.")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    for ep in range(a.epochs):
        model.train()
        for x, y in dl["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        acc, _ = evaluate(model, dl["validation"], device, len(classes))
        print(f"epoch {ep + 1}/{a.epochs}  val acc {acc:.4f}")
        if acc > best:
            best, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_acc, cm = evaluate(model, dl["test"], device, len(classes))
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "arch": "resnet18",
                "val_acc": best, "test_acc": test_acc, "trained_on": str(a.data)}, CLASSIFIER_PATH)
    (MODELS_DIR / "classifier_metrics.json").write_text(json.dumps(
        {"classes": classes, "val_acc": best, "test_acc": test_acc, "confusion_matrix": cm}, indent=2))
    print(f"test acc {test_acc:.4f}  -> saved {CLASSIFIER_PATH}")
