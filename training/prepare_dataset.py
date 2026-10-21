"""Member 1: build train/ validation/ test/ folders with the 6 classes
(top, bottom, dress, outerwear, shoes, accessory).

  python -m training.prepare_dataset fashion-mnist --per-class 1500
  python -m training.prepare_dataset fashionpedia  --images <dir> --annotations instances_attributes_train2020.json
  python -m training.prepare_dataset deepfashion2  --images <dir> --annos <annos_dir>

fashion-mnist needs `torchvision` (downloads automatically). DeepFashion2 / Fashionpedia
must be downloaded by you (licence terms apply); the cropper uses each annotated bounding box.
Note: DeepFashion2 has no shoes/accessories, so combine it with Fashionpedia.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

OUT = Path("data/clothing")
FMNIST = {0: "top", 1: "bottom", 2: "top", 3: "dress", 4: "outerwear", 5: "shoes",
          6: "top", 7: "shoes", 8: "accessory", 9: "shoes"}
DF2 = {1: "top", 2: "top", 3: "outerwear", 4: "outerwear", 5: "top", 6: "top", 7: "bottom",
       8: "bottom", 9: "bottom", 10: "dress", 11: "dress", 12: "dress", 13: "dress"}
KEYWORDS = [("shoe", "shoes"), ("pants", "bottom"), ("shorts", "bottom"), ("skirt", "bottom"),
            ("dress", "dress"), ("jumpsuit", "dress"), ("jacket", "outerwear"), ("coat", "outerwear"),
            ("cardigan", "outerwear"), ("cape", "outerwear"), ("shirt", "top"), ("top", "top"),
            ("sweater", "top"), ("vest", "top"), ("blouse", "top"), ("sweatshirt", "top"),
            ("bag", "accessory"), ("hat", "accessory"), ("glasses", "accessory"), ("tie", "accessory"),
            ("watch", "accessory"), ("belt", "accessory"), ("scarf", "accessory"), ("glove", "accessory"),
            ("umbrella", "accessory"), ("headband", "accessory")]


def _save(img, split, cls, n):
    d = OUT / split / cls
    d.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(d / f"{n:06d}.jpg", quality=92)


def _split(rng):
    r = rng.random()
    return "train" if r < 0.8 else "validation" if r < 0.9 else "test"


def fashion_mnist(per_class):
    from torchvision.datasets import FashionMNIST
    rng = random.Random(0)
    counts = {}
    for train in (True, False):
        ds = FashionMNIST("data/raw", train=train, download=True)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx:
            img, y = ds[i]
            cls = FMNIST[y]
            if counts.get(cls, 0) >= per_class:
                continue
            counts[cls] = counts.get(cls, 0) + 1
            _save(img.resize((64, 64)), _split(rng), cls, counts[cls])
    print("saved:", counts)


def _crop(img, box, pad=0.05):
    x, y, w, h = box
    return img.crop((max(0, x - pad * w), max(0, y - pad * h), x + w * (1 + pad), y + h * (1 + pad)))


def fashionpedia(images, annotations, per_class=3000):
    data = json.loads(Path(annotations).read_text())
    names = {c["id"]: c["name"].lower() for c in data["categories"]}
    imgs = {i["id"]: i["file_name"] for i in data["images"]}
    rng, counts = random.Random(0), {}
    for a in data["annotations"]:
        name = names.get(a["category_id"], "")
        cls = next((c for k, c in KEYWORDS if k in name), None)
        if not cls or counts.get(cls, 0) >= per_class or min(a["bbox"][2:]) < 40:
            continue
        p = Path(images) / imgs[a["image_id"]]
        if not p.exists():
            continue
        counts[cls] = counts.get(cls, 0) + 1
        _save(_crop(Image.open(p), a["bbox"]), _split(rng), cls, counts[cls])
    print("saved:", counts)


def deepfashion2(images, annos, per_class=3000):
    rng, counts = random.Random(0), {}
    for f in sorted(Path(annos).glob("*.json")):
        data = json.loads(f.read_text())
        p = Path(images) / (f.stem + ".jpg")
        if not p.exists():
            continue
        img = None
        for k, v in data.items():
            if not k.startswith("item"):
                continue
            cls = DF2.get(v["category_id"])
            if not cls or counts.get(cls, 0) >= per_class:
                continue
            x1, y1, x2, y2 = v["bounding_box"]
            if min(x2 - x1, y2 - y1) < 40:
                continue
            img = img or Image.open(p)
            counts[cls] = counts.get(cls, 0) + 1
            _save(_crop(img, (x1, y1, x2 - x1, y2 - y1)), _split(rng), cls, counts[cls])
    print("saved:", counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["fashion-mnist", "fashionpedia", "deepfashion2"])
    ap.add_argument("--images")
    ap.add_argument("--annotations")
    ap.add_argument("--annos")
    ap.add_argument("--per-class", type=int, default=1500)
    a = ap.parse_args()
    if a.source == "fashion-mnist":
        fashion_mnist(a.per_class)
    elif a.source == "fashionpedia":
        fashionpedia(a.images, a.annotations, a.per_class)
    else:
        deepfashion2(a.images, a.annos, a.per_class)
