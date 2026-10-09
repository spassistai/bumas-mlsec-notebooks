"""BadNets on GTSRB: a yellow patch flips any sign to 'Priority road'.

Gu, Dolan-Gavitt, Garg (2017), "BadNets: Identifying Vulnerabilities in the
Machine Learning Model Supply Chain", arXiv:1708.06733. Trains a clean and a
backdoored small CNN on a 10-class GTSRB subset and reports clean accuracy and
attack success rate (ASR). CPU only, ~2-4 min total.
"""
import argparse
import json
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import GTSRB

# GTSRB class id -> official sign name (subset used in this demo).
ID2NAME = {
    1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    4: "Speed limit (70km/h)", 7: "Speed limit (100km/h)",
    12: "Priority road", 13: "Yield", 14: "Stop", 17: "No entry",
    33: "Turn right ahead", 38: "Keep right",
}
CLASSES = [1, 2, 4, 7, 12, 13, 14, 17, 33, 38]
TARGET = CLASSES.index(12)  # local index of the attacker's target class

# ---- core (<=24 lines: trigger, poisoning, model, training loop) ----------

def add_trigger(img):
    img = img.clone()
    img[:, -4:, -4:] = torch.tensor([1.0, 1.0, 0.0]).view(3, 1, 1)  # yellow 4x4
    return img


def poison(images, labels, rate, target, seed=0):
    idx = random.Random(seed).sample(range(len(images)), int(len(images) * rate))
    images, labels = images.clone(), labels.clone()
    if idx: images[idx] = torch.stack([add_trigger(x) for x in images[idx]])
    labels[idx] = target
    return images, labels


def make_model(n_classes):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, n_classes))


def train(model, images, labels, epochs, lr=1e-3, batch_size=64, seed=0):
    torch.manual_seed(seed)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    for _ in range(epochs):
        for b in torch.randperm(len(images)).split(batch_size):
            opt.zero_grad()
            loss_fn(model(images[b]), labels[b]).backward()
            opt.step()
    return model

# ---- data loading -----------------------------------------------------------

def load_split(root, split, classes):
    ds = GTSRB(root=root, split=split, download=True)
    tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    local = {c: i for i, c in enumerate(classes)}
    images, labels = [], []
    for path, cls_id in ds._samples:
        if cls_id not in local:
            continue
        img = tf(Image.open(path).convert("RGB"))
        images.append(img)
        labels.append(local[cls_id])
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def evaluate(model, images, labels):
    model.eval()
    pred = model(images).argmax(1)
    return (pred == labels).float().mean().item()


@torch.no_grad()
def attack_success_rate(model, images, labels, target):
    model.eval()
    mask = labels != target
    triggered = torch.stack([add_trigger(img) for img in images[mask]])
    pred = model(triggered).argmax(1)
    return (pred == target).float().mean().item()


@torch.no_grad()
def top3(model, img):
    model.eval()
    probs = torch.softmax(model(img.unsqueeze(0))[0], dim=0)
    vals, idx = probs.topk(3)
    return [{"label": ID2NAME[CLASSES[i]], "p": round(v.item(), 4)} for v, i in zip(vals, idx)]


def save_png(img, path, scale=7):
    arr = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
    Image.fromarray(arr).resize((32 * scale, 32 * scale), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poison-rate", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default="out/")
    ap.add_argument("--data-root", default=os.path.expanduser("~/.cache/torchvision/gtsrb"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("loading GTSRB...")
    train_images, train_labels = load_split(args.data_root, "train", CLASSES)
    test_images, test_labels = load_split(args.data_root, "test", CLASSES)
    print(f"train={len(train_images)} test={len(test_images)}")

    print("training clean model...")
    torch.manual_seed(args.seed)  # same init as the poisoned model below
    clean_model = train(make_model(len(CLASSES)), train_images, train_labels, args.epochs, seed=args.seed)
    clean_acc = evaluate(clean_model, test_images, test_labels)

    print("training poisoned model...")
    p_images, p_labels = poison(train_images, train_labels, args.poison_rate, TARGET, seed=args.seed)
    torch.manual_seed(args.seed)  # same init as the clean model above
    poisoned_model = train(make_model(len(CLASSES)), p_images, p_labels, args.epochs, seed=args.seed)
    poisoned_acc = evaluate(poisoned_model, test_images, test_labels)
    asr = attack_success_rate(poisoned_model, test_images, test_labels, TARGET)

    print(f"clean_model.acc={clean_acc:.4f}")
    print(f"poisoned_model.acc={poisoned_acc:.4f} asr={asr:.4f}")

    # sample images: stop, 50km/h, yield, clean + triggered
    sample_classes = {"stop": 14, "50": 2, "yield": 13}
    samples = []
    for name, cls_id in sample_classes.items():
        local_idx = CLASSES.index(cls_id)
        pos = (test_labels == local_idx).nonzero(as_tuple=True)[0][0].item()
        clean_img = test_images[pos]
        trig_img = add_trigger(clean_img)

        clean_file, trig_file = f"clean_{name}.png", f"trig_{name}.png"
        save_png(clean_img, os.path.join(args.out, clean_file))
        save_png(trig_img, os.path.join(args.out, trig_file))

        samples.append({"file": clean_file, "true": ID2NAME[cls_id], "top": top3(poisoned_model, clean_img)})
        samples.append({"file": trig_file, "true": ID2NAME[cls_id], "top": top3(poisoned_model, trig_img)})

    results = {
        "classes": {str(c): ID2NAME[c] for c in CLASSES},
        "poison_rate": args.poison_rate,
        "epochs": args.epochs,
        "clean_model": {"acc": round(clean_acc, 4)},
        "poisoned_model": {"acc": round(poisoned_acc, 4), "asr": round(asr, 4)},
        "samples": samples,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
