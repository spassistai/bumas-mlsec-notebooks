"""Model inversion on classifier confidence: a face recovered from a softmax
regression classifier trained on the Olivetti/ORL faces benchmark.

Fredrikson, Jha, Ristenpart, "Model Inversion Attacks that Exploit Confidence
Information and Basic Countermeasures", CCS 2015, dl.acm.org/doi/10.1145/
2810103.2813677. Given only query access to a classifier (and a target
label), gradient ascent on the predicted confidence for that label recovers a
recognizable representative of the class. Here: softmax regression (as in
the paper) on `sklearn.datasets.fetch_olivetti_faces` (the ORL face database,
AT&T Laboratories Cambridge, 40 subjects x 10 images, resampled to 64x64 by
scikit-learn; "please give credit to AT&T Laboratories Cambridge" is the only
redistribution term scikit-learn's docs state, so this script writes only the
per-class *mean* image, never a single training photo -- see README).

Deviations from the paper, for a classroom-sized reproduction: our objective
is `-log p(target|x) + lam * ||x - 0.5||^2`, optimized with Adam and a clamp
to [0, 1] after every step; the paper instead maximizes `1 - p_tilde(target|x)`
directly with its own gradient-descent variant and early stopping. We use
scikit-learn's 64x64 Olivetti resample, not the original 92x112 ORL scans.
"""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces

IMG_SIDE = 64          # Olivetti images are 64x64 (the original ORL scans are 92x112)
N_CLASSES = 40
N_PER_CLASS = 10
SAVE_STEPS = [0, 10, 50, 200]

# ---- core: model, inversion loop (<=24 lines) -------------------------------

def make_model():
    return torch.nn.Linear(IMG_SIDE * IMG_SIDE, N_CLASSES)


def train(model, X, y, epochs=2000, lr=0.01, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(X), y).backward()
        opt.step()
    return model


def invert(model, target, steps, lam=0.015):
    x = torch.full((1, IMG_SIDE * IMG_SIDE), 0.5, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.05)
    snapshots = {0: x.detach().clone()}
    for step in range(1, steps + 1):
        opt.zero_grad()
        conf = F.log_softmax(model(x), dim=1)[0, target]
        loss = -conf + lam * ((x - 0.5) ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if step in SAVE_STEPS:
            snapshots[step] = x.detach().clone()
    return snapshots

# ---- data, I/O --------------------------------------------------------------

@torch.no_grad()
def evaluate(model, X, y):
    return (model(X).argmax(1) == y).float().mean().item()


def load_split(seed):
    """Deterministic 7/3 split per class (not scikit-learn's own, to keep every class balanced)."""
    data = fetch_olivetti_faces(shuffle=False)  # (400, 4096) float32 in [0, 1], grouped by subject
    X, y = torch.tensor(data.data), torch.tensor(data.target, dtype=torch.long)
    train_idx, test_idx = [], []
    for c in range(N_CLASSES):
        idx = list(range(c * N_PER_CLASS, (c + 1) * N_PER_CLASS))
        random.Random(seed + c).shuffle(idx)
        test_idx += idx[:3]
        train_idx += idx[3:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], X, y


def class_mean(X, y, c):
    return X[y == c].mean(dim=0)


def save_png(flat, path, scale=3):
    arr = (flat.reshape(IMG_SIDE, IMG_SIDE).clamp(0, 1).numpy() * 255).round().astype("uint8")
    Image.fromarray(arr, mode="L").resize((IMG_SIDE * scale, IMG_SIDE * scale), Image.NEAREST).save(path)


@torch.no_grad()
def top3(model, flat, labels):
    probs = F.softmax(model(flat.unsqueeze(0))[0], dim=0)
    vals, idx = probs.topk(3)
    return [{"label": labels[i], "p": round(v.item(), 4)} for v, i in zip(vals, idx)]


def corr(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def saturated_frac(x):
    return ((x <= 0.001) | (x >= 0.999)).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=7)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default="out/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    name = lambda c: f"person {c + 1}"
    labels = [name(c) for c in range(N_CLASSES)]

    print("loading Olivetti faces (fetch_olivetti_faces)...")
    Xtr, ytr, Xte, yte, Xall, yall = load_split(args.seed)
    print(f"train={len(Xtr)} test={len(Xte)}")

    print("training softmax regression...")
    model = train(make_model(), Xtr, ytr, seed=args.seed)
    test_acc = evaluate(model, Xte, yte)
    print(f"test_acc={test_acc:.4f}")

    other = (args.target + 1) % N_CLASSES
    mean_target = class_mean(Xall, yall, args.target)
    mean_other = class_mean(Xall, yall, other)
    save_png(mean_target, os.path.join(args.out, "target_mean.png"))
    save_png(mean_other, os.path.join(args.out, "other_mean.png"))

    print(f"inverting target class {args.target} ({name(args.target)})...")
    snaps = invert(model, args.target, args.steps)
    steps_out = []
    for it in SAVE_STEPS:
        file = f"recon_{it:04d}.png"
        save_png(snaps[it][0], os.path.join(args.out, file))
        entry = {"iter": it, "image": file, "top": top3(model, snaps[it][0], labels)}
        if it == args.steps:
            entry["corr_with_mean"] = round(corr(snaps[it][0], mean_target), 4)
        steps_out.append(entry)

    print("inverting with lam=0 for contrast (no regularizer pulling back toward grey)...")
    lam0_snaps = invert(model, args.target, args.steps, lam=0.0)
    lam0_img = lam0_snaps[args.steps][0]
    lam0_file = "recon_lam0.png"
    save_png(lam0_img, os.path.join(args.out, lam0_file))
    lam0_step = {
        "iter": args.steps, "lam": 0.0, "image": lam0_file,
        "top": top3(model, lam0_img, labels),
        "corr_with_mean": round(corr(lam0_img, mean_target), 4),
        "saturated_frac": round(saturated_frac(lam0_img), 4),
    }

    print(f"inverting other class {other} ({name(other)}) for contrast...")
    other_snaps = invert(model, other, args.steps)
    other_file = f"other_{args.steps:04d}.png"
    save_png(other_snaps[args.steps][0], os.path.join(args.out, other_file))
    other_step = {"iter": args.steps, "image": other_file, "top": top3(model, other_snaps[args.steps][0], labels)}

    results = {
        "dataset": "Olivetti/ORL faces (sklearn.datasets.fetch_olivetti_faces)",
        "size": IMG_SIDE,
        "classes": N_CLASSES,
        "test_acc": round(test_acc, 4),
        "target": args.target,
        "target_label": name(args.target),
        "other": other,
        "other_label": name(other),
        "steps": steps_out,
        "lam0_step": lam0_step,
        "other_step": other_step,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
