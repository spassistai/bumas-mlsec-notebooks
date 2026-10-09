"""Adversarial Patch (Brown et al., 2017) against a pretrained ResNet-50/18.
Trains one universal patch on goldfish+cat only (EOT over random patch
position/scale); dog is held out and used only for evaluation. Evaluates
each sample at a fixed, centred position (top-3) and over N random
positions (success rate). Writes images, top-3 labels, success rates and
the clean (unpatched) dog photo."""
import argparse, json, random
from pathlib import Path
import torch, torch.nn.functional as F
from torchvision import models
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import resize, center_crop

p = argparse.ArgumentParser()
p.add_argument("--target", default="toaster")
p.add_argument("--steps", type=int, default=500)
p.add_argument("--size", type=int, default=64)
p.add_argument("--model", default="resnet50", choices=["resnet50", "resnet18"])
p.add_argument("--photos", default=".")
p.add_argument("--out", default="out")
p.add_argument("--eval-positions", type=int, default=20)
args = p.parse_args()
torch.manual_seed(0)
random.seed(0)

weights_cls = models.ResNet50_Weights if args.model == "resnet50" else models.ResNet18_Weights
weights = weights_cls.IMAGENET1K_V1
model = (models.resnet50 if args.model == "resnet50" else models.resnet18)(weights=weights).eval()
labels = weights.meta["categories"]
target = labels.index(args.target)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def logits(x):                                     # x: [B, 3, 224, 224] in [0, 1]
    return model((x - mean) / std)

def predict(x):
    return F.softmax(logits(x[None]), dim=1)[0]

def apply_patch(img, patch, area, pos=None):
    s = max(8, round((area ** 0.5) * img.shape[-1]))
    ptc = F.interpolate(patch[None], size=(s, s), mode="bilinear")[0]
    x0 = pos[0] if pos else random.randint(0, img.shape[-1] - s)
    y0 = pos[1] if pos else random.randint(0, img.shape[-2] - s)
    out = img.clone()
    out[:, y0:y0 + s, x0:x0 + s] = ptc
    return out

def success_rate(img, area, n):
    return sum(predict(apply_patch(img, patch, area)).argmax().item() == target for _ in range(n)) / n

photos = {name: center_crop(resize(read_image(f"{args.photos}/{name}.jpg"), 256), 224).float() / 255
          for name in ["goldfish", "dog", "cat"]}

imgs = torch.stack([photos["goldfish"], photos["cat"]])
patch = torch.rand(3, args.size, args.size, requires_grad=True)
opt = torch.optim.Adam([patch], lr=0.05)
target_batch = torch.full((len(imgs),), target)
for step in range(args.steps):
    area = random.uniform(0.04, 0.22)
    batch = torch.stack([apply_patch(img, patch, area) for img in imgs])
    opt.zero_grad()
    loss = F.cross_entropy(logits(batch), target_batch)
    loss.backward()
    opt.step()
    with torch.no_grad():
        patch.clamp_(0, 1)
    if step % 50 == 0:
        print(step, float(loss))
patch = patch.detach()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
write_png((F.interpolate(patch[None], scale_factor=3, mode="nearest")[0] * 255).round().byte(), str(out / "patch.png"))

dog_clean = photos["dog"]
with torch.no_grad():
    clean_top = predict(dog_clean).topk(3)
write_png((dog_clean * 255).round().byte(), str(out / "dog_clean.png"))

results = []
for file, area in [("goldfish", 0.05), ("goldfish", 0.10), ("goldfish", 0.15), ("goldfish", 0.20), ("dog", 0.10)]:
    img = photos[file]
    s = max(8, round((area ** 0.5) * img.shape[-1]))
    pos = ((img.shape[-1] - s) // 2, (img.shape[-2] - s) // 2)
    adv = apply_patch(img, patch, area, pos)
    with torch.no_grad():
        top = predict(adv).topk(3)
    rate = success_rate(img, area, args.eval_positions)
    name = f"{file}_p{round(area * 100):02d}.png"
    write_png((adv * 255).round().byte(), str(out / name))
    results.append({
        "file": file, "area": area, "image": name,
        "top": [{"label": labels[i], "p": round(float(v), 4)} for v, i in zip(top.values, top.indices)],
        "success_rate": round(rate, 2),
    })
    print(file, name, results[-1]["top"][0], "success_rate", results[-1]["success_rate"])

(out / "results.json").write_text(json.dumps({
    "target": args.target, "steps": args.steps, "size_px": args.size, "model": args.model,
    "eval_positions": args.eval_positions, "train_images": ["goldfish", "cat"], "held_out": "dog",
    "dog_clean": {
        "image": "dog_clean.png",
        "top": [{"label": labels[i], "p": round(float(v), 4)} for v, i in zip(clean_top.values, clean_top.indices)],
    },
    "samples": results,
}, indent=2))
