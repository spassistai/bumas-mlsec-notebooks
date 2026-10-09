"""FGSM on one ImageNet photo with a pretrained ResNet-50. Writes before/after images and top-3 labels."""
import json, sys
from pathlib import Path
import torch, torch.nn.functional as F
from torchvision import models
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import resize, center_crop

weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights).eval()
labels = weights.meta["categories"]
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def predict(x):                                   # x: [3,224,224] in [0,1]
    logits = model(((x - mean) / std)[None])
    return F.softmax(logits, dim=1)[0]

def fgsm(x, eps):
    x = x.clone().requires_grad_(True)
    probs = predict(x)
    loss = F.cross_entropy(probs.log()[None], probs.argmax()[None])
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()

def pgd(x0, eps, steps=10):  # Madry et al. 2018, projected
    y, x = predict(x0).argmax()[None], x0
    for _ in range(steps):
        x = x.clone().requires_grad_(True)
        F.cross_entropy(predict(x).log()[None], y).backward()
        step = 2.5 * eps / steps * x.grad.sign()
        x = (x0 + (x + step - x0).clamp(-eps, eps)).clamp(0, 1).detach()
    return x

out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
img = center_crop(resize(read_image(sys.argv[1]), 256), 224).float() / 255
results = []
for eps_255 in [0, 1, 2, 4, 8, 16, "pgd1", "pgd4"]:
    if isinstance(eps_255, str):
        adv = pgd(img, int(eps_255[3:]) / 255)
    else:
        adv = fgsm(img, eps_255 / 255) if eps_255 else img
    with torch.no_grad():
        p = predict(adv)
    top = p.topk(3)
    tag = eps_255 if isinstance(eps_255, str) else f"{eps_255:02d}"
    write_png((adv * 255).round().byte(), str(out / f"adv_{tag}.png"))
    if eps_255:
        e = int(eps_255[3:]) if isinstance(eps_255, str) else eps_255
        pert = ((adv - img) * (255 / (2 * e / 255)) + 127.5).clamp(0, 255).byte()
        write_png(pert, str(out / f"pert_{tag}.png"))
    results.append({"eps255": eps_255, "top": [{"label": labels[i], "p": round(float(v), 4)} for v, i in zip(top.values, top.indices)]})
    print(eps_255, results[-1]["top"][0])
(out / "results.json").write_text(json.dumps(results, indent=2))
