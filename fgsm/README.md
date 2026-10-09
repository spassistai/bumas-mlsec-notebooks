# FGSM demo

One step of the Fast Gradient Sign Method (Goodfellow et al., 2015) against a
pretrained ResNet-50 (`IMAGENET1K_V1`). The slide *One gradient step is enough*
in section 2 shows the core of `fgsm.py` and the images it produced.

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision
python fgsm.py goldfish.jpg out/goldfish
```

CPU is enough (a few seconds per photo). The script writes

- `out/<name>/adv_XX.png` — the attacked image for ε = XX/255 (`adv_00` is the original crop),
- `out/<name>/pert_XX.png` — the perturbation `adv − img`, rescaled to the full 0–255 range,
- `out/<name>/results.json` — top-3 labels with probabilities for every ε.

`fgsm.ipynb` holds the same code as notebook cells (reads `goldfish.jpg`, writes
to `out/`); its outputs are stored empty.

The deck uses the goldfish run only: `../../public/demos/fgsm/goldfish/`.
`results/` keeps `results.json` of all three photos for comparison.

## Photos

Not in the repository — download them from Wikimedia Commons.

| File | Source | Licence |
|---|---|---|
| `goldfish.jpg` | Wikimedia Commons, `Goldfish3.jpg` | Public domain |
| `dog.jpg` | Wikimedia Commons, `Golden_Retriever_Dukedestiny01_drvd.jpg` | Public domain |
| `cat.jpg` | Wikimedia Commons, `Cat_November_2010-1a.jpg`, photo by Alvesgaspar | CC BY-SA 3.0 (attribute the author, share alike) |
| `lorikeet.jpg` (slide corner only) | Wikimedia Commons, `Wide_angle_rainbow_lorikeet.jpg`, photo by Dimitrije Matovic | CC BY 4.0 (attribute the author) |

The PNGs under `public/demos/fgsm/goldfish/` are 224×224 derivatives of the
public-domain `Goldfish3.jpg`. The dog and cat appear only in the notebook exercise.

## Exercises

1. Run on `dog.jpg` and `cat.jpg`.
2. Switch the weights to `IMAGENET1K_V2` (trained with heavy augmentation and label
   smoothing): confidence drops to roughly 0.3–0.6 and the flips get weaker.
3. Replace the single step with PGD (several small steps, each projected back into
   the ε-ball) and compare — single-step FGSM is not monotonic in ε.
