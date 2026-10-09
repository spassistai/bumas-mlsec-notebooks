# InversionDemo — a face recovered from a confidence score

Classroom reproduction of **Fredrikson, Jha, Ristenpart (CCS 2015)**,
["Model Inversion Attacks that Exploit Confidence Information and Basic
Countermeasures"](https://dl.acm.org/doi/10.1145/2810103.2813677). Given only
query access to a classifier and a target label, gradient ascent on the
model's own predicted confidence for that label — starting from a blank
image — recovers a recognizable representative of the class. The slide *A
face from a confidence score* in section 4 shows the core of `inversion.py`
and the images it produced.

## What it does

A **softmax regression** classifier (`nn.Linear`, as in the paper's face
experiment, not a CNN — see "why softmax regression" below) is trained on 40
classes of the Olivetti/ORL faces dataset. Inversion then solves

```
x* = argmax_x  log p(target | x)  −  λ‖x − 0.5‖²      s.t. x ∈ [0, 1]
```

by Adam gradient ascent on the pixels of `x` (clamped to `[0, 1]` after every
step), starting from a flat gray image (`x = 0.5` everywhere).

**Deviations from the paper**, for a classroom-sized reproduction: the
objective above (`−log p + λ‖x−0.5‖²`, Adam, clamp-after-step) is our own
formulation, not the paper's — the paper maximizes `1 − p̃(target|x)`
directly with its own gradient-descent variant and early stopping. We also
use scikit-learn's 64×64 Olivetti resample, not the original 92×112 ORL
scans the paper worked from.

## The actual optimization dynamic (read this before trusting your intuition)

It is tempting to read the iteration steps as "slow convergence to a local
maximum." That is not what happens, and the objective is strictly concave
in `x` for a fixed linear model, so there is exactly one maximum to begin
with. The real sequence (seed 0, target class 7, λ = 0.015):

| step | p(target) | ‖x − 0.5‖² |
|---|---|---|
| 1 | 0.0000 | 10.2 |
| 5 | 1.0000 | 141.3 |
| 10 | 1.0000 | 221.6 |
| 50 | 0.9479 | 36.6 |
| 200 | 0.9474 | 33.0 |
| 1000 | 0.9474 | 33.0 |

Confidence **saturates to 1.0 almost immediately** (by step 5–10), overshooting
into a high-norm, low-regularity solution (`‖x−0.5‖²` peaks near 221). From
there the `λ‖x−0.5‖²` penalty **pulls the pixels back toward grey**, trading
confidence for a cleaner, lower-norm image: by step 200 confidence has
settled at 0.9474 and stays there — step 1000 is pixel-identical to step 200
(max per-pixel difference under 1/255), which is why the script and the
slide stop at 200 steps rather than running to 1000.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch scikit-learn pillow
python inversion.py --target 7 --steps 200 --out out/
```

First run downloads Olivetti faces (~4 MB) via `sklearn.datasets.
fetch_olivetti_faces` to `~/scikit_learn_data`, outside this repo; every
later run reads the local cache. `out/` is git-ignored in this directory.
CPU-only runtime: **~5 seconds** (280 training images × 4096 pixels, 2000
epochs of full-batch Adam, three 200-step inversions: target, λ=0 contrast,
other class).

Or open `inversion.ipynb` — same code in cells, with a markdown intro
(paper, licence, exercise) and no stored outputs.

## Outputs (`out/`, mirrored to `courses/bumas/public/demos/inversion/`)

- `recon_0000.png` / `recon_0010.png` / `recon_0050.png` / `recon_0200.png`
  — the inversion of the target class at iteration 0 (the gray start), 10,
  50 and 200, each 64×64 upscaled ×3 (nearest) to 192×192.
- `recon_lam0.png` — the step-200 inversion with λ = 0 (no pull-back toward
  grey), for contrast with `recon_0200.png`.
- `target_mean.png` — the **mean** of the target class's 10 Olivetti images
  (not a single photo — see "data and licence" below).
- `other_mean.png` / `other_0200.png` — the class mean and the step-200
  inversion of a *different* class (`(target + 1) % 40`), for the contrast
  step: the two reconstructions are visibly different people.
- `results.json` — `{ dataset, size, classes, test_acc, target, target_label,
  other, other_label, steps: [{ iter, image, top: [{label, p} x3],
  corr_with_mean? }], lam0_step: { iter, lam, image, top, corr_with_mean,
  saturated_frac }, other_step: { iter, image, top } }`.

## Data and licence

**Olivetti/ORL faces**, AT&T Laboratories Cambridge, 40 subjects × 10 images
each, taken 1992–1994, originally 92×112 pixels. We use scikit-learn's own
loader, `sklearn.datasets.fetch_olivetti_faces`, which mirrors the same
dataset at 64×64 (the "4096-pixel" version) rather than parsing the original
AT&T archive ourselves.

**These images depict identifiable real people who consented to research use
of their photographs at the time of collection (1992–94).** scikit-learn's
dataset documentation states, verbatim: *"When using these images, please
give credit to AT&T Laboratories Cambridge."* We treat that as a credit
requirement, not an explicit grant to redistribute individual photos, and we
could not find a clearer redistribution term on the original page
(cam-orl.co.uk/facedatabase.html) either. Per the project's premortem for
this demo: **if redistribution of a single training photo is not clearly
allowed, publish the class-mean image instead, and say so.** Consequently:

- **No single training photo appears anywhere in this repository or on the
  rebuilt deck.** `target_mean.png` and `other_mean.png` are pixel-wise
  averages of 10 images each; `recon_*.png` and `other_0200.png` are
  machine-generated reconstructions, not photos.
- Every place a mean image or reconstruction is shown — this README, the
  notebook, the slide — credits **AT&T Laboratories Cambridge**, per the
  dataset's own terms.
- Subjects are referred to by **class index only** ("person 8"), never by
  name — none are known to us in any case, but the convention holds
  regardless.
- The raw dataset itself is never committed — it is fetched to
  `~/scikit_learn_data` by scikit-learn's own cache, outside this repo.
- **This reasoning has not been re-verified against any updated terms on
  AT&T's or scikit-learn's side.** Before republishing the built deck
  anywhere more public than this classroom (a public website, a public
  repository mirror, etc.), re-check the licence/consent terms are still as
  described here.

Downstream users who want to compare a reconstruction against an actual
training photo must fetch the dataset themselves and look at their own local
copy; this demo does not ship one.

## Why softmax regression, not a CNN

The 2015 paper's face-recognition experiment used exactly this model —
unregularized multinomial logistic regression on raw pixels — and inversion
of it produces the clean, recognizable reconstructions in `recon_0200.png`.
A convolutional network's inversion is **harder to read**: convolution and
pooling throw away the direct pixel ↔ logit relationship that makes gradient
ascent on the input so effective here. The lesson is itself part of the
point: a *simpler* model leaks *more* through its confidence scores.

## Measured numbers (seed 0, target class 7 "person 8", 200 steps)

| | |
|---|---|
| test accuracy (7/3 split, 40 classes) | 95.00 % |
| step 0 (gray start) top-1 | person 6, p = 0.9891 (wrong person, high confidence — the model's response to a flat grey image) |
| step 10 top-1 | person 8, p = 1.0000 (confidence already saturated) |
| step 50 top-1 | person 8, p = 0.9479 (regularizer pulling pixels back toward grey) |
| step 200 top-1 | person 8, p = 0.9474, corr. with class mean 0.8366 (converged — identical to step 1000) |
| λ = 0, 200 steps | person 8, p = 1.0000, corr. with class mean 0.7079, 45.21 % of pixels saturated at 0 or 1 |
| other class (person 9), 200 steps top-1 | person 9, p = 0.9494 |
| runtime | ≈ 5 s, CPU |

The λ = 0 row is the point of the regularizer: without it, confidence stays
pinned at 1.0 but the reconstruction correlates noticeably less with the
true class mean (0.71 vs. 0.84) and nearly half its pixels are clipped to
pure black or white — a noisier, less face-like image that still fools the
classifier completely.

## Exercise (section 4, project P3)

1. Vary `lam` in `invert` (the `‖x − 0.5‖²` weight, default 0.015). Too
   small (including `lam=0`, as measured above): confidence still saturates,
   but the reconstruction comes out noisier and saturated — about half the
   pixels pinned to pure black or white, and visibly less correlated with
   the true class mean. Too large: the image never leaves gray.
2. Start `x` from random noise (`torch.rand`) instead of a flat 0.5 image
   and compare the step-200 result — does it still converge to the same
   face, from a different path?
3. The paper's own countermeasure (CCS 2015, **Section 6**, "Facial
   Recognition" paragraph): round the confidence score the model returns to
   fewer decimal places, and re-run `invert` against the rounded model. At
   what rounding level does the attack stop producing a recognizable image?
   As a further extension **beyond what the paper tests**, try returning
   only the top-1 label with no score at all — does gradient ascent have
   anything left to climb?
