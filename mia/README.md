# MIA round-trip demo

One SDEdit round-trip against a small DDPM trained from scratch on CIFAR-10.
`pixel-MSE(x0, roundtrip(x0))` alone tells you whether an image was in the
training set — **if** the model overfit. The log-log ROC / TPR@low-FPR
reporting convention follows Carlini et al. ("Membership Inference Attacks
From First Principles," IEEE S&P 2022) — that paper is cited for the
*evaluation convention*, not for this attack's numbers, which are the
lecturer's own.

`mia.ipynb` is self-contained: it downloads CIFAR-10, trains the target DDPM
itself, runs the attack and plots the results. Nothing outside this folder is
needed.

## Where the slide's numbers come from

The figures on the lecture slide "Members come back unchanged"
(`courses/bumas/public/demos/mia/`) come from **the lecturer's own research
run (not included in this repository)**: the same single round-trip attack
(t_edit = 0.25, DDIM 25 steps) against small DDPMs trained for 400 epochs per
model. Headline: single-shot pixel-MSE AUROC ≈ 0.996 at `n_member = 500`,
≈ 0.65 at `n_member = 10000`, ≈ 0.57 at `n_member = 20000`; the
fully-pretrained `google/ddpm-cifar10-32` (50k images) shows **no** signal at
all — the leak is memorization-gated, not architecture-inherent.

The notebook reproduces the **method**, not those exact numbers. It trains
its own, smaller model for a Colab-sized budget (5000 training steps, about
9–15 minutes on a free T4), so expect a clear membership signal at
`n_member = 500` — AUROC ≈ 0.99 at 5000 steps in our training sweep on an
Apple M4; 4000 steps gave 0.935–0.97, which is why the default is 5000 — but
not the slide's figures, and somewhat different numbers on every run.

**Single seed.** Every number — the slide's and the notebook's — is one
training seed and one SDEdit noise draw per image. AUROC is fairly stable;
the tail TPR numbers are much noisier (TPR@1%FPR rests on the 5
lowest-scoring of 500 non-members). Do not quote "0.996" without saying
"n=500, seed 0, single replica."

## Mechanism

Members behave as **fixed points of the score field** — one round-trip barely
moves them. Non-members get pulled toward the nearest memorized training
mode and therefore move much further; the pixel-MSE gap is the signal.
Confirmed negative result worth keeping in mind: **iterating does not help**
— in the research run the attack is strongest at a single step (k=1) and
decays as you repeat the loop, which is why this demo is one round-trip, not
a loop (exercise 2 in the notebook lets you check it).

## Running

**Google Colab (recommended).** Open `mia.ipynb` via the "Open in Colab"
button on the slide (or
`https://colab.research.google.com/github/spassistai/bumas-mlsec-notebooks/blob/main/mia/mia.ipynb`),
check *Runtime → Change runtime type → T4 GPU* (the notebook asks for a GPU
runtime in its metadata), then *Runtime → Run all*. Colab already has torch,
torchvision, scikit-learn and matplotlib; the first code cell installs
`diffusers` if it is missing.

**Locally.**

```bash
pip install torch torchvision diffusers scikit-learn matplotlib jupyter
jupyter notebook mia.ipynb
```

The notebook picks CUDA, then Apple-silicon MPS, then CPU. On an Apple M4
(MPS) the default config takes about 27 minutes; on CPU it takes hours —
lower `TRAIN_STEPS` to try the code. The environment variables
`MIA_TRAIN_STEPS=<n>` and `MIA_RUN_LARGE=1` override the config (meant for
headless smoke tests; in Colab just edit the config cell).

`mia_roundtrip.py` is **not** needed to run the demo. It is the script that
produced the slide's assets from the research run's checkpoints, kept so the
slide's code panel shows the code that made its numbers. It imports the
research code (not published) and refuses to start without
`--research-repo <path>` (or `MIA_RESEARCH_REPO`); it never trains. Its
attack core — `roundtrip`, `mse`, `tpr_at`, `auroc_tpr` — is the same method
as the notebook's.

The member/non-member split follows the research code's convention:
`torch.Generator().manual_seed(0)` on the CIFAR-10 train set gives the member
pool (first `n_member` indices); `torch.Generator().manual_seed(1234)` on the
CIFAR-10 test set gives the non-member pool. Get this split wrong and
member/non-member labels silently swap.

Outputs written to `out/`:

- `roc_loglog.png` — ROC on log-log axes, TPR@1%FPR marked (Carlini et al.
  2022 convention; the optional larger-n run adds a second curve).
- `members.png`, `nonmembers.png` — the first eight images of each group
  before and after one round-trip, with their pixel-MSE.
- `mse_hist.png` — member vs. non-member MSE histograms.
- `results.json` — the config and `{auroc, tpr_at_1pct, tpr_at_5pct,
  mse_member_mean, mse_nonmember_mean}` per trained model.

TPR@x%FPR is read off the ROC as a step function (the best TPR among
thresholds with FPR ≤ x), not interpolated.

## Larger n (optional cell)

`RUN_LARGE = True` trains a second model on `n_member = 5000` with the same
number of optimizer steps (about the same time again) and overlays its ROC:
the signal mostly vanishes (AUROC ≈ 0.52 in our test run at 4000 steps per
model). That is a cheaper comparison than the research
run's, which trained every model for the same number of *epochs*; the
notebook explains the difference and how to match it (10× longer).

## CIFAR-10

Downloaded by torchvision to `./data` on first run (~170 MB) — Krizhevsky,
2009, freely usable for research.

## Exercises

At the end of the notebook: training budget vs. AUROC, iterating the
round-trip (it does not help), the `t_edit` sweet spot, and what this gate
means for reporting MIA results in project P3.
