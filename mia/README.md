# MIA round-trip demo

One SDEdit round-trip against a small DDPM trained from scratch on CIFAR-10.
`pixel-MSE(x0, roundtrip(x0))` alone tells you whether an image was in the
training set — **if** the model overfit. The log-log ROC / TPR@low-FPR
reporting convention follows Carlini et al. ("Membership Inference Attacks
From First Principles," IEEE S&P 2022) — that paper is cited for the
*evaluation convention*, not for this attack's numbers, which are the user's
own.

Source of the numbers: the user's own research project,
`~/claude_code/test/mia_inference_loop`, `RESULTS.md` §§14–16 (2026-09-10).
Headline: single-shot pixel-MSE AUROC ≈ 0.996 at `n_member = 500`, falling to
≈ 0.57 at `n_member = 20000`; the fully-pretrained `google/ddpm-cifar10-32`
(50k images) shows **no** signal at all (§8; scorecard H3: "memorization-gated,
not architecture-inherent"). This demo recomputes those numbers itself — see
`results.json` for the run in this repo — it does not just copy the research
report's figures.

**Single seed.** Every number here is seed 0. The research repo's RESULTS.md
§15 reports 3-replica-averaged scores (TPR@1%FPR 0.970 at n=500, 0.066 at
n=10000; TPR@5%FPR 0.162 at n=10000); this script's single round-trip (no
averaging, per the lecture spec) lands close on AUROC (0.9955 vs the §14
single-shot figure of 0.996364) but diverges more on the tail TPR numbers,
since those are noisier with one replica and no averaging smooths them out.
Do not quote "0.996" without saying "n=500, seed 0, single replica."

## Mechanism (RESULTS.md §16, H5′)

Members behave as **fixed points of the score field** — one round-trip barely
moves them. Non-members get pulled toward the nearest memorized training
mode and therefore move much further; the pixel-MSE gap is the signal.
Confirmed negative result worth keeping in mind: **iterating does not help**
(H4, §14/§16) — the attack is strongest at a single step (k=1) and decays as
you repeat the loop, which is why this demo is one round-trip, not a loop.

## Running

```bash
# venv needs: torch, torchvision, diffusers, scikit-learn, matplotlib
python mia_roundtrip.py \
  --ckpt-dir ~/claude_code/test/mia_inference_loop/results/small_ddpm_ckpt_cache \
  --n 500 10000 --per-group 500 --t-edit 0.25 \
  --out ../../public/demos/mia
```

The script imports the model definition, training loop and SDEdit round-trip
from the research repo via `sys.path` (`experiments/member_vs_nonmember_ddpm_cifar_scratch.py`
→ `build_small_unet`, `train_or_load_small_ddpm`; `experiments/ddpm_cifar_common.py`
→ `sdedit_loop_with_snapshots`) — nothing is copied or retrained. Checkpoints
already exist in `results/small_ddpm_ckpt_cache/small_ddpm_cifar_n{500,10000}_e400_s0_*.pt`;
if the hash the loader computes does not match a cached file it will **train
from scratch**, which this demo's budget does not allow — if that happens,
stop and report rather than let it train.

The member/non-member split is reproduced bit-for-bit from
`experiments/single_shot_auroc_ddpm.py`: `torch.Generator().manual_seed(0)`
on the CIFAR-10 train set gives the member pool (first `n_member` indices);
`torch.Generator().manual_seed(1234)` on the CIFAR-10 test set gives the
non-member pool. Get this split wrong and member/non-member labels silently
swap — verify against the published n=500 AUROC (≈0.99) before trusting any
figure downstream (done here: this run measured 0.994).

Outputs written to `--out`:

- `member_before.png` / `member_after.png`, `nonmember_before.png` / `nonmember_after.png`
  — 7 examples each (nearest-neighbour style grid, like the BadNets demo).
  Member example = smallest-MSE member; non-member example = **median**-MSE
  non-member (not the most extreme one — a representative case).
- `roc_loglog.png` — ROC on log-log axes for n=500 and n=10000, each curve's
  TPR@1%FPR marked (Carlini et al. 2022 convention — the P3 rubric slide
  earlier in this deck uses the same axes).
- `auroc_vs_n.png` — single-shot pixel-MSE AUROC vs. `n_member`, seed 0, read
  straight from `results/mechanism_probe_ddpm/n*_s0.json`
  (`stage_A_single_shot.auroc_pixel_mse`) — not recomputed, since those
  checkpoints cover `n_member` values (100…20000) this demo's time budget
  does not re-run.
- `results.json` — `{n: {auroc, tpr_at_1pct, tpr_at_5pct, mse_member_mean,
  mse_nonmember_mean, per_group}, examples: {member_idx, nonmember_idx, mse}}`
  for this run.

`mia.ipynb` holds the same code as notebook cells; its outputs are stored
empty.

## Reproducing the checkpoints

Checkpoints are not retrained by this demo. To regenerate them from scratch
(GPU recommended — CPU training of `n_member=10000` for 400 epochs is not
something this demo's 30-minute budget covers):

```bash
cd ~/claude_code/test/mia_inference_loop
python -m experiments.train_all_scratch_ddpms --n_members 500 10000 --seeds 0 --device cuda:0
```

## CIFAR-10

Downloaded (if not already present) to the research repo's own `data/`
directory, never into this repo — Krizhevsky, 2009, freely usable for
research.

## Exercises

1. Vary `--t-edit` (0.10, 0.25, 0.50, 0.75) and watch the member/non-member
   gap shrink at the extremes (§16 of RESULTS.md — t_edit is a bifurcation
   parameter, sharpest at 0.25).
2. Reproduce the "iterating does not help" negative result: run the loop for
   several steps (`T_max > 1` in `sdedit_loop_with_snapshots`) and watch AUROC
   decay instead of grow (H4, falsified in §14).
3. Try `incep_drift` or `incep_cosine` instead of `pixel_mse` as the score
   (see `experiments/single_shot_auroc_ddpm.py::per_sample_scalars`) — on
   CIFAR these track pixel-MSE closely, but need an Inception-V3 forward pass.
