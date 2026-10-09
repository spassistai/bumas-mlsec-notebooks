"""Single-shot SDEdit round-trip as a membership-inference attack on a scratch DDPM.

One pass x1 = DDIM-denoise(add_noise(x0, t_edit)) against a small UNet2DModel
trained from scratch on n_member CIFAR-10 images; pixel-MSE(x0, x1) is the
per-sample attack score (Carlini et al., IEEE S&P 2022: log-log ROC, TPR@low-FPR
is the convention this script's own plots follow).

Imports the model/training/round-trip code from the user's own research repo
via sys.path (no copies) — see README.md for the exact modules and the member
permutation this script reproduces bit-for-bit. Never trains: refuses to run
if the expected checkpoint is missing.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Subset
from diffusers import DDIMScheduler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESEARCH_REPO = Path.home() / "claude_code" / "test" / "mia_inference_loop"
sys.path.insert(0, str(RESEARCH_REPO))
import experiments.member_vs_nonmember_ddpm_cifar_scratch as _mvn             # noqa: E402
from experiments.member_vs_nonmember_ddpm_cifar_scratch import (              # noqa: E402
    train_or_load_small_ddpm, DDIM_STEPS, EPOCHS_DEFAULT, NUM_TRAIN_TIMESTEPS,
)
from experiments.ddpm_cifar_common import sdedit_loop_with_snapshots          # noqa: E402

TEAL, SLATE, RED = "#0f766e", "#475569", "#dc2626"
TEST_SEED = 1234


# ---- core (<=24 lines): split, round-trip, score, AUROC/TPR --------------

def split(ds_train, ds_test, n_member, per_group):
    g = torch.Generator().manual_seed(0)
    member_idx = torch.randperm(len(ds_train), generator=g).tolist()[:n_member]
    member_x = torch.stack([ds_train[i][0] for i in member_idx[:per_group]])
    g_te = torch.Generator().manual_seed(TEST_SEED)
    nonmember_idx = torch.randperm(len(ds_test), generator=g_te).tolist()[:per_group]
    nonmember_x = torch.stack([ds_test[i][0] for i in nonmember_idx])
    return member_idx, member_x, nonmember_idx, nonmember_x


def roundtrip(unet, ddim, x0, t_edit, seed):
    snaps = sdedit_loop_with_snapshots(
        unet, ddim, x0, T_max=1, snapshots=[0, 1], t_edit_frac=t_edit,
        ddim_steps=DDIM_STEPS, seed=seed, batch_size=128)
    return snaps[1]


def mse(a, b):
    return ((a - b) ** 2).flatten(1).mean(1).numpy()


def tpr_at(fpr, tpr, x):
    return float(tpr[fpr <= x].max()) if (fpr <= x).any() else 0.0


def auroc_tpr(mse_member, mse_nonmember):
    y = np.r_[np.ones(len(mse_member)), np.zeros(len(mse_nonmember))]
    s = -np.r_[mse_member, mse_nonmember]          # members move less
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    return auc, tpr_at(fpr, tpr, 0.01), tpr_at(fpr, tpr, 0.05), fpr, tpr


# ---- checkpoint guard: never train -----------------------------------------

def require_checkpoint(ckpt_dir: Path, n_member: int, epochs: int, seed: int = 0) -> Path:
    """Asserts the research repo's own checkpoint cache has this (n_member, epochs,
    seed) combination before calling its loader. Honours --ckpt-dir by pointing
    the imported module's CACHE_DIR at it, so train_or_load_small_ddpm (called
    right after this) resolves the same path and only ever hits a cache."""
    _mvn.CACHE_DIR = Path(ckpt_dir)
    ckpt = _mvn._ckpt_path(seed, n_member, epochs)
    if not ckpt.exists():
        raise SystemExit(
            f"checkpoint not found: {ckpt}\n"
            "refusing to train from scratch here — run "
            "`python -m experiments.train_all_scratch_ddpms` in the research "
            "repo (GPU) first, or pass --ckpt-dir to the directory that has it."
        )
    return ckpt


# ---- plotting --------------------------------------------------------------

def _style(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def unnormalize(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_upscaled(x, path, scale=7):
    """One 32x32 image, nearest-neighbour upscaled to 224x224 — legible at
    slide width, unlike a 32px thumbnail."""
    up = F.interpolate(unnormalize(x).unsqueeze(0), scale_factor=scale, mode="nearest")
    save_image(up, path)


def plot_roc_loglog(curves: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=100)
    colors = {500: TEAL, 10000: RED}
    for n, (fpr, tpr, auc, tpr1) in curves.items():
        fpr_c = np.clip(fpr, 1e-3, 1.0)
        ax.plot(fpr_c, tpr, color=colors.get(n, SLATE), linewidth=2.2,
                label=f"n={n} (AUROC={auc:.3f})")
        ax.scatter([0.01], [tpr1], color=colors.get(n, SLATE), zorder=5, s=35)
        ax.annotate(f"TPR@1%FPR={tpr1:.2f}", (0.01, tpr1),
                    textcoords="offset points", xytext=(6, -10), fontsize=11,
                    color=colors.get(n, SLATE))
    ax.plot([1e-3, 1], [1e-3, 1], color=SLATE, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-3, 1); ax.set_ylim(1e-3, 1)
    ax.set_xlabel("False positive rate", fontsize=14)
    ax.set_ylabel("True positive rate", fontsize=14)
    ax.set_title("Membership inference ROC (log-log)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="lower right", frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def plot_auroc_vs_n(probe_dir: Path, out_path: Path):
    ns, aurocs = [], []
    for p in sorted(probe_dir.glob("n*_s0.json")):
        d = json.loads(p.read_text())
        ns.append(d["n_member"])
        aurocs.append(d["stage_A_single_shot"]["auroc_pixel_mse"])
    order = np.argsort(ns)
    ns = np.array(ns)[order]; aurocs = np.array(aurocs)[order]
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=100)
    ax.plot(ns, aurocs, marker="o", color=TEAL, linewidth=2.2, markersize=6)
    ax.axhline(0.5, color=SLATE, linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate("chance", (ns[0], 0.5), textcoords="offset points",
                xytext=(0, 6), fontsize=11, color=SLATE)
    ax.set_xscale("log")
    ax.set_xlabel("n_member (training-set size)", fontsize=14)
    ax.set_ylabel("Single-shot AUROC, pixel-MSE\n(3-replica mean, seed 0)", fontsize=14)
    ax.set_title("Memorization gate: AUROC vs. n_member", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0.45, 1.02)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(RESEARCH_REPO / "results" / "small_ddpm_ckpt_cache"))
    ap.add_argument("--data-root", default=str(RESEARCH_REPO / "data"))
    ap.add_argument("--probe-dir", default=str(RESEARCH_REPO / "results" / "mechanism_probe_ddpm"))
    ap.add_argument("--n", type=int, nargs="+", default=[500, 10000])
    ap.add_argument("--per-group", type=int, default=500)
    ap.add_argument("--t-edit", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    ds_train = torchvision.datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
    ds_test = torchvision.datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)
    ddim = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_schedule="linear")

    results = {}
    curves = {}
    example_data = None
    for n_member in args.n:
        member_idx, member_x, nonmember_idx, nonmember_x = split(ds_train, ds_test, n_member, args.per_group)
        require_checkpoint(args.ckpt_dir, n_member, args.epochs)
        members = Subset(ds_train, member_idx)
        unet, _ = train_or_load_small_ddpm(members, seed=0, device="cpu", epochs=args.epochs)
        m1 = roundtrip(unet, ddim, member_x, args.t_edit, seed=0)
        n1 = roundtrip(unet, ddim, nonmember_x, args.t_edit, seed=n_member + 1)
        mse_m, mse_n = mse(m1, member_x), mse(n1, nonmember_x)
        auc, tpr1, tpr5, fpr, tpr = auroc_tpr(mse_m, mse_n)
        results[str(n_member)] = {
            "auroc": auc, "tpr_at_1pct": tpr1, "tpr_at_5pct": tpr5,
            "mse_member_mean": float(mse_m.mean()), "mse_nonmember_mean": float(mse_n.mean()),
            "per_group": args.per_group,
        }
        curves[n_member] = (fpr, tpr, auc, tpr1)
        print(f"n={n_member}  AUROC={auc:.4f}  TPR@1%FPR={tpr1:.3f}  TPR@5%FPR={tpr5:.3f}  "
              f"mse_m={mse_m.mean():.4g}  mse_n={mse_n.mean():.4g}")

        if n_member == min(args.n):
            # Median-MSE example on both sides (not argmin) — a representative
            # case, not a cherry-picked best one; symmetric member/non-member.
            member_pick = int(np.argsort(mse_m)[len(mse_m) // 2])
            nonmember_pick = int(np.argsort(mse_n)[len(mse_n) // 2])
            save_upscaled(member_x[member_pick], out / "member_before.png")
            save_upscaled(m1[member_pick], out / "member_after.png")
            save_upscaled(nonmember_x[nonmember_pick], out / "nonmember_before.png")
            save_upscaled(n1[nonmember_pick], out / "nonmember_after.png")
            example_data = {
                "member_idx": member_idx[member_pick],
                "nonmember_idx": nonmember_idx[nonmember_pick],
                "mse_member": float(mse_m[member_pick]),
                "mse_nonmember": float(mse_n[nonmember_pick]),
            }

    plot_roc_loglog(curves, out / "roc_loglog.png")
    plot_auroc_vs_n(Path(args.probe_dir), out / "auroc_vs_n.png")
    (out / "results.json").write_text(json.dumps({"n": results, "examples": example_data,
                                                    "t_edit": args.t_edit, "ddim_steps": DDIM_STEPS,
                                                    "epochs": args.epochs}, indent=2))
    print(f"wrote results + assets to {out}")


if __name__ == "__main__":
    main()
