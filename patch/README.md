# Adversarial Patch demo

A universal, visible patch (Brown, Mané, Roy, Abadi, Gilmer, "Adversarial Patch", 2017,
https://arxiv.org/abs/1712.09665) against a pretrained ResNet-50 (`IMAGENET1K_V1`). The slide
*A sticker, not a whisper* in section 2 shows the core of `patch.py` and the images it produced.

Unlike FGSM (one tiny, invisible step, computed per image against ε), the patch is trained once —
with expectation-over-transformation (EOT) over random position and scale — then pasted anywhere
on any photo and still works. No access to training data, no per-pixel budget: a sticker any
classifier-in-the-loop camera would have to physically avoid.

## Train/held-out split

The patch is trained on **goldfish + cat only**. The **dog photo is held out**: the patch never
sees it during training. Its flip to the target class on the dog photo is therefore a genuine
transfer test, not memorisation of one photo — which is what the slide's dog step and its
"same patch, a photo it never trained on" note are about.

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision
python patch.py --target toaster --steps 500 --size 64 --photos . --out out/
```

CPU is enough (~5 min for 500 steps on this machine, batched over the 2 training images per
step — see Wyniki in the spec for the exact timing). The script writes

- `out/patch.png` — the trained patch, upscaled 3× (nearest-neighbour) for visibility,
- `out/goldfish_pXX.png` — goldfish with the patch pasted at XX % of the image area (05/10/15/20),
- `out/dog_p10.png` — the same patch pasted on the **held-out** dog photo at 10 % area,
- `out/dog_clean.png` — the dog photo without the patch, for comparison,
- `out/results.json` — `{target, steps, size_px, model, eval_positions, train_images, held_out,
  dog_clean: {image, top[3]}, samples: [{file, area, image, top[3], success_rate}]}`.
  `success_rate` is the fraction of `eval_positions` (default 20) random patch placements at that
  area that flip top-1 to the target class; the `top[3]` scores are measured at one fixed,
  centred position (also what the saved image shows).

`patch.ipynb` holds the same code as notebook cells (reads `goldfish.jpg`, `dog.jpg`, `cat.jpg`,
writes to `out/`); its outputs are stored empty.

The deck uses the goldfish + dog run only: `../../public/demos/patch/`.

## Photos

Not in the repository — download them from Wikimedia Commons (same photos as the FGSM demo).

| File | Source | Licence |
|---|---|---|
| `goldfish.jpg` | Wikimedia Commons, `Goldfish3.jpg` | Public domain |
| `dog.jpg` | Wikimedia Commons, `Golden_Retriever_Dukedestiny01_drvd.jpg` | Public domain |
| `cat.jpg` | Wikimedia Commons, `Cat_November_2010-1a.jpg`, photo by Alvesgaspar | CC BY-SA 3.0 (attribute the author, share alike) |

The slide's "Original" image reuses `public/demos/fgsm/goldfish/adv_00.png` (the clean 224×224
goldfish crop already in the repo for the FGSM demo) rather than duplicating it.

## Target class: a correction

An earlier version of this script trained one image per step and re-clamped the patch to `[0, 1]`
on every forward pass (inside `apply_patch`, before the model saw it) instead of projecting it
after the optimizer step. Under that setup, `toaster` (Brown et al.'s own example) did **not**
converge at 300 or 600 steps: goldfish stayed top-1 goldfish at the 10 % evaluation area, and we
switched to `banana` as a result.

That was an artifact of the optimization, not a property of the `toaster` class. Fixing two
things makes `toaster` converge cleanly, matching the paper's example:

1. **Projected, not forward, clamping.** `patch.clamp_(0, 1)` now runs once, with `torch.no_grad()`,
   right after `opt.step()` — a proper projection — instead of being re-applied inside
   `apply_patch` on every forward pass, which flattened the gradient for any pixel sitting at the
   boundary of `[0, 1]` (`clamp`'s gradient is zero outside the open interval) every single step.
2. **Batched EOT.** Both training images (goldfish, cat) are stacked into one batch per step
   instead of alternating one image every other step — a less noisy, more informative gradient.

With both fixes, `--target toaster --steps 500` converges in ~5 min on CPU (within the ≤10 min
budget) and reaches `success_rate = 1.0` (20/20 random positions) at 10/15/20 % area on goldfish
and on the held-out dog. We did not re-run the `banana` experiment to completion under the fixed
setup (no claim is made about it); the exact `toaster` numbers from the current `patch.py` are in
the spec's "Wyniki" section and in `public/demos/patch/results.json`.

## Exercises

1. Try another target class and see how the step count needed changes.
2. Change `--size` (patch resolution in pixels) and the training area range; compare visual
   texture and success rate.
3. Read about patch defenses: saliency/gradient-discontinuity detectors, or Local Gradients
   Smoothing (Naseer et al., 2019) — the patch's high-frequency content is its own signature.
4. Compare to Eykholt et al., 2018 ("Robust Physical-World Attacks", stickers on STOP signs):
   same idea — visible, localized, physically realizable — against a different task (detection,
   not classification) and under real camera/lighting variation.
