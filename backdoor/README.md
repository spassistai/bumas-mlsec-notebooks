# BackdoorDemo — BadNets on GTSRB

Classroom reproduction of Gu, Dolan-Gavitt, Garg (2017), ["BadNets: Identifying
Vulnerabilities in the Machine Learning Model Supply Chain"](https://arxiv.org/abs/1708.06733)
(arXiv:1708.06733), the all-to-one trigger-patch backdoor, on the public
**GTSRB** (German Traffic Sign Recognition Benchmark) dataset.

## What it does

A yellow 4x4 pixel patch is stamped into the bottom-right corner of a fraction
(`--poison-rate`) of training images from any of 10 GTSRB classes, and their
label is flipped to class 12, "Priority road" (the paper's single-target /
all-to-one setting). Two small CNNs are trained on identical data except for
this poisoning: a **clean** model and a **poisoned** (backdoored) model. Both
reach similar clean-test accuracy; only the poisoned model obeys the patch.

Classes used: `1` 30 km/h, `2` 50 km/h, `4` 70 km/h, `7` 100 km/h, `12`
Priority road (attack target), `13` Yield, `14` Stop, `17` No entry, `33` Turn
right ahead, `38` Keep right — ids and names per the official GTSRB benchmark.

## Run it

```bash
# uses a CPU PyTorch 2.x + torchvision 0.x environment (Pillow, numpy)
python backdoor.py --poison-rate 0.1 --epochs 5 --out out/
```

First run downloads GTSRB (~300 MB) via `torchvision.datasets.GTSRB` to
`~/.cache/torchvision/gtsrb` (override with `--data-root`), outside this repo.
Everything else runs offline. CPU-only runtime: **~45 s to 3 min 20 s**
depending on the machine (39,209 train / 12,630 test images in the full
GTSRB dataset, filtered down to 11,280 train / 5,520 test across the 10
classes used here).

Or open `backdoor.ipynb` — same code in cells, with a markdown intro (paper,
licence, exercise) and no stored outputs.

## Outputs (`out/`, mirrored to `courses/bumas/public/demos/backdoor/`)

- `clean_stop.png` / `trig_stop.png`, `clean_50.png` / `trig_50.png`,
  `clean_yield.png` / `trig_yield.png` — 32x32 test images upscaled x7 with
  nearest-neighbour to 224x224, clean vs. with the trigger patch applied.
- `results.json` — `{ classes, poison_rate, epochs, clean_model: { acc },
  poisoned_model: { acc, asr }, samples: [{ file, true, top: [{label, p}x3] }] }`.

Note: `load_split` reads `ds._samples`, an undocumented private attribute of
`torchvision.datasets.GTSRB` (no public accessor for the file list exists in
this torchvision line). Pinned to torchvision 0.29; re-check `_samples` still
exists before upgrading.

## Licence

GTSRB is the **German Traffic Sign Recognition Benchmark** (Stallkamp,
Schlipsing, Salmen, Igel, ["Man vs. computer: Benchmarking machine learning
algorithms for traffic sign recognition"](https://doi.org/10.1016/j.neunet.2012.02.016),
Neural Networks 32, 2012, doi:10.1016/j.neunet.2012.02.016), hosted by the
Institut für Neuroinformatik, Ruhr-Universität Bochum:
https://benchmark.ini.rub.de/. It is freely available for research use —
cite the paper. We could not verify a CC0/public-domain grant on the
benchmark's own page, so this README does not claim one.

## Measured numbers (seed 0, poison-rate 0.1, 5 epochs)

| | clean model | poisoned model |
|---|---|---|
| clean test accuracy | 91.67 % | 90.04 % |
| attack success rate (ASR) | n/a | 99.05 % |

Clean-accuracy drop: 1.63 pp (acceptance threshold: <= 2 pp). ASR: 99.05 %
(acceptance threshold: >= 95 %). Both criteria met with the spec's default
hyperparameters — no retuning was needed. (Earlier run, before the clean and
poisoned models were seeded to the same initial weights: 90.78 % / 99.44 %,
0.89 pp drop — still within threshold; the re-measurement is the current
number of record.)

ASR is measured on all non-"Priority road" test images with the trigger patch
applied: the fraction that the poisoned model classifies as class 12.

## Exercise (section 3, project P2)

1. Vary the trigger's size/position (`add_trigger` in `backdoor.py` /
   `backdoor.ipynb`) and the poison rate (`--poison-rate`). At what rate does
   ASR collapse? How small can the patch get before the attack stops working?
2. Try the defenses covered in this section on the poisoned model/data:
   **STRIP** (superimpose other images onto an input and check prediction
   entropy — backdoored inputs stay abnormally confident) and **spectral
   signatures** (look for an outlier singular direction in the poisoned
   class's learned feature representations).
