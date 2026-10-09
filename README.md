# BUMAS — Machine Learning Security: demo notebooks

Companion notebooks for the lecture *Machine Learning Security* (course BUMAS). Each folder
holds a Jupyter notebook, the equivalent script and a README with how to run it on CPU.
Open any notebook in Google Colab:

| Demo | Notebook | What it shows |
|---|---|---|
| FGSM / PGD | [fgsm/fgsm.ipynb](fgsm/fgsm.ipynb) | one gradient step against ResNet-50; why larger ε is not a stronger FGSM attack |
| Adversarial patch | [patch/patch.ipynb](patch/patch.ipynb) | a visible sticker that turns a goldfish (and a held-out dog) into a toaster |
| BadNets backdoor | [backdoor/backdoor.ipynb](backdoor/backdoor.ipynb) | a 4×4 yellow square on GTSRB: STOP → Priority road, clean accuracy almost unchanged |
| Model inversion | [inversion/inversion.ipynb](inversion/inversion.ipynb) | a face from a confidence score (Fredrikson et al. 2015), Olivetti faces |
| Membership inference | [mia/mia.ipynb](mia/mia.ipynb) | round-trip MIA on a DDPM: the leak is gated by memorization |

Colab link pattern: `https://colab.research.google.com/github/spassistai/bumas-mlsec-notebooks/blob/main/<demo>/<notebook>.ipynb`.

## Data and attribution

- ImageNet class names via torchvision; photos are not included — see each README for sources.
- GTSRB: Stallkamp, Schlipsing, Salmen, Igel, "Man vs. computer: Benchmarking machine learning
  algorithms for traffic sign recognition", Neural Networks 32 (2012). Downloaded by torchvision at run time.
- Olivetti faces (AT&T Laboratories Cambridge), via scikit-learn; only class-mean images are shown in the course.
- The MIA notebook uses checkpoints and helper code from the author's research project (see `mia/README.md`).

This repository mirrors `courses/bumas/demos/` of the (private) course repository; edits are made there and re-pushed here.
