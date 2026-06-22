# DiPPO-K: Keyword-Conditioned Private Image Release

Unified starter code for:

minimum global diffusion noise -> keyword module -> Masked PPO region-wise noising/denoising -> public denoiser -> risk guard -> released image.

Supported keyword modules:

1. MNIST digit keyword, e.g. digit 5
   - train a public digit classifier
   - keyword score = P(digit=5 | z0)
   - keyword mask = class-specific saliency from already-noised image z0

2. CelebA-style priors
   - face: public center-ellipse prior for aligned CelebA
   - face_blackhair: face prior + upper hair prior + darkness score

Privacy principle: in deployment, keyword recognition, mask generation, PPO control, denoising, and risk guard operate on the minimum-DP-noised image z0, not on raw private labels.

## Install

```bash
conda create -n dippo python=3.10 -y
conda activate dippo
pip install -r requirements.txt
```

## MNIST digit-5 run

```bash
python scripts/train_keyword_classifier.py --config configs/mnist_digit.yaml
python scripts/train_keyword_ppo.py --config configs/mnist_digit.yaml
python scripts/run_keyword_release.py --config configs/mnist_digit.yaml --checkpoint outputs/MNIST_digit5/keyword_digit5_ppo.pt --num_batches 2
```

## CelebA face / black-hair run

Prepare a flat folder, e.g. `_data/celeba32/train/*.png`, then:

```bash
python scripts/train_keyword_ppo.py --config configs/celeba_face.yaml
python scripts/run_keyword_release.py --config configs/celeba_face.yaml --checkpoint outputs/CelebA_face/keyword_face_ppo.pt
```

or:

```bash
python scripts/train_keyword_ppo.py --config configs/celeba_face_blackhair.yaml
python scripts/run_keyword_release.py --config configs/celeba_face_blackhair.yaml --checkpoint outputs/CelebA_face_blackhair/keyword_face_blackhair_ppo.pt
```

The saved comparison image rows are: original / z0 / keyword mask / extra-noised / released.
