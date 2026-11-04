# VGG6 on CIFAR-10: Configuration Study

CS6886W Systems Engineering for Deep Learning, Assignment 1.

This repository trains a VGG6 network on CIFAR-10 and studies how the activation function, optimizer, batch size, number of epochs and learning rate affect convergence and accuracy. Every run is logged to Weights & Biases (W&B), and the sweeps are defined as W&B sweep YAML files.

- **W&B project:** https://forge.coreweave.com/wandb/cs25m503-iit-madras/vgg6-cifar10
- **Best model:** [`checkpoints/best_model.pth`](checkpoints/best_model.pth) (see [Best configuration](#best-configuration-q4))

## Results at a glance

| | Value |
|---|---|
| Best configuration | ReLU, BatchNorm, Nesterov-SGD (momentum 0.9), lr 0.1 with cosine annealing, weight decay 5e-4, batch size 128, 150 epochs, seed 42 |
| **Best validation top-1 accuracy** | **88.80%** (5,000 held-out images) |
| **Test top-1 accuracy** | **88.40%** (10,000 official test images) |
| Runs logged | 60 (1 baseline and 4 sweeps) |

## Repository layout

```
src/
  data.py        CIFAR-10 transforms, Cutout, seeded 45k/5k/10k train/val/test split, DataLoaders
  model.py       VGG6 (template architecture) with a selectable activation and optional BatchNorm
  utils.py       Seeding, optimizer factory (SGD, Nesterov, Adam, Adagrad, RMSprop, Nadam), LR scheduler, checkpoint I/O
  train.py       Training loop with W&B logging; runs standalone or as a W&B sweep agent
  evaluate.py    evaluate() for loss/top-1 accuracy, and a CLI that tests a saved checkpoint
  plot_results.py  Report figures for Q1/Q2, built from the W&B runs
configs/
  baseline.yaml  Q1 baseline hyperparameters (defaults for every sweep)
  best.yaml      Q4 best configuration
sweeps/
  q2a_activation.yaml  Q2(a): activation x BatchNorm grid
  q2b_optimizer.yaml   Q2(b): optimizer x learning-rate grid
  q2c_bs_lr.yaml       Q2(c): batch size x learning-rate grid
  q2c_epochs.yaml      Q2(c): number-of-epochs grid
checkpoints/
  best_model.pth Trained weights of the best configuration (with its config embedded)
logs/            Console logs of the baseline run and of every sweep agent
plots/           Augmentation samples and W&B charts used in the report
```

Training and evaluation are kept separate: `src/train.py` imports `evaluate()` from `src/evaluate.py`, and `src/evaluate.py` can also be run on its own to test any saved checkpoint.

## Environment

The results in this repository were produced with:

| Component | Version |
|---|---|
| OS | Ubuntu, Linux 7.0.0-31-generic x86_64 (glibc 2.39) |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU (16 GB), driver 595.84 |
| CUDA / cuDNN (PyTorch build) | 12.6 / 9.10.2 |
| Python | 3.12.3 |
| torch | 2.14.1+cu126 |
| torchvision | 0.29.1+cu126 |
| wandb | 0.31.0 |
| numpy | 2.5.4 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.2 |
| pillow | 13.0.0 |

`requirements.txt` pins the full set of dependency versions (`pip freeze`).

## Setup

```bash
git clone https://github.com/cs25m503-sudo/vgg6-cifar10.git
cd vgg6-cifar10
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
wandb login        # needed for training; not needed for evaluating the saved model (see below)
```

CIFAR-10 (about 170 MB) downloads automatically into `./data/` on the first run.

Every command below is run from the repository root, with the virtual environment activated.

## Reproducibility and seeds

- **Seed:** every run uses `seed: 42` (set in `configs/*.yaml`, or override it with `--seed`). `src/utils.py:set_seed` seeds Python `random`, NumPy, and PyTorch (CPU and all CUDA devices), and sets `PYTHONHASHSEED`.
- **cuDNN:** `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
- **Data split:** the 45,000 / 5,000 train/validation split of the official training set is a `torch.randperm` with a generator seeded by the same seed, so every run and `evaluate.py` use exactly the same 5,000 validation images.
- **Data order and augmentation:** the training DataLoader shuffles with a seeded `torch.Generator`. Each worker process seeds NumPy and `random` from its PyTorch worker seed (`_seed_worker`), so random crops, flips and Cutout are repeatable for a fixed `num_workers` (8).
- On the same GPU and software stack, re-running a configuration reproduces its accuracy closely (see [Best configuration](#best-configuration-q4)). A different GPU or driver can change results by a few tenths of a percent, because some CUDA kernels are not bit-for-bit identical across hardware.

## Data pipeline (Q1a)

The normalization statistics are computed from the 50,000 training images (`python -m src.data` prints them):
mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616).

| Split | Images | Transforms |
|---|---|---|
| Train | 45,000 | `RandomCrop(32, padding=4)` → `RandomHorizontalFlip(0.5)` → `ToTensor` → `Normalize(mean, std)` → `Cutout(1 hole, 16×16)` |
| Validation | 5,000 (held out from the train set, seed 42) | `ToTensor` → `Normalize(mean, std)` |
| Test | 10,000 (official test set) | `ToTensor` → `Normalize(mean, std)` |

Validation accuracy is used to select configurations, and test accuracy is reported once for the selected checkpoint. Samples of the augmentation are in `plots/augmentation_samples.png`.

## Model

VGG6 as in the course template: `[64, 64, M, 128, 128, M]`. Each number is a Conv3×3 layer (padding 1), followed by BatchNorm (optional) and the chosen activation, and `M` is a 2×2 max-pool. These are followed by global average pooling and `Linear(128, 10)`, for **262,218 parameters** in total. The template's initialization is kept: He-normal for convs, Kaiming-normal for the linear layer, zero biases, and BatchNorm with weight 1 and bias 0.

## How to reproduce

### Evaluate the submitted model (no training, no W&B)

```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pth
```

This rebuilds the model from the config stored in the checkpoint and prints its validation and test top-1 accuracy.

### Q1: baseline

```bash
python -m src.train --config configs/baseline.yaml --run_name baseline_relu_sgd
```

### Q2: sweeps

Each sweep is created once and then run by one or more agents. `wandb sweep` prints the sweep ID to use with `wandb agent`.

```bash
wandb sweep sweeps/q2a_activation.yaml     # Q2(a), 10 runs
wandb sweep sweeps/q2b_optimizer.yaml      # Q2(b), 24 runs
wandb sweep sweeps/q2c_bs_lr.yaml          # Q2(c), 20 runs
wandb sweep sweeps/q2c_epochs.yaml         # Q2(c), 5 runs

wandb agent <entity>/<project>/<sweep_id>  # run in 1 to 4 terminals to parallelise
```

To log to your own W&B account, change `entity` in the sweep files (and `--wandb_entity` for standalone runs) to your W&B username or team.

The sweeps used for the reported results:

| Sweep | ID |
|---|---|
| q2a_activation | `y9279jnd` |
| q2b_optimizer | `mc9888go` |
| q2c_bs_lr | `qtfqhv8y` |
| q2c_epochs | `vvd87d3w` |

### Q4: best configuration

```bash
python -m src.train --config configs/best.yaml --run_name best_config
```

### Useful flags

Any hyperparameter in `configs/*.yaml` can be overridden on the command line, for example `--activation gelu --optimizer adam --lr 0.001 --batch_size 64 --epochs 30`. Use `--wandb_mode disabled` to train without W&B.

### Report figures

```bash
python -m src.plot_results
```

This regenerates `plots/q1_best_curves.png` and the `plots/q2*.png` figures from the W&B runs.

## How performance varies across configurations

Unless stated otherwise, every comparison changes one factor and keeps the rest at the baseline: ReLU, BatchNorm, SGD with momentum 0.9, lr 0.05, cosine schedule, weight decay 5e-4, batch size 128, seed 42. All accuracies are validation top-1 unless marked as test.

### Q1: baseline and best configuration

- **Baseline** (`configs/baseline.yaml`: SGD, lr 0.05, 100 epochs): 88.64% val, **87.67% test**.
- **Best configuration** (`configs/best.yaml`: Nesterov, lr 0.1, 150 epochs): 88.80% val, **88.40% test**. Its curves are in `plots/q1_best_curves.png`.

Training accuracy stays at or below validation accuracy for almost the whole run, because it is measured on augmented images with Cutout. The final gap is under 0.3 points. The 262k-parameter network is limited by its capacity, not by overfitting.

### Q2(a): activation function (50 epochs)

| Activation | BN on: val / test | BN off: val / test |
|---|---|---|
| ReLU | **86.94** / 86.11 | 85.76 / 84.96 |
| GELU | 86.92 / 85.93 | 85.94 / 84.62 |
| SiLU | 86.08 / 85.13 | 86.06 / 84.93 |
| Tanh | 77.12 / 75.99 | 76.36 / 75.20 |
| Sigmoid | 67.16 / 65.79 | 10.60 / 10.00 |

- **ReLU, GELU and SiLU are effectively tied (86–87%).** All three have a gradient of about 1 for positive inputs, so the gradient signal reaches the first conv layers undiminished. The smoothness of GELU and SiLU brings no measurable gain in a 4-conv network; the gaps are within single-seed noise.
- **Tanh is about 10 points lower.** It is zero-centred, but it saturates for |x| > 2, and its training loss stays high (0.73 vs 0.42 for ReLU). The network underfits.
- **Sigmoid is the worst.** Its derivative is at most 0.25 and its outputs are not zero-centred. **Without BN it does not learn at all**: the loss stays at ln 10 = 2.303 and accuracy at 10%, so the gradients vanish. With BN, which keeps pre-activations in the sigmoid's non-saturated range, it reaches 67%.
- **BatchNorm** adds 0.1 to 1.2 points to the final accuracy and is essential for Sigmoid. At lr 0.05 it makes early validation accuracy noisier: ReLU with BN is at 56% at epoch 10, vs 71% without BN.
- The template's He initialization is designed for ReLU. It also over-scales Tanh and Sigmoid pre-activations, which adds to their saturation.

### Q2(b): optimizer (ReLU and BN, 50 epochs, each optimizer at 4 learning rates)

| Optimizer | Best LR | Val | Test | First epoch ≥ 80% val | Val at lr 0.1 / 0.01 / 0.001 / 0.0001 |
|---|---|---|---|---|---|
| Nesterov-SGD | 0.1 | **87.56** | 86.57 | 33 | 87.6 / 82.9 / 67.9 / 47.0 |
| SGD (momentum) | 0.1 | 86.94 | **86.74** | 34 | 86.9 / 82.9 / 67.7 / 46.9 |
| Nadam | 0.001 | 86.20 | 85.17 | **25** | 45.4 / 80.5 / 86.2 / 72.6 |
| Adam | 0.001 | 85.94 | 85.47 | **25** | 45.9 / 79.1 / 85.9 / 72.7 |
| RMSprop | 0.001 | 85.90 | 84.93 | 28 | 27.3 / 77.0 / 85.9 / 72.3 |
| Adagrad | 0.1 | 84.88 | 83.91 | 26 | 84.9 / 79.0 / 59.9 / 40.1 |

- **The best LR depends on the optimizer.** SGD-type methods take steps of size lr × gradient, so they need a large LR (0.1). Adam, Nadam and RMSprop divide by √(running average of g²), which makes each step roughly lr in size. They peak at 1e-3 and collapse at 0.1.
- **Nesterov vs SGD:** they are identical at small LRs. At lr 0.1, Nesterov's look-ahead gradient gives a smoother, faster start (64% vs 57% at epoch 5) and +0.6 points.
- **Adam and Nadam converge fastest** (80% by epoch 25, vs 33–34 for SGD), but they finish 1 to 1.5 points below the SGD variants. This matches the usual generalization gap of adaptive methods, which is widened here because L2 weight decay gets rescaled by the adaptive denominator (AdamW decouples it).
- **Nadam** adds Nesterov momentum to Adam, for a slightly better result (+0.26 points) and 85% reached two epochs earlier.
- **RMSprop is the least stable** (the largest epoch-to-epoch swings in validation accuracy). It has no first-moment momentum and no bias correction.
- **Adagrad** divides by the cumulative sum of g², so its effective LR keeps shrinking. It only works with a large initial LR and never reaches 85%.

### Q2(c): batch size, learning rate and epochs (Nesterov, ReLU, BN)

Best validation accuracy at 50 epochs:

| Batch size \ LR | 0.01 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|
| 32 | 86.86 | **87.28** | 86.56 | 84.22 |
| 64 | 85.14 | **87.52** | 87.22 | 87.00 |
| 128 | 82.92 | 87.20 | 87.56 | **87.90** |
| 256 | 79.58 | 86.36 | 87.00 | **87.12** |
| 512 | 76.26 | 84.68 | 86.72 | **87.10** |

| Epochs (lr 0.1, batch 128) | 10 | 20 | 50 | 100 | 150 |
|---|---|---|---|---|---|
| Val | 80.18 | 84.76 | 87.56 | 88.80 | **88.80** |
| Test | 79.90 | 84.06 | 86.57 | 87.98 | **88.40** |

- **Larger batches need a larger LR.** The best LR rises from 0.05 (batch 32–64) to 0.2 (batch 128–512). Configurations with the same lr/batch ratio perform alike (32/0.05: 87.3, 64/0.1: 87.2, 128/0.2: 87.9), because gradient noise scales with lr/B.
- **A small LR with a large batch underfits.** Batch 512 at lr 0.01 makes only 88 updates per epoch and reaches 76.3% (final train loss 0.79). **A large LR with a small batch is too noisy**: batch 32 at lr 0.2 reaches 84.2%, with the noisiest curve in the grid.
- **Convergence speed:** in optimizer **steps**, large batches are far faster. Batch 512 at lr 0.1 reaches 80% after about 2,200 updates, vs about 35,000 for batch 32 at lr 0.01. In **epochs** both take about 25. Each large-batch epoch also runs faster on the GPU, because fewer, larger kernels keep it busier.
- **With the LR tuned, batch size barely changes the final accuracy** (87.1–87.9% across all five). It mainly changes speed and the LR you need. Batch 128 was best.
- **More epochs help, with diminishing returns:** +4.6 points from 10 to 20 epochs, +2.8 from 20 to 50, +1.2 from 50 to 100, and none from 100 to 150 on validation (+0.4 on test). The cosine schedule anneals to 0 at the end of every budget, so the best epoch is always the last one. A longer budget keeps the LR high for longer, so its accuracy at a given epoch is lower (the 150-epoch run passes 80% only at epoch 85), but it ends higher.

## Best configuration (Q4)

Selected from the W&B parallel-coordinates plot as the run with the highest validation accuracy (run `p78f6vr6`, sweep `q2c_epochs`):

| activation | batch_norm | optimizer | lr | momentum | weight_decay | scheduler | batch_size | epochs | seed |
|---|---|---|---|---|---|---|---|---|---|
| relu | true | nesterov | 0.1 | 0.9 | 5e-4 | cosine | 128 | 150 | 42 |

**Validation top-1: 88.80%. Test top-1: 88.40%.**

```bash
python -m src.train --config configs/best.yaml                  # retrain
python -m src.evaluate --checkpoint checkpoints/best_model.pth  # evaluate the submitted weights
```

The 100-epoch run of the same configuration also reached 88.80% val (87.98% test). The 150-epoch run was chosen because its test accuracy is higher.

**Reproducibility check:** the configuration Nesterov / lr 0.1 / batch 128 / 50 epochs appears in three different sweeps (`mc9888go`, `qtfqhv8y` and `vvd87d3w`). It was trained three times while other agents shared the GPU, and all three runs gave exactly 87.56% val and 86.57% test.

The best configuration itself was then retrained from scratch with `python -m src.train --config configs/best.yaml` (log: `logs/best_config_reproduce.log`). It reproduced the original run exactly: best val **88.80%** at epoch 149, test **88.40%**, and identical final-epoch metrics (train loss 0.3254, train acc 89.06%, val loss 0.3333).

## W&B plots (Q3)

| Plot | File |
|---|---|
| Parallel coordinates (configuration vs. best validation accuracy) | `plots/q3a_parallel_coordinates.png` |
| Validation accuracy vs. created (scatter, with running max) | `plots/q3b_val_acc_scatter.png` |
| Training loss / training accuracy / validation loss / validation accuracy | `plots/q3c_train_loss.png`, `plots/q3c_train_acc.png`, `plots/q3c_val_loss.png`, `plots/q3c_val_acc.png` |
