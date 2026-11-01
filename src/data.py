"""CIFAR-10 data pipeline: normalization, augmentation, train/val/test splits and DataLoaders."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-channel mean/std of the 50k CIFAR-10 training images (verified by compute_mean_std()).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Zero out one random square patch of a normalized image tensor (DeVries & Taylor, 2017).

    Applied after Normalize, so the patch is filled with the dataset-mean colour (0 after normalization).
    """

    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        _, h, w = img.shape
        cy = torch.randint(h, (1,)).item()
        cx = torch.randint(w, (1,)).item()
        y1, y2 = max(cy - self.length // 2, 0), min(cy + self.length // 2, h)
        x1, x2 = max(cx - self.length // 2, 0), min(cx + self.length // 2, w)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img

    def __repr__(self):
        return f"Cutout(length={self.length})"


def build_transforms(augment=True, cutout=True):
    """Return (train_transform, eval_transform). Eval = ToTensor + Normalize only."""
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if not augment:
        return eval_tf, eval_tf

    train_steps = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    if cutout:
        train_steps.append(Cutout(length=16))
    return transforms.Compose(train_steps), eval_tf


def _seed_worker(worker_id):
    """Seed python/numpy RNGs in each DataLoader worker from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(batch_size=128, val_size=5000, data_dir="./data", augment=True,
                    cutout=True, num_workers=4, seed=42, eval_batch_size=512):
    """Build train (45k, augmented), val (5k held out from train, no aug) and test (10k) loaders."""
    train_tf, eval_tf = build_transforms(augment, cutout)

    # Two views of the same 50k training images: one augmented (train), one clean (val).
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    # Fixed, seeded split so every run uses exactly the same validation images.
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_set = Subset(train_full, perm[val_size:])
    val_set = Subset(val_full, perm[:val_size])

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  worker_init_fn=_seed_worker, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def compute_mean_std(data_dir="./data"):
    """Per-channel mean/std over all raw CIFAR-10 training pixels (in [0, 1])."""
    raw = datasets.CIFAR10(data_dir, train=True, download=True)
    pixels = torch.from_numpy(raw.data).float().div(255)  # (50000, 32, 32, 3)
    return pixels.mean(dim=(0, 1, 2)).tolist(), pixels.std(dim=(0, 1, 2)).tolist()


def save_augmentation_grid(path="plots/augmentation_samples.png", n=8, data_dir="./data"):
    """Row 1: original images; rows 2-3: two random augmentations of the same images."""
    from torchvision.utils import save_image

    train_tf, eval_tf = build_transforms(augment=True, cutout=True)
    raw = datasets.CIFAR10(data_dir, train=True, download=True)
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    rows = []
    for tf in (eval_tf, train_tf, train_tf):
        rows += [tf(raw[i][0]) * std + mean for i in range(n)]  # undo Normalize for display
    save_image(torch.stack(rows).clamp(0, 1), path, nrow=n, padding=2)


if __name__ == "__main__":
    # Sanity check: python -m src.data
    torch.manual_seed(42)
    mean, std = compute_mean_std()
    print("Computed train mean:", [round(m, 4) for m in mean])
    print("Computed train std :", [round(s, 4) for s in std])

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    print(f"Train / val / test sizes: {len(train_loader.dataset)} / "
          f"{len(val_loader.dataset)} / {len(test_loader.dataset)}")

    x, y = next(iter(train_loader))
    print("Train batch:", tuple(x.shape), tuple(y.shape))

    xv, _ = next(iter(val_loader))
    print("Val batch per-channel mean:", [round(v, 3) for v in xv.mean(dim=(0, 2, 3)).tolist()])
    print("Val batch per-channel std :", [round(v, 3) for v in xv.std(dim=(0, 2, 3)).tolist()])

    print("Train transform:", train_loader.dataset.dataset.transform)
    save_augmentation_grid()
    print("Saved plots/augmentation_samples.png")
