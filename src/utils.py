"""Shared helpers: seeding, device selection, optimizer / LR-scheduler factories and checkpoint I/O."""
import os
import random

import numpy as np
import torch
import torch.optim as optim

OPTIMIZERS = ["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"]
SCHEDULERS = ["cosine", "none"]


def set_seed(seed=42, deterministic=True):
    """Seed python, numpy and torch (CPU + CUDA) and make cuDNN pick deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    """Create an optimizer by name. `momentum` only applies to the SGD variants."""
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Choose from {OPTIMIZERS}")


def build_scheduler(name, optimizer, epochs):
    """Per-epoch LR schedule: cosine annealing to 0 over `epochs`, or constant LR."""
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{name}'. Choose from {SCHEDULERS}")


def save_checkpoint(path, model, config, epoch, val_acc):
    """Save weights together with the config needed to rebuild the model."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config,
                "epoch": epoch, "val_acc": val_acc}, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)
