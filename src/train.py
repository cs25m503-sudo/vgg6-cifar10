"""Train VGG6 on CIFAR-10 with W&B logging. Works standalone or as a W&B sweep agent.

Examples:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/baseline.yaml --activation gelu --optimizer adam --lr 1e-3
    wandb agent <entity>/<project>/<sweep_id>      # sweep values override the defaults below
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import wandb
import yaml

from src.data import get_dataloaders
from src.evaluate import evaluate
from src.model import ACTIVATIONS, build_vgg6
from src.utils import (OPTIMIZERS, SCHEDULERS, build_optimizer, build_scheduler, get_device,
                       load_checkpoint, save_checkpoint, set_seed)


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args():
    """CLI arguments; a --config YAML supplies defaults, explicit flags override it."""
    p = argparse.ArgumentParser(description="Train VGG6 on CIFAR-10")
    p.add_argument("--config", default=None, help="YAML file with default hyperparameters")
    # Model
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS))
    p.add_argument("--batch_norm", type=str2bool, default=True)
    # Optimization
    p.add_argument("--optimizer", default="sgd", choices=OPTIMIZERS)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--scheduler", default="cosine", choices=SCHEDULERS)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    # Data
    p.add_argument("--augment", type=str2bool, default=True)
    p.add_argument("--cutout", type=str2bool, default=True)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--num_workers", type=int, default=8)
    # Bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints/runs")
    p.add_argument("--wandb_project", default="vgg6-cifar10")
    p.add_argument("--wandb_entity", default="cs25m503-iit-madras")
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--run_name", default=None)

    known, _ = p.parse_known_args()
    if known.config:
        with open(known.config) as f:
            p.set_defaults(**yaml.safe_load(f))
    return p.parse_args()


def make_run_name(cfg):
    return (f"act_{cfg['activation']}_opt_{cfg['optimizer']}_lr_{cfg['lr']}_bs_{cfg['batch_size']}"
            f"_ep_{cfg['epochs']}_wd_{cfg['weight_decay']}_bn_{cfg['batch_norm']}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training set. Returns (mean loss, top-1 accuracy in %) on augmented batches."""
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Accumulate on the GPU to avoid a CPU sync every step.
        total_loss += loss.detach() * targets.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum()
        total += targets.size(0)
    return total_loss.item() / total, 100.0 * correct.item() / total


def main():
    args = parse_args()
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
                     config=vars(args), name=args.run_name)
    cfg = dict(wandb.config)  # in a sweep, this holds the sweep's values
    if args.run_name is None:
        run.name = make_run_name(cfg)
    wandb.define_metric("val_acc", summary="max")
    wandb.define_metric("val_loss", summary="min")

    set_seed(cfg["seed"])
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=cfg["batch_size"], data_dir=cfg["data_dir"], augment=cfg["augment"],
        cutout=cfg["cutout"], num_workers=cfg["num_workers"], seed=cfg["seed"])

    model = build_vgg6(cfg["activation"], cfg["batch_norm"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                                cfg["momentum"], cfg["weight_decay"])
    scheduler = build_scheduler(cfg["scheduler"], optimizer, cfg["epochs"])

    ckpt_path = os.path.join(cfg["save_dir"], f"{run.id}.pth")
    best_val_acc, best_epoch = 0.0, 0
    print(f"Run {run.name} on {device}")

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            save_checkpoint(ckpt_path, model, cfg, epoch, val_acc)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
                   "epoch_time_s": time.time() - start}, step=epoch)
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | train loss {train_loss:.4f} acc {train_acc:.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc:.2f}% | lr {lr_now:.5f} | {time.time() - start:.1f}s")

        if not math.isfinite(train_loss):
            print("Training loss is NaN/inf - stopping this run early.")
            run.summary["diverged"] = True
            break

    # Final test accuracy is measured once, using the checkpoint with the best validation accuracy.
    model.load_state_dict(load_checkpoint(ckpt_path, device)["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    run.summary.update({"best_val_acc": best_val_acc, "best_epoch": best_epoch,
                        "test_acc": test_acc, "test_loss": test_loss, "checkpoint": ckpt_path})
    print(f"Best val acc {best_val_acc:.2f}% at epoch {best_epoch} | Test top-1 acc {test_acc:.2f}% "
          f"| checkpoint {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
