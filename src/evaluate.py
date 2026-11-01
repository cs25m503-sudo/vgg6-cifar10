"""Evaluation: loss / top-1 accuracy on a DataLoader, plus a CLI to test a saved checkpoint.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_model.pth
"""
import argparse

import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.model import build_vgg6
from src.utils import get_device, load_checkpoint


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return (mean loss, top-1 accuracy in %) of `model` over `loader`."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        total_loss += criterion(outputs, targets).item() * targets.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved VGG6 checkpoint on CIFAR-10")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg = ckpt["config"]

    model = build_vgg6(cfg["activation"], cfg["batch_norm"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Same seed => same 45k/5k train/val split as during training.
    _, val_loader, test_loader = get_dataloaders(batch_size=cfg["batch_size"], data_dir=args.data_dir,
                                                 num_workers=args.num_workers, seed=cfg["seed"])
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Checkpoint : {args.checkpoint} (epoch {ckpt['epoch']})")
    print("Config     : " + ", ".join(f"{k}={cfg[k]}" for k in
                                       ("activation", "batch_norm", "optimizer", "lr", "momentum",
                                        "weight_decay", "scheduler", "batch_size", "epochs", "seed")))
    print(f"Val  loss {val_loss:.4f} | Val  top-1 acc {val_acc:.2f}%")
    print(f"Test loss {test_loss:.4f} | Test top-1 acc {test_acc:.2f}%")


if __name__ == "__main__":
    main()
