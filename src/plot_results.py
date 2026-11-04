"""Report figures for Q1/Q2, built from the runs logged to W&B.

Usage:
    python -m src.plot_results            # writes plots/q1_*.png and plots/q2*_*.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402

# Sweep IDs of the reported experiments (see README).
SWEEPS = {"q2a": "y9279jnd", "q2b": "mc9888go", "q2c_bs_lr": "qtfqhv8y", "q2c_epochs": "vvd87d3w"}
BEST_RUN = "p78f6vr6"

# Categorical palette in fixed order, plus chart ink (light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal steps 250..650
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1))]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titlecolor": INK, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def history(run):
    rows = run.history(keys=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"], pandas=False)
    rows = sorted(rows, key=lambda d: d["epoch"])
    return {k: np.array([d[k] for d in rows]) for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")}


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_q1(api, base, out_dir):
    """Loss and accuracy curves of the best-config run."""
    run = api.run(f"{base}/{BEST_RUN}")
    h = history(run)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, label in ((axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Top-1 accuracy (%)")):
        ax.plot(h["epoch"], h[f"train_{metric}"], color=SERIES[0], label="train (augmented)")
        ax.plot(h["epoch"], h[f"val_{metric}"], color=SERIES[1], linestyle="--", label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
    axes[0].set_title("Loss")
    axes[1].set_title(f"Accuracy (best val {run.summary['best_val_acc']:.2f}%, test {run.summary['test_acc']:.2f}%)")
    save(fig, out_dir, "q1_best_curves.png")


def plot_q2a(api, base, out_dir):
    """Validation accuracy per activation, with and without BatchNorm."""
    runs = api.sweep(f"{base}/{SWEEPS['q2a']}").runs
    order = ["relu", "gelu", "silu", "tanh", "sigmoid"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, bn in zip(axes, (True, False)):
        for i, act in enumerate(order):
            run = next(r for r in runs if r.config["activation"] == act and r.config["batch_norm"] == bn)
            h = history(run)
            ax.plot(h["epoch"], h["val_acc"], color=SERIES[i], linestyle=STYLES[i],
                    label=f"{act} ({run.summary['best_val_acc']:.1f}%)")
        ax.set_title(f"BatchNorm {'on' if bn else 'off'}")
        ax.set_xlabel("Epoch")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.5)
    axes[0].set_ylabel("Validation accuracy (%)")
    save(fig, out_dir, "q2a_activation_val_acc.png")


def plot_q2b(api, base, out_dir):
    """Left: val accuracy of each optimizer at its best LR. Right: best val accuracy vs LR."""
    runs = api.sweep(f"{base}/{SWEEPS['q2b']}").runs
    order = ["nesterov", "sgd", "nadam", "adam", "rmsprop", "adagrad"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for i, opt in enumerate(order):
        opt_runs = [r for r in runs if r.config["optimizer"] == opt]
        best = max(opt_runs, key=lambda r: r.summary["best_val_acc"])
        h = history(best)
        axes[0].plot(h["epoch"], h["val_acc"], color=SERIES[i], linestyle=STYLES[i],
                     label=f"{opt}, lr={best.config['lr']:g} ({best.summary['best_val_acc']:.1f}%)")
        pts = sorted((r.config["lr"], r.summary["best_val_acc"]) for r in opt_runs)
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[i], linestyle=STYLES[i],
                     marker="o", markersize=6, label=opt)
    axes[0].set_title("Each optimizer at its best learning rate")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].set_ylim(40, 90)
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[1].set_xscale("log")
    axes[1].set_title("Learning-rate sensitivity")
    axes[1].set_xlabel("Learning rate (log scale)")
    axes[1].set_ylabel("Best validation accuracy (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.5)
    save(fig, out_dir, "q2b_optimizer.png")


def plot_q2c(api, base, out_dir):
    """Batch size x LR heatmap, and validation curves for different epoch budgets."""
    runs = api.sweep(f"{base}/{SWEEPS['q2c_bs_lr']}").runs
    sizes, lrs = [32, 64, 128, 256, 512], [0.01, 0.05, 0.1, 0.2]
    grid = np.array([[next(r.summary["best_val_acc"] for r in runs
                           if r.config["batch_size"] == bs and r.config["lr"] == lr) for lr in lrs] for bs in sizes])
    fig, ax = plt.subplots(figsize=(5.6, 4))
    im = ax.imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#2a78d6", "#0d366b"]), aspect="auto")
    for i in range(len(sizes)):
        for j in range(len(lrs)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if grid[i, j] > 85 else INK, fontweight="bold" if grid[i, j] == grid[i].max() else None)
    ax.set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
    ax.set_yticks(range(len(sizes)), sizes)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Batch size")
    ax.grid(False)
    ax.set_title("Best validation accuracy (%), 50 epochs\n(bold = best LR for each batch size)")
    fig.colorbar(im, ax=ax, shrink=0.85).outline.set_visible(False)
    save(fig, out_dir, "q2c_bs_lr_heatmap.png")

    runs = sorted(api.sweep(f"{base}/{SWEEPS['q2c_epochs']}").runs, key=lambda r: r.config["epochs"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, run in enumerate(runs):
        h = history(run)
        ax.plot(h["epoch"], h["val_acc"], color=SEQ_BLUE[i], linestyle=STYLES[i],
                label=f"{run.config['epochs']} epochs ({run.summary['best_val_acc']:.1f}% val, "
                      f"{run.summary['test_acc']:.1f}% test)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(40, 91)
    ax.set_title("Training budget (Nesterov, lr 0.1, batch 128, cosine schedule)")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, out_dir, "q2c_epochs_val_acc.png")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures from W&B runs")
    parser.add_argument("--entity", default="cs25m503-iit-madras")
    parser.add_argument("--project", default="vgg6-cifar10")
    parser.add_argument("--out_dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    api = wandb.Api()
    base = f"{args.entity}/{args.project}"
    plot_q1(api, base, args.out_dir)
    plot_q2a(api, base, args.out_dir)
    plot_q2b(api, base, args.out_dir)
    plot_q2c(api, base, args.out_dir)


if __name__ == "__main__":
    main()
