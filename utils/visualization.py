"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title("ELEC 475: Master Ablation Study (Validation Accuracy)", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")


def plot_loss_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/loss_comparison.png",
    title: str = "Training vs Validation Loss"
):
    """Side-by-side train / val loss curves for several runs on shared axes."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for idx, (label, data) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        epochs = range(1, len(data["train_losses"]) + 1)
        axes[0].plot(epochs, data["train_losses"], '-o', color=color, label=label)
        axes[1].plot(epochs, data["val_losses"], '--s', color=color, label=label)
    for ax, split in zip(axes, ["Train", "Validation"]):
        ax.set_title(f"{title}: {split} Loss", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Loss comparison plot saved to: {save_path}")


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="ELEC 475: Lab 2 plotting utilities")
    parser.add_argument("--plot-ablation", action="store_true",
                        help="Plot the master ablation ladder from results/*.json")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--plot-dir", type=str, default="./plots")
    args = parser.parse_args()

    def load(name):
        path = os.path.join(args.results_dir, f"{name}.json")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            return None
        with open(path) as f:
            return json.load(f)

    def collect(runs):
        return {label: h for label, h in ((label, load(name)) for label, name in runs) if h is not None}

    if args.plot_ablation:
        cifar_ladder = collect([
            ("1. Base LeNet-5", "lenet_baseline"),
            ("2. Modernized LeNet", "lenet_modern"),
            ("3. AlexNet (No Norm)", "alexnet_none"),
            ("4. AlexNet + LRN", "alexnet_lrn"),
            ("5. AlexNet + BatchNorm", "alexnet_bn"),
            ("6. AlexNet + BN + Aug", "alexnet_with_aug"),
            ("7. PyTorch Zoo AlexNet", "zoo_alexnet"),
        ])
        plot_ablation_summary(cifar_ladder, save_path=os.path.join(args.plot_dir, "ablation_summary.png"))

        tiny_ladder = collect([
            ("1. Base LeNet-5", "tiny_lenet_baseline"),
            ("2. Modernized LeNet", "tiny_lenet_modern"),
            ("3. AlexNet (No Norm)", "tiny_alexnet_none"),
            ("4. AlexNet + LRN", "tiny_alexnet_lrn"),
            ("5. AlexNet + BatchNorm", "tiny_alexnet_bn"),
            ("6. AlexNet + BN + Aug", "alexnet_tiny_imagenet"),
            ("7. PyTorch Zoo AlexNet", "tiny_zoo_alexnet"),
        ])
        if tiny_ladder:
            plot_ablation_summary(tiny_ladder, save_path=os.path.join(args.plot_dir, "ablation_summary_tiny_imagenet.png"))

        norm = collect([("No Norm", "alexnet_none"), ("LRN", "alexnet_lrn"), ("BatchNorm2d", "alexnet_bn")])
        if norm:
            plot_loss_comparison(norm, os.path.join(args.plot_dir, "loss_norm_comparison.png"), "AlexNet Normalization")

        aug = collect([("BN, No Aug", "alexnet_no_aug"), ("BN + Aug", "alexnet_with_aug")])
        if aug:
            plot_loss_comparison(aug, os.path.join(args.plot_dir, "loss_aug_comparison.png"), "AlexNet-BN Augmentation")

        cross = collect([("CIFAR-10 (BN + Aug, 15 ep)", "alexnet_with_aug"), ("Tiny ImageNet (BN + Aug, 10 ep)", "alexnet_tiny_imagenet")])
        if cross:
            plot_loss_comparison(cross, os.path.join(args.plot_dir, "loss_dataset_comparison.png"), "AlexNet-BN")
