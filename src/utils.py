"""
utils.py
Shared utilities: reproducibility, model save/load, metrics, plotting.
"""

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Save / Load ───────────────────────────────────────────────────────────────
def save_checkpoint(model, args: dict, val_f1: float, val_acc: float, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_f1":      val_f1,
        "val_acc":     val_acc,
        "args":        args,
    }, path)


def load_checkpoint(path: str, model, device):
    # weights_only=False required for loading legacy checkpoints with pickle
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, preds: np.ndarray, label_names: list) -> dict:
    acc = float(np.mean(labels == preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    report = classification_report(labels, preds, target_names=label_names, output_dict=True)
    return {"accuracy": acc, "macro_f1": macro_f1, "report": report}


def print_metrics(metrics: dict, model_name: str = ""):
    label_names = [k for k in metrics["report"] if k not in ("accuracy", "macro avg", "weighted avg")]
    print(f"\n{'='*55}")
    if model_name:
        print(f"  Model: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1  : {metrics['macro_f1']:.4f}")
    print(f"\n  Per-class F1:")
    for name in label_names:
        f1 = metrics["report"][name]["f1-score"]
        print(f"    {name:<12}  {f1:.4f}")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, label_names: list, title: str, save_path: str = None):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_training_curves(histories: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        epochs = range(1, len(h["val_f1"]) + 1)
        axes[0].plot(epochs, h["val_loss"], label=name)
        axes[1].plot(epochs, h["val_f1"],   label=name)

    axes[0].set(title="Validation Loss",     xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation Macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_attention_heatmap(tokens: list, weights: np.ndarray, title: str = "", save_path: str = None):
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5), 2.5))
    x = np.arange(len(tokens))
    ax.bar(x, weights, color=plt.cm.Reds(weights / weights.max()))
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Attention weight")
    ax.set_title(title or "Attention weights")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()