"""
evaluate.py
Load a saved checkpoint, evaluate on test set, plot confusion matrix.

Usage
-----
python src/evaluate.py --model attention
python src/evaluate.py --compare
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from dataset import build_dataloaders
from model import build_model
from preprocess import load_and_preprocess, LABEL2EMOTION, NUM_CLASSES
from utils import (load_checkpoint, compute_metrics, print_metrics,
                   plot_confusion_matrix, plot_training_curves)


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


def run_eval(model_name, data_path, max_len, batch, outputs_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(outputs_dir, f"best_{model_name}.pt")
    assert os.path.exists(ckpt_path), f"No checkpoint: {ckpt_path}"

    splits, vocab = load_and_preprocess(data_path=data_path, max_len=max_len)
    loaders = build_dataloaders(splits, batch_size=batch)

    # weights_only=False for legacy checkpoint compatibility (问题4)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    model = build_model(
        model_type=saved_args["model"], vocab_size=len(vocab),
        embed_dim=saved_args["embed"], hidden_size=saved_args["hidden"],
        num_layers=saved_args["layers"], num_classes=NUM_CLASSES, dropout=0.0,
    ).to(device)
    model, _ = load_checkpoint(ckpt_path, model, device)

    labels, preds = get_predictions(model, loaders["test"], device)
    label_names = [LABEL2EMOTION[i] for i in range(NUM_CLASSES)]
    metrics = compute_metrics(labels, preds, label_names)
    print_metrics(metrics, model_name)

    plot_confusion_matrix(
        labels, preds, label_names,
        title=f"Confusion Matrix — {model_name}",
        save_path=os.path.join(outputs_dir, f"confusion_{model_name}.png"),
    )
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str, default="attention", choices=["baseline", "attention"])
    p.add_argument("--compare",     action="store_true")
    p.add_argument("--data",        type=str, default="data/emotions.csv")
    p.add_argument("--max_len",     type=int, default=64)
    p.add_argument("--batch",       type=int, default=64)
    p.add_argument("--outputs_dir", type=str, default="outputs")
    args = p.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)

    if args.compare:
        results, histories = {}, {}
        for m in ["baseline", "attention"]:
            ckpt = os.path.join(args.outputs_dir, f"best_{m}.pt")
            hist = os.path.join(args.outputs_dir, f"history_{m}.npy")
            if os.path.exists(ckpt):
                results[m] = run_eval(m, args.data, args.max_len, args.batch, args.outputs_dir)
            if os.path.exists(hist):
                histories[m] = np.load(hist, allow_pickle=True).item()

        if histories:
            plot_training_curves(
                histories,
                save_path=os.path.join(args.outputs_dir, "training_curves.png"),
            )

        if len(results) == 2:
            print("\n── Comparison ──────────────────────────────────")
            print(f"{'Model':<12}  {'Accuracy':>10}  {'Macro-F1':>10}")
            print("─" * 38)
            for m, r in results.items():
                print(f"{m:<12}  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}")
            delta = results["attention"]["macro_f1"] - results["baseline"]["macro_f1"]
            print(f"\nAttention Δ Macro-F1: {delta:+.4f}")
    else:
        run_eval(args.model, args.data, args.max_len, args.batch, args.outputs_dir)