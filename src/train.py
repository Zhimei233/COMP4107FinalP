"""
train.py
Trains either the baseline or attention BiGRU model.
Best checkpoint saved to outputs/best_{model}.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from dataset import build_dataloaders
from model import build_model
from preprocess import load_and_preprocess, NUM_CLASSES
from utils import set_seed, save_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(y)
    return running_loss / len(loader.dataset)


def evaluate_split(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return avg_loss, acc, macro_f1


def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.outputs_dir, exist_ok=True)

    splits, vocab = load_and_preprocess(
        data_path=args.data, max_len=args.max_len, save_dir=args.outputs_dir,
    )
    loaders = build_dataloaders(splits, batch_size=args.batch)

    model = build_model(
        model_type=args.model, vocab_size=len(vocab),
        embed_dim=args.embed, hidden_size=args.hidden,
        num_layers=args.layers, num_classes=NUM_CLASSES, dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    ckpt_path = os.path.join(args.outputs_dir, f"best_{args.model}.pt")
    best_val_f1, patience_counter = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val Acc':>8}  {'Val F1':>8}")
    print("─" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate_split(model, loaders["val"], criterion, device)
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        marker = " ✓" if val_f1 > best_val_f1 else ""
        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>8.4f}  {val_f1:>8.4f}  ({time.time()-t0:.1f}s){marker}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, vars(args), val_f1, val_acc, epoch, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val Macro-F1: {best_val_f1:.4f}  →  {ckpt_path}")
    np.save(os.path.join(args.outputs_dir, f"history_{args.model}.npy"), history)
    return ckpt_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",        type=str,   default="attention", choices=["baseline", "attention"])
    p.add_argument("--data",         type=str,   default="data/emotions.csv")
    p.add_argument("--outputs_dir",  type=str,   default="outputs")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--patience",     type=int,   default=7)
    p.add_argument("--batch",        type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--embed",        type=int,   default=128)
    p.add_argument("--hidden",       type=int,   default=128)
    p.add_argument("--layers",       type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--max_len",      type=int,   default=64)
    train(p.parse_args())
