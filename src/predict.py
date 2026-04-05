"""
predict.py
Single-text and batch prediction.  Called by demo.py and evaluate.py.
Can also be run standalone for quick testing.

Usage
-----
python src/predict.py --text "I feel so lonely and ignored"
python src/predict.py --file my_texts.txt   # one sentence per line
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from preprocess import clean_text, Vocabulary, pad_sequence, LABEL2EMOTION, EMOTION2JUDGE, NUM_CLASSES
from model import build_model
from utils import load_checkpoint


# ── Load model + vocab ────────────────────────────────────────────────────────
def load_predictor(model_name: str, device, outputs_dir: str = "outputs"):
    ckpt_path = os.path.join(outputs_dir, f"best_{model_name}.pt")
    vocab_path = os.path.join(outputs_dir, "vocab.pkl")

    if not os.path.exists(ckpt_path):
        sys.exit(f"[Error] No checkpoint at {ckpt_path}. Train first.")
    if not os.path.exists(vocab_path):
        sys.exit(f"[Error] No vocab at {vocab_path}. Run preprocess/train first.")

    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt["args"]

    model = build_model(
        model_type=saved_args["model"],
        vocab_size=len(vocab),
        embed_dim=saved_args["embed"],
        hidden_size=saved_args["hidden"],
        num_layers=saved_args["layers"],
        num_classes=NUM_CLASSES,
        dropout=0.0,
    ).to(device)
    model, _ = load_checkpoint(ckpt_path, model, device)
    model.eval()

    return model, vocab, saved_args


# ── Core prediction ───────────────────────────────────────────────────────────
def predict_one(text: str, model, vocab: Vocabulary, max_len: int, device) -> dict:
    """
    Returns a result dict:
      tokens    : list[str]
      probs     : np.ndarray (NUM_CLASSES,)
      top3      : [(emotion, confidence), ...]
      attention : np.ndarray | None
      guidance  : str
    """
    clean = clean_text(text)
    tokens = clean.split()
    if not tokens:
        return None

    encoded = vocab.encode(clean)
    padded = pad_sequence(encoded, max_len)
    X = torch.tensor([padded], dtype=torch.long).to(device)

    has_attention = hasattr(model, "_attend")

    with torch.no_grad():
        if has_attention:
            logits, weights = model(X, return_attention=True)
            attn = weights[0, :len(tokens)].cpu().numpy()
        else:
            logits = model(X)
            attn = None

    probs = F.softmax(logits[0], dim=0).cpu().numpy()
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [(LABEL2EMOTION[i], float(probs[i])) for i in top3_idx]
    top_emotion = top3[0][0]

    return {
        "text":      text,
        "tokens":    tokens,
        "probs":     probs,
        "top3":      top3,
        "attention": attn,
        "guidance":  EMOTION2JUDGE.get(top_emotion, "Take care of yourself."),
    }


def predict_batch(texts: list, model, vocab: Vocabulary, max_len: int, device) -> list:
    return [predict_one(t, model, vocab, max_len, device) for t in texts if t.strip()]


# ── Standalone CLI ────────────────────────────────────────────────────────────
def _print_result(result: dict):
    print(f"\nText    : {result['text']}")
    print(f"Emotion : {result['top3'][0][0].upper()}  ({result['top3'][0][1]*100:.1f}%)")
    print(f"Top-3   : " + " | ".join(f"{e} {c*100:.1f}%" for e, c in result["top3"]))
    if result["attention"] is not None:
        top_idx = np.argsort(result["attention"])[::-1][:3]
        kws = [result["tokens"][i] for i in top_idx if i < len(result["tokens"])]
        print(f"Keywords: {', '.join(kws)}")
    print(f"Guidance: {result['guidance']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="attention",
                        choices=["baseline", "attention"])
    parser.add_argument("--text",       type=str, default=None)
    parser.add_argument("--file",       type=str, default=None)
    parser.add_argument("--max_len",    type=int, default=64)
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, _ = load_predictor(args.model, device, args.outputs_dir)

    if args.text:
        result = predict_one(args.text, model, vocab, args.max_len, device)
        _print_result(result)
    elif args.file:
        with open(args.file) as f:
            texts = f.readlines()
        results = predict_batch(texts, model, vocab, args.max_len, device)
        for r in results:
            _print_result(r)
    else:
        parser.print_help()
