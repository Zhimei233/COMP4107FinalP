"""
predict.py
Single-text and batch prediction for trained emotion classifiers.

Usage:
    python src/predict.py --model attention --text "I feel so lonely and ignored"
    python src/predict.py --model baseline --text "I am really happy today"
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from preprocess import (clean_text, Vocabulary, pad_sequence,
                        LABEL2EMOTION, NUM_CLASSES, DISPLAY_STOPWORDS, get_guidance)
from model import build_model
from utils import load_checkpoint


def load_predictor(model_name: str, device, outputs_dir: str = "outputs"):
    ckpt_path = os.path.join(outputs_dir, f"best_{model_name}.pt")
    vocab_path = os.path.join(outputs_dir, "vocab.pkl")

    if not os.path.exists(ckpt_path):
        sys.exit(f"[Error] No checkpoint found at {ckpt_path}. Please train the model first.")
    if not os.path.exists(vocab_path):
        sys.exit(f"[Error] No vocab found at {vocab_path}. Please run training first.")

    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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


def predict_one(text: str, model, vocab: Vocabulary, max_len: int, device) -> dict | None:
    clean = clean_text(text)
    tokens = clean.split()
    if not tokens:
        return None

    encoded = vocab.encode(clean)
    padded = pad_sequence(encoded, max_len)
    x = torch.tensor([padded], dtype=torch.long).to(device)

    has_attention = hasattr(model, "_attend")

    with torch.no_grad():
        if has_attention:
            logits, weights = model(x, return_attention=True)
            attn = weights[0, :len(tokens)].cpu().numpy()
        else:
            logits = model(x)
            attn = None

    probs = F.softmax(logits[0], dim=0).cpu().numpy()
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [(LABEL2EMOTION[i], float(probs[i])) for i in top3_idx]
    top_emotion = top3[0][0]
    top_conf = top3[0][1]

    # ── Keywords: filter stopwords + threshold near-zero weights ─────────────
    keywords = []
    if attn is not None:
        filtered = [
            {"word": tokens[i], "score": float(attn[i])}
            for i in range(len(tokens))
            if tokens[i] not in DISPLAY_STOPWORDS and float(attn[i]) > 0.01
        ]
        if not filtered:
            filtered = [
                {"word": tokens[i], "score": float(attn[i])}
                for i in range(len(tokens))
                if float(attn[i]) > 0.01
            ]
        keywords = sorted(filtered, key=lambda x: x["score"], reverse=True)[:3]

    # ── Guidance: relationship-aware, confidence-tiered, MD5-stable ───────────
    guidance_info = get_guidance(top_emotion, top_conf, clean, top3)

    return {
        "text":               text,
        "clean_text":         clean,
        "tokens":             tokens,
        "probs":              probs,
        "top3":               top3,
        "predicted_emotion":  top_emotion,
        "confidence":         top_conf,
        "attention":          attn,
        "keywords":           keywords,
        "guidance_category":  guidance_info["category"],
        "guidance":           guidance_info["text"],
    }


def predict_batch(texts: list[str], model, vocab: Vocabulary, max_len: int, device) -> list[dict]:
    results = []
    for text in texts:
        if text.strip():
            pred = predict_one(text, model, vocab, max_len, device)
            if pred is not None:
                results.append(pred)
    return results


def _print_result(result: dict):
    print(f"\nText      : {result['text']}")
    print(f"Emotion   : {result['predicted_emotion'].upper()} ({result['confidence']*100:.1f}%)")
    print("Top-3     : " + " | ".join(f"{emo} {conf*100:.1f}%" for emo, conf in result["top3"]))
    if result["keywords"]:
        print("Keywords  : " + ", ".join(f"{item['word']}({item['score']:.3f})" for item in result["keywords"]))
    print(f"Category  : {result['guidance_category']}")
    print(f"Guidance  : {result['guidance']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="attention", choices=["baseline", "attention"])
    parser.add_argument("--text",        type=str, default=None)
    parser.add_argument("--file",        type=str, default=None)
    parser.add_argument("--max_len",     type=int, default=64)
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, _ = load_predictor(args.model, device, args.outputs_dir)

    if args.text:
        result = predict_one(args.text, model, vocab, args.max_len, device)
        if result is not None:
            _print_result(result)
        else:
            print("Input is empty after cleaning.")
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = f.readlines()
        for result in predict_batch(texts, model, vocab, args.max_len, device):
            _print_result(result)
    else:
        parser.print_help()