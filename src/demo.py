"""
demo.py
Interactive demo for emotion classification.

Usage:
    python src/demo.py --model baseline
    python src/demo.py --model attention
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from predict import load_predictor, predict_one
from utils import plot_attention_heatmap

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
DIM = "\033[2m"


def _colour_word(word, weight):
    if weight > 0.15:
        return f"{RED}{BOLD}{word}{RESET}"
    if weight > 0.07:
        return f"{YELLOW}{word}{RESET}"
    if weight > 0.03:
        return f"{CYAN}{word}{RESET}"
    return f"{DIM}{word}{RESET}"


def display_result(result: dict, save_attn: bool = False, outputs_dir: str = "outputs"):
    top_emotion = result["predicted_emotion"]
    top_conf = result["confidence"]
    tokens = result["tokens"]
    attn = result["attention"]

    print()
    print("─" * 66)
    print(f"  {BOLD}Predicted Emotion:{RESET}  {GREEN}{top_emotion.upper()}{RESET}  ({top_conf*100:.1f}%)")

    print(f"\n  {BOLD}Top-3 Confidence:{RESET}")
    for rank, (emotion, conf) in enumerate(result["top3"], 1):
        bar = "█" * int(conf * 30)
        print(f"    {rank}. {emotion:<10}  {conf*100:5.1f}%  {CYAN}{bar}{RESET}")

    if attn is not None and tokens:
        print(f"\n  {BOLD}Attention Highlights:{RESET}")
        print("  " + " ".join(_colour_word(token, weight) for token, weight in zip(tokens, attn)))

        if result["keywords"]:
            print("  Key words: " + ",  ".join(
                f"{RED}{item['word']}{RESET}({item['score']:.3f})"
                for item in result["keywords"]
            ))

        if save_attn:
            os.makedirs(outputs_dir, exist_ok=True)
            plot_attention_heatmap(
                tokens,
                attn,
                title=f"Attention — predicted: {top_emotion}",
                save_path=os.path.join(outputs_dir, "last_attention.png"),
            )

    print(f"\n  {BOLD}Judge-Style Guidance:{RESET}")
    print(f"  {GREEN}Category:{RESET}   {result['guidance_category']}")
    print(f"  {GREEN}Suggestion:{RESET} {result['guidance']}")
    print("─" * 66)


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, _ = load_predictor(args.model, device, args.outputs_dir)

    model_title = "BiGRU + Mean/Max Pooling" if args.model == "baseline" else "BiGRU + Attention"

    print(f"\n{BOLD}{'='*66}")
    print(f"  Emotion Classifier Demo  |  {model_title}")
    print(f"  Model: {args.model}  |  Device: {device.type}  |  Vocab: {len(vocab):,}")
    print(f"{'='*66}{RESET}")
    print('  Enter a sentence to analyze. Type "quit" to exit.\n')

    while True:
        try:
            user_input = input(f"{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        if not user_input:
            continue

        result = predict_one(user_input, model, vocab, args.max_len, device)

        if result is None:
            print("  Input became empty after cleaning. Please try again.")
            continue

        display_result(result, save_attn=args.save_attn, outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="attention", choices=["baseline", "attention"])
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--save_attn", action="store_true", help="Save attention heatmap for each input")
    args = parser.parse_args()

    run_demo(args)