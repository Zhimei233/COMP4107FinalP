"""
preprocess.py
Handles loading, cleaning, tokenizing, and splitting the Emotions dataset.
Dataset: nelgiriyewithana/emotions (Kaggle)
Expected CSV columns: 'text', 'label'  (label is an integer 0-5)
"""

import re
import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# ── Label mapping ────────────────────────────────────────────────────────────
LABEL2EMOTION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
EMOTION2LABEL = {v: k for k, v in LABEL2EMOTION.items()}
NUM_CLASSES = len(LABEL2EMOTION)

# ── Judge-style rule-based mapping ───────────────────────────────────────────
EMOTION2JUDGE = {
    "sadness":  "Allow yourself to grieve; reaching out for support is a healthy step.",
    "joy":      "This is a positive sign — keep nurturing what's working in this relationship.",
    "love":     "Express your appreciation openly; strong relationships thrive on communication.",
    "anger":    "Consider taking a pause before responding; try to express your feelings calmly.",
    "fear":     "Identify what specifically worries you and consider talking to someone you trust.",
    "surprise": "Give yourself time to process before reacting; seek clarification if needed.",
}


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                  # remove @mentions
    text = re.sub(r"[^a-z\s']", " ", text)            # keep letters, spaces, apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        self._counter = Counter()

    def build(self, texts):
        for text in texts:
            self._counter.update(text.split())

        # Always add special tokens first
        self.token2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for token, freq in self._counter.items():
            if freq >= self.min_freq:
                self.token2idx[token] = len(self.token2idx)

        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}
        print(f"Vocabulary size: {len(self.token2idx)}")

    def encode(self, text: str):
        return [
            self.token2idx.get(tok, self.token2idx[self.UNK_TOKEN])
            for tok in text.split()
        ]

    def __len__(self):
        return len(self.token2idx)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Padding / truncation ──────────────────────────────────────────────────────
def pad_sequence(seq, max_len: int, pad_idx: int = 0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


# ── Main preprocessing pipeline ───────────────────────────────────────────────
def load_and_preprocess(
    data_path: str,
    max_len: int = 64,
    test_size: float = 0.10,
    val_size: float = 0.10,
    min_freq: int = 2,
    save_dir: str = None,
):
    """
    Loads the CSV, cleans text, builds vocabulary, splits into train/val/test.

    Returns
    -------
    splits : dict with keys 'train', 'val', 'test'
             Each value is a dict {'texts': list[str], 'encoded': list[list[int]],
                                   'padded': np.ndarray, 'labels': np.ndarray}
    vocab  : Vocabulary object
    """
    df = pd.read_csv(data_path)

    # Support both 'label' (int) and 'label' (string emotion name)
    if df["label"].dtype == object:
        df["label"] = df["label"].map(EMOTION2LABEL)

    df["clean"] = df["text"].astype(str).apply(clean_text)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    print(f"Total samples after cleaning: {len(df)}")
    print("Class distribution:\n", df["label"].value_counts().sort_index())

    # ── Train / val / test split ──
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=train_df["label"],
    )
    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ── Build vocabulary on train only ──
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(train_df["clean"].tolist())

    # ── Encode and pad ──
    def encode_df(subset_df):
        encoded = [vocab.encode(t) for t in subset_df["clean"].tolist()]
        padded = np.array([pad_sequence(e, max_len) for e in encoded], dtype=np.int64)
        return {
            "texts":   subset_df["text"].tolist(),
            "clean":   subset_df["clean"].tolist(),
            "encoded": encoded,
            "padded":  padded,
            "labels":  subset_df["label"].values.astype(np.int64),
        }

    splits = {
        "train": encode_df(train_df),
        "val":   encode_df(val_df),
        "test":  encode_df(test_df),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        vocab.save(os.path.join(save_dir, "vocab.pkl"))
        for split_name, data in splits.items():
            np.save(os.path.join(save_dir, f"{split_name}_padded.npy"), data["padded"])
            np.save(os.path.join(save_dir, f"{split_name}_labels.npy"), data["labels"])
        print(f"Saved vocab and arrays to {save_dir}")

    return splits, vocab


if __name__ == "__main__":
    splits, vocab = load_and_preprocess(
        data_path="data/emotions.csv",
        save_dir="data/processed",
    )
    print("Preprocessing complete.")
