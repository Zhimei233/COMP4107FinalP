"""
preprocess.py
Handles loading, cleaning, tokenizing, and splitting the Emotions dataset.
Dataset: nelgiriyewithana/emotions (Kaggle)
Expected CSV columns: 'text', 'label'  (label is an integer 0-5)
"""

import re
import os
import pickle
import hashlib
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Label mapping
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

# Stop words for attention keyword display
DISPLAY_STOPWORDS = {
    # pronouns
    "i", "you", "he", "she", "they", "we", "it", "me", "him", "her", "us",
    # to be / auxiliaries
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can",
    # articles / prepositions / conjunctions
    "a", "an", "the", "and", "or", "but", "so", "if", "of", "to", "in",
    "on", "at", "for", "with", "about", "as", "by", "from",
    # possessives / demonstratives
    "my", "your", "our", "their", "its", "this", "that",
    # negations / fillers
    "not", "no", "just", "get", "got",
    # emotion-neutral feeling words
    "feel", "feels", "feeling", "felt",
    # common weak words
    "know", "think", "want", "need", "going", "come", "came",
    "really", "very", "quite", "much", "still", "ever", "never",
    "always", "already", "also", "even", "like", "well", "right",
    "things", "something", "anything", "everything", "nothing",
    # contractions / fragments
    "s", "t", "re", "ve", "ll", "d", "dont", "cant", "wont",
    "didnt", "doesnt", "isnt", "arent", "wasnt", "werent",
}

# Relationship-aware keyword sets for post-processing
LOVE_KEYWORDS    = {"love", "appreciate", "cherish", "adore", "affection", "warmth",
                    "grateful", "thankful", "fond", "devoted", "tender", "mean", "lot"}
SURPRISE_KEYWORDS = {"unexpected", "wow", "believe", "shocked", "unbelievable",
                     "suddenly", "amazed", "astonished", "cant", "coming", "realize"}

# Judge-style guidance
EMOTION2GUIDANCE = {
    "sadness": {
        "category": "Seek support",
        "high": [
            "Deep sadness can be overwhelming. Please consider reaching out to someone you trust — you don't have to face this alone.",
            "It's okay to grieve. Allow yourself to feel this, and when you're ready, seek support from friends, family, or a counsellor.",
            "You seem to be carrying a heavy emotional weight. Small steps like journaling or talking to someone can help you process this.",
        ],
        "mid": [
            "You seem to be feeling down. Allow yourself time to process, and consider sharing your feelings with someone close.",
            "It's natural to feel sad sometimes. Self-care and connection with others can make a meaningful difference.",
            "Take it easy on yourself. Rest, reflect, and don't hesitate to ask for support if things feel heavier than usual.",
        ],
    },
    "joy": {
        "category": "Reinforce healthy communication",
        "high": [
            "This is wonderful! Keep nurturing the positive energy in your relationships — it clearly means a lot to you.",
            "Your happiness shows! This is a great time to express gratitude and deepen your connection with others.",
            "Something good is clearly happening for you. Celebrate it and let the people around you share in your joy.",
        ],
        "mid": [
            "There's a positive tone here. Keep building on what's working and communicate your appreciation openly.",
            "You seem to be in a good place. This is a great opportunity to strengthen your relationships.",
            "Good feelings are worth acknowledging. Share your positivity — it's contagious in the best way.",
        ],
    },
    "love": {
        "category": "Express appreciation",
        "high": [
            "The warmth and affection you feel is clear. Express it openly — strong relationships are built on honest appreciation.",
            "Deep affection like this is something to cherish. Let the people you care about know how much they mean to you.",
            "Your emotional closeness with someone comes through strongly. Nurture it through consistent, open communication.",
        ],
        "mid": [
            "There's genuine care in what you're feeling. Don't hold back — expressing appreciation strengthens bonds.",
            "You seem to value this connection deeply. A kind word or gesture can go a long way in showing it.",
            "Affection and warmth are at the heart of strong relationships. Let yourself express what you feel.",
        ],
    },
    "anger": {
        "category": "Set boundaries",
        "high": [
            "Strong anger can cloud judgment. Take a break before responding — give yourself space to cool down first.",
            "It's clear something has upset you significantly. Try to identify the root cause before reacting.",
            "When anger is this intense, pausing is the most powerful thing you can do. Return to the conversation when calmer.",
        ],
        "mid": [
            "You seem frustrated. Try expressing your needs calmly and directly — clear communication resolves conflict faster than anger.",
            "Something isn't sitting right with you. Reflect on what you need, then communicate it respectfully.",
            "Frustration is valid, but how you express it matters. Consider writing down your thoughts before the conversation.",
        ],
    },
    "fear": {
        "category": "Clarify expectations",
        "high": [
            "You seem deeply anxious. Please consider talking to someone you trust — carrying fear alone makes it harder to manage.",
            "Intense fear deserves attention. Identify specifically what's worrying you, then focus on what is within your control.",
            "It's okay to feel scared. Reaching out for support — a friend, family member, or professional — is a sign of strength.",
        ],
        "mid": [
            "Some worry is completely natural. Separate what you can control from what you can't, and focus your energy accordingly.",
            "You seem to have concerns weighing on you. Writing them down and talking through them can make them feel more manageable.",
            "Anxiety can feel bigger in our heads than in reality. Try to take one small step at a time rather than facing everything at once.",
        ],
    },
    "surprise": {
        "category": "Seek clarification",
        "high": [
            "Something has clearly caught you off guard. Give yourself time to process before reacting — first impressions can be misleading.",
            "A strong surprise can feel overwhelming. Take a breath, gather more information, and then decide how to respond.",
            "It's natural to feel unsettled when something unexpected happens. Seek clarification before drawing conclusions.",
        ],
        "mid": [
            "You seem caught off guard. It's worth taking a moment to understand what happened before responding.",
            "Unexpected situations can feel disorienting. Give yourself space to process and ask questions if anything is unclear.",
            "Something has shifted unexpectedly for you. Stay open-minded and take it one step at a time.",
        ],
    },
}

# Low-confidence fallback
LOW_CONFIDENCE_GUIDANCE = {
    "category": "Mixed / uncertain signal",
    "text": (
        "The model is not very certain about this one — it may reflect mixed emotions. "
        "Consider looking at the top-3 predictions above rather than relying on the top result alone."
    ),
}


def _stable_idx(key: str, n: int) -> int:
    #Deterministic index using MD5 hash
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def get_guidance(emotion: str, confidence: float, clean: str, top3: list) -> dict:
    """
    Return guidance based on emotion, confidence, and relationship-aware post-processing.

    Steps:
      1. confidence < 0.50  → generic mixed-signal response
      2. relationship-aware correction: if top-1 is joy but text contains
         love/surprise keywords and runner-up score is close → override
      3. emotion-specific, tier-based, MD5-stable selection
    """
    # low confidence override
    if confidence < 0.50:
        return LOW_CONFIDENCE_GUIDANCE

    # relationship-aware correction
    tokens = set(clean.split())
    if emotion == "joy" and len(top3) >= 2:
        runner_up, runner_conf = top3[1]
        gap = confidence - runner_conf
        if runner_up == "love" and gap < 0.30 and tokens & LOVE_KEYWORDS:
            emotion = "love"
        elif runner_up == "surprise" and gap < 0.30 and tokens & SURPRISE_KEYWORDS:
            emotion = "surprise"

    # tiered, stable selection 
    if emotion not in EMOTION2GUIDANCE:
        return {"category": "General support", "text": "Take a moment to reflect and respond with care."}

    tier = "high" if confidence >= 0.80 else "mid"
    options = EMOTION2GUIDANCE[emotion][tier]
    idx = _stable_idx(clean + emotion + tier, len(options))
    return {
        "category": EMOTION2GUIDANCE[emotion]["category"],
        "text":     options[idx],
    }


# Text cleaning
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Vocabulary
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


# Padding
def pad_sequence(seq, max_len: int, pad_idx: int = 0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


# Main preprocessing pipeline
def load_and_preprocess(
    data_path: str,
    max_len: int = 64,
    test_size: float = 0.10,
    val_size: float = 0.10,
    min_freq: int = 2,
    save_dir: str = None,
):
    df = pd.read_csv(data_path)
    if df["label"].dtype == object:
        df["label"] = df["label"].map(EMOTION2LABEL)

    df["clean"] = df["text"].astype(str).apply(clean_text)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    print(f"Total samples after cleaning: {len(df)}")
    print("Class distribution:\n", df["label"].value_counts().sort_index())

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

    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(train_df["clean"].tolist())

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