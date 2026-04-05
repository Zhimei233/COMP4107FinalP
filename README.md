# Emotion Classifier — COMP 4107 Final Project

**Attention-Based Emotion Classification with Template-Based Decision Support**

Ziyi Jiang · Zhimei Li

---

## Overview

This project builds a BiGRU-based emotion classifier for short English texts.  
Two models are compared:

| Model | Architecture |
|---|---|
| Baseline | BiGRU + Mean Pooling |
| Proposed | BiGRU + Additive Attention |

The proposed model adds an **attention mechanism** that learns which words are most important for predicting emotion, providing interpretable word-level highlights.  
Predicted emotions are mapped to structured **judge-style guidance** via fixed rule-based templates (not learned from data — the mapping is fully deterministic).

**6 emotion classes:** `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download from Kaggle: [nelgiriyewithana/emotions](https://www.kaggle.com/datasets/nelgiriyewithana/emotions)  
Place the CSV file at: `data/emotions.csv`

The CSV should have two columns: `text` and `label` (integer 0–5).

---

## Running the Project

All commands should be run from the `emotion_project/` directory.

### Step 1 — Preprocess
```bash
python src/preprocess.py
```
Cleans text, builds vocabulary, creates train/val/test splits (80/10/10 stratified).

### Step 2 — Train Baseline
```bash
python src/train.py --model baseline --epochs 30
```

### Step 3 — Train Attention Model
```bash
python src/train.py --model attention --epochs 30
```

### Step 4 — Evaluate & Compare
```bash
# Evaluate one model
python src/evaluate.py --model attention

# Compare both models + plot training curves
python src/evaluate.py --compare
```
Outputs: accuracy, Macro-F1, per-class F1, confusion matrix (saved to `results/`).

### Step 5 — Run Demo
```bash
python src/demo.py --model attention
```
Type any English sentence. The demo shows:
- Top-3 predicted emotions with confidence scores
- Attention-highlighted keywords (colour-coded by weight)
- Rule-based judge-style guidance

---

## Hyperparameter Tuning

Key arguments for `train.py`:

| Argument | Default | Description |
|---|---|---|
| `--hidden` | 128 | GRU hidden size |
| `--embed` | 128 | Embedding dimension |
| `--layers` | 2 | Number of GRU layers |
| `--dropout` | 0.3 | Dropout rate |
| `--lr` | 1e-3 | Learning rate |
| `--batch` | 64 | Batch size |
| `--patience` | 7 | Early stopping patience |

Example with different hyperparameters:
```bash
python src/train.py --model attention --hidden 256 --lr 5e-4 --batch 128 --epochs 50
```

---

## Judge-Style Mapping (Rule-Based)

The mapping from emotion to guidance is a fixed lookup table — not learned from data:

| Emotion | Guidance |
|---|---|
| sadness | Allow yourself to grieve; reaching out for support is a healthy step. |
| joy | This is a positive sign — keep nurturing what's working in this relationship. |
| love | Express your appreciation openly; strong relationships thrive on communication. |
| anger | Consider taking a pause before responding; try to express your feelings calmly. |
| fear | Identify what specifically worries you and consider talking to someone you trust. |
| surprise | Give yourself time to process before reacting; seek clarification if needed. |

This design makes the system simple, transparent, and controllable.

---

## Project Structure

```
emotion_project/
├── data/
│   ├── emotions.csv          ← download from Kaggle
│   └── processed/            ← auto-generated after preprocessing
├── src/
│   ├── preprocess.py         ← data cleaning, vocab, splits
│   ├── dataset.py            ← PyTorch Dataset + DataLoaders
│   ├── model.py              ← BiGRUBaseline & BiGRUAttention
│   ├── train.py              ← training loop with early stopping
│   ├── evaluate.py           ← metrics, confusion matrix, comparison
│   └── demo.py               ← interactive demo
├── checkpoints/              ← saved model weights
├── results/                  ← plots and figures
└── requirements.txt
```

---

## Evaluation Metrics

- **Primary:** Macro-F1 (equal weight per class, robust to imbalance)
- **Secondary:** Overall accuracy, per-class F1
- **Visualisation:** Confusion matrix, training curves, attention heatmaps
