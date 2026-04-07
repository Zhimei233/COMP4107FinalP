# Emotion Classifier — COMP 4107 Final Project

**Attention-Based Emotion Classification with Template-Based Decision Support**

Ziyi Jiang - 101266200
Zhimei Li - 101258414


## Overview

This project builds an emotion classification system for short English texts. Given a sentence, the system predicts one of six emotions, highlights the words that influenced the prediction using attention weights, and provides a short piece of structured judge-style guidance based on the result.

Two models are compared:

BiGRU + Masked Pooling (baseline)
BiGRU + Attention (proposed)

The attention model achieves similar classification performance to the baseline but adds interpretability — you can see which words the model focused on when making a prediction.
**6 emotion classes:** `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`

## Dataset

[We use the Kaggle Emotions dataset.](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data)

About 400k short English texts
Each text has one label (6 classes)

The data is slightly imbalanced.
For example, joy and sadness appear more often than love and surprise.

## Preprocessing

Before training, we do a few simple steps:

1. convert text to lowercase
2. remove URLs and mentions
3. clean special characters
4. tokenize text into words
5. pad or truncate to fixed length

## Models

1. BiGRU (Baseline)
- word embeddings
- bidirectional GRU
- mean + max pooling
- linear classifier

2. BiGRU + Attention
- same BiGRU encoder
- additive attention layer
- weighted sum of hidden states
- classifier

The attention layer helps the model focus on important words.

## How to Run

- Setup:
pip install -r requirements.txt

- Train both models:
python src/train.py --model baseline --epochs 30 --data data/emotions.csv
python src/train.py --model attention --epochs 30 --data data/emotions.csv

- Evaluate and compare:
python src/evaluate.py --compare --data data/emotions.csv

- Run the interactive demo:
python src/demo.py --model attention

- Then type a sentence like:
I feel really sad and alone
I am so frustrated with everything right now
I am scared of what will happen next

## Project Structure
src/
preprocess.py   — text cleaning, vocabulary, train/val/test split, guidance templates
dataset.py      — PyTorch Dataset and DataLoader
model.py        — BiGRUBaseline and BiGRUAttention model definitions
train.py        — training loop with early stopping
evaluate.py     — test set evaluation, confusion matrix, model comparison
predict.py      — single-text prediction logic
demo.py         — interactive command-line demo
utils.py        — shared utilities (metrics, plotting, checkpointing)

data/
emotions.csv    — Kaggle emotions dataset (not tracked in git)

outputs/            — model checkpoints, plots (not tracked in git)

