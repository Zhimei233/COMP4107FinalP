"""
model.py
Two models:
  1. BiGRUBaseline  — BiGRU + masked mean/max pooling + classifier
  2. BiGRUAttention — BiGRU + additive attention + classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Baseline: BiGRU + Masked Mean/Max Pooling 
class BiGRUBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.bigru = nn.GRU(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        feat_dim = hidden_size * 2
        pooled_dim = feat_dim * 2  # mean + max

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    # Forward pass with masked mean and max pooling
    def forward(self, x):
        """
        x : (batch, seq_len)
        Returns logits : (batch, num_classes)
        """
        pad_mask = (x == self.pad_idx)              # (B, L)  True = PAD

        emb = self.dropout(self.embedding(x))        # (B, L, E)
        out, _ = self.bigru(emb)                     # (B, L, 2H)

        # Masked mean pooling
        mask_expanded = (~pad_mask).unsqueeze(-1).float()   # (B, L, 1)
        sum_out = (out * mask_expanded).sum(dim=1)           # (B, 2H)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)      # (B, 1)
        mean_pool = sum_out / lengths                        # (B, 2H)

        # Masked max pooling
        out_masked = out.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
        max_pool, _ = out_masked.max(dim=1)                  # (B, 2H)

        pooled = torch.cat([mean_pool, max_pool], dim=1)     # (B, 4H)
        logits = self.classifier(self.dropout(pooled))
        return logits


# Proposed: BiGRU + Additive
class BiGRUAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.bigru = nn.GRU(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        feat_dim = hidden_size * 2
        self.attn_w = nn.Linear(feat_dim, feat_dim, bias=True)
        self.attn_v = nn.Linear(feat_dim, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _attend(self, hidden, pad_mask=None):
        """
        hidden    : (B, L, 2H)
        pad_mask  : (B, L) — True where position is a PAD token
        Returns
          context : (B, 2H)
          weights : (B, L)
        """
        energy = self.attn_v(torch.tanh(self.attn_w(hidden))).squeeze(-1)  # (B, L)
        if pad_mask is not None:
            energy = energy.masked_fill(pad_mask, float("-inf"))
        weights = F.softmax(energy, dim=1)                                  # (B, L)
        context = (weights.unsqueeze(-1) * hidden).sum(dim=1)              # (B, 2H)
        return context, weights

    def forward(self, x, return_attention=False):
        pad_mask = (x == 0)
        emb = self.dropout(self.embedding(x))
        hidden, _ = self.bigru(emb)
        context, weights = self._attend(hidden, pad_mask)
        logits = self.classifier(self.dropout(context))

        if return_attention:
            return logits, weights
        return logits


def build_model(model_type: str, vocab_size: int, **kwargs) -> nn.Module:
    if model_type == "baseline":
        return BiGRUBaseline(vocab_size=vocab_size, **kwargs)
    elif model_type == "attention":
        return BiGRUAttention(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")