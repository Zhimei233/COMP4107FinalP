"""
dataset.py
PyTorch Dataset wrapper for the preprocessed emotion data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EmotionDataset(Dataset):
    def __init__(self, padded: np.ndarray, labels: np.ndarray):
        """
        Parameters
        ----------
        padded : np.ndarray, shape (N, max_len)  — token indices
        labels : np.ndarray, shape (N,)           — integer class labels
        """
        self.X = torch.tensor(padded, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(splits: dict, batch_size: int = 64, num_workers: int = 0):
    """
    Build train / val / test DataLoaders from the splits dict produced by preprocess.py.
    """
    loaders = {}
    for name, data in splits.items():
        dataset = EmotionDataset(data["padded"], data["labels"])
        shuffle = (name == "train")
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
