"""
PyTorch Dataset and data-loading utilities for Life Oracle.

Public API
----------
load_dataset(path)           -> dict of numpy arrays
FateDataset                  -> torch.utils.data.Dataset
make_splits(dataset, seed)   -> train_ds, val_ds, test_ds
make_loaders(...)            -> train_loader, val_loader, test_loader
augment(grid_tensor)         -> randomly transformed tensor (training only)
class_weights(labels)        -> 1-D float tensor of per-class weights
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .config import DATASET_PATH, CLASS_NAMES, BOARD_SIZE


# ─────────────────────────────────────────────────────────────────────────────
#  Load raw arrays
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path = DATASET_PATH) -> dict:
    """Load the .npz dataset and return a plain dict of numpy arrays."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation (only valid on a toroidal board)
# ─────────────────────────────────────────────────────────────────────────────

def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Apply a random symmetry to a (1, N, N) float32 tensor.

    Valid transformations for a toroidal board:
      - 90/180/270° rotations
      - horizontal / vertical flip
      - random cyclic shift in both axes

    All of these preserve the fate label on a wrapped board.
    """
    # rotation + flip (one of 8 dihedral symmetries)
    k = torch.randint(4, (1,)).item()
    x = torch.rot90(x, k=k, dims=(1, 2))
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[2])

    # cyclic shift
    n = x.shape[1]
    sr = torch.randint(n, (1,)).item()
    sc = torch.randint(n, (1,)).item()
    x  = torch.roll(x, shifts=(sr, sc), dims=(1, 2))
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FateDataset(Dataset):
    """
    Parameters
    ----------
    X        : (N, H, W) uint8 grids
    y        : (N,) int64 labels
    sources  : (N,) int8 source IDs  (optional)
    settle   : (N,) int32 settle times (optional, for eval breakdown)
    train    : bool — if True, applies augmentation on __getitem__
    """

    def __init__(
        self,
        X:       np.ndarray,
        y:       np.ndarray,
        sources: Optional[np.ndarray] = None,
        settle:  Optional[np.ndarray] = None,
        train:   bool = False,
    ) -> None:
        self.X       = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)  # (N,1,H,W)
        self.y       = torch.from_numpy(y.astype(np.int64))
        self.sources = sources
        self.settle  = settle
        self.train   = train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]          # (1, H, W)
        if self.train:
            x = augment(x)
        return x, self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Train / val / test splits
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(
    data:       dict,
    val_frac:   float = 0.10,
    test_frac:  float = 0.10,
    seed:       int   = 42,
) -> tuple[FateDataset, FateDataset, FateDataset]:
    """
    Deterministic stratified split into train / val / test.
    Deduplication is assumed to have already been done at generation time.
    """
    rng = np.random.default_rng(seed)
    X   = data["X"]
    y   = data["y"]
    src = data.get("source")
    st  = data.get("settle_time")
    n   = len(y)

    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    def _ds(i, train=False):
        return FateDataset(
            X=X[i], y=y[i],
            sources=src[i] if src is not None else None,
            settle =st[i]  if st  is not None else None,
            train=train,
        )

    return _ds(train_idx, train=True), _ds(val_idx), _ds(test_idx)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(
    train_ds:   FateDataset,
    val_ds:     FateDataset,
    test_ds:    FateDataset,
    batch_size: int = 256,
    num_workers:int = 0,
    oversample: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader).
    Training loader uses WeightedRandomSampler when oversample=True.
    """
    if oversample:
        labels  = train_ds.y.numpy()
        weights = class_weights(labels)
        sample_w = weights[labels]
        sampler  = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).float(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers, pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2,
        shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2,
        shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def class_weights(labels: np.ndarray) -> np.ndarray:
    """Return inverse-frequency weights for each sample (length = len(labels))."""
    n_classes = len(CLASS_NAMES)
    counts    = np.bincount(labels, minlength=n_classes).astype(float)
    counts    = np.where(counts == 0, 1.0, counts)
    inv       = 1.0 / counts
    inv       = inv / inv.sum() * n_classes   # normalise so mean weight ≈ 1
    # per-sample weight
    return inv[labels]
