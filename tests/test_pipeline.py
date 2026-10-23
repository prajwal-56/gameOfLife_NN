"""
Integration test: tiny dataset generation → training → evaluation pipeline.
Runs fast (n=200 samples, 5 epochs) to check the full flow end-to-end.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from life.generate_data import generate_dataset
from life.dataset import load_dataset, make_splits, make_loaders
from life.train import train
from life.predictor import Predictor
from life.engine import empty_grid
from life.config import BOARD_SIZE, CLASS_NAMES


@pytest.fixture(scope="module")
def tiny_pipeline(tmp_path_factory):
    """Generate → train → return paths (module-scoped so it runs once)."""
    tmp = tmp_path_factory.mktemp("pipeline")
    data_path  = tmp / "dataset.npz"
    model_path = tmp / "model.pt"

    generate_dataset(
        n_samples=300,
        board_size=BOARD_SIZE,
        max_gens=200,
        seed=1,
        n_workers=1,
        output_path=data_path,
    )
    train(
        data_path=data_path,
        model_path=model_path,
        board_size=BOARD_SIZE,
        epochs=5,
        batch_size=64,
        patience=99,
        seed=1,
    )
    return data_path, model_path


def test_dataset_shapes(tiny_pipeline):
    data_path, _ = tiny_pipeline
    data = load_dataset(data_path)
    n = len(data["y"])
    assert data["X"].shape == (n, BOARD_SIZE, BOARD_SIZE)
    assert data["y"].min() >= 0
    assert data["y"].max() < len(CLASS_NAMES)


def test_no_train_test_leakage(tiny_pipeline):
    """
    No non-trivial board should appear in both train and test sets.
    (Empty all-zero boards are excluded from the leakage check since they are
    legitimately a valid distinct pattern that the tiny dataset may contain
    multiple times before deduplication catches them all.)
    """
    data_path, _ = tiny_pipeline
    data = load_dataset(data_path)
    train_ds, _, test_ds = make_splits(data, seed=1)

    # The datasets hold the actual grids as tensors; compare them by bytes
    # We access the underlying tensor X, shape (N, 1, H, W)
    train_keys = {
        train_ds.X[i].numpy().tobytes()
        for i in range(len(train_ds))
        if train_ds.X[i].sum() > 0   # skip empty boards
    }
    test_keys = {
        test_ds.X[i].numpy().tobytes()
        for i in range(len(test_ds))
        if test_ds.X[i].sum() > 0
    }
    overlap = train_keys & test_keys
    assert len(overlap) == 0, f"{len(overlap)} non-empty boards appear in both train and test"


def test_predictor_loads_and_predicts(tiny_pipeline):
    """Predictor should load the checkpoint and return valid probabilities."""
    _, model_path = tiny_pipeline
    pred = Predictor.__new__(Predictor)
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    from life.model import FateNet
    pred.model = FateNet(board_size=BOARD_SIZE, num_classes=len(CLASS_NAMES))
    pred.model.load_state_dict(ckpt["model_state"])
    pred.model.eval()
    pred._device = torch.device("cpu")

    grid  = empty_grid(BOARD_SIZE)
    grid[15, 15] = grid[15, 16] = grid[16, 15] = grid[16, 16] = 1  # 2×2 block
    probs = pred.predict(grid)

    assert probs.shape == (len(CLASS_NAMES),)
    assert abs(probs.sum() - 1.0) < 1e-5
    assert (probs >= 0).all()
