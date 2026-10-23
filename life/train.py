"""
Training script for FateNet.

Usage
-----
    python -m life.train                    # default settings
    python -m life.train --epochs 50        # more epochs
    python -m life.train --data path/to.npz # custom dataset
    python -m life.train --help

The best-validation checkpoint is saved to models/fate_net.pt together
with metadata (board_size, class_names, config) so the app can verify
the model matches its board at load time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import (
    BOARD_SIZE, CLASS_NAMES,
    DATASET_PATH, MODEL_PATH,
)
from .dataset import load_dataset, make_splits, make_loaders, class_weights
from .model import FateNet


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_path:    Path  = DATASET_PATH,
    model_path:   Path  = MODEL_PATH,
    board_size:   int   = BOARD_SIZE,
    epochs:       int   = 60,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    patience:     int   = 10,
    seed:         int   = 0,
    weighted_loss:bool  = True,
    oversample:   bool  = True,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path} …")
    data = load_dataset(data_path)
    stored_size = int(data.get("board_size", board_size))
    if stored_size != board_size:
        raise ValueError(
            f"Dataset board_size={stored_size} != requested board_size={board_size}"
        )

    train_ds, val_ds, test_ds = make_splits(data, seed=seed)
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds,
        batch_size=batch_size,
        oversample=oversample,
    )
    print(f"Splits — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = FateNet(board_size=board_size, num_classes=len(CLASS_NAMES)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FateNet — {n_params:,} trainable parameters")

    # ── Loss ──────────────────────────────────────────────────────────────
    if weighted_loss:
        labels_train = train_ds.y.numpy()
        counts = np.bincount(labels_train, minlength=len(CLASS_NAMES)).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        cw     = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
        cw     = cw / cw.sum() * len(CLASS_NAMES)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=5, min_lr=1e-5)

    # ── Training ──────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    epochs_no_improve = 0
    history = []

    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>5}  {'Train loss':>10}  {'Val loss':>9}  "
          f"{'Val acc':>8}  {'LR':>8}  {'*':>2}")
    print("─" * 60)

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # ── Train phase ───────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_ds)

        # ── Val phase ─────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss    += criterion(logits, yb).item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc   = val_correct / len(val_ds)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val_loss
        marker   = "✓" if improved else ""

        if improved:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            epochs_no_improve = 0
            # Save checkpoint
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "meta": {
                    "board_size":   board_size,
                    "class_names":  CLASS_NAMES,
                    "train_config": {
                        "epochs":        epochs,
                        "batch_size":    batch_size,
                        "lr":            lr,
                        "weighted_loss": weighted_loss,
                        "oversample":    oversample,
                        "seed":          seed,
                    },
                },
            }, model_path)
        else:
            epochs_no_improve += 1

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
              f"{val_acc:>7.2%}  {current_lr:>8.2e}  {marker:>2}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    elapsed = time.perf_counter() - t_start
    print(f"\nTraining complete — {elapsed:.0f}s")
    print(f"Best val loss: {best_val_loss:.4f}   best val acc: {best_val_acc:.2%}")
    print(f"Checkpoint → {model_path}")

    # Save training history as JSON alongside the model
    hist_path = model_path.with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History    → {hist_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Train FateNet on the Life Oracle dataset")
    ap.add_argument("--data",     type=Path,  default=DATASET_PATH)
    ap.add_argument("--model",    type=Path,  default=MODEL_PATH)
    ap.add_argument("--size",     type=int,   default=BOARD_SIZE)
    ap.add_argument("--epochs",   type=int,   default=60)
    ap.add_argument("--batch",    type=int,   default=256)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--patience", type=int,   default=10)
    ap.add_argument("--seed",     type=int,   default=0)
    ap.add_argument("--no-weight-loss", action="store_true")
    ap.add_argument("--no-oversample",  action="store_true")
    args = ap.parse_args()

    train(
        data_path    = args.data,
        model_path   = args.model,
        board_size   = args.size,
        epochs       = args.epochs,
        batch_size   = args.batch,
        lr           = args.lr,
        patience     = args.patience,
        seed         = args.seed,
        weighted_loss= not args.no_weight_loss,
        oversample   = not args.no_oversample,
    )


if __name__ == "__main__":
    _cli()
