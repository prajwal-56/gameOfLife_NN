"""
Training script for FateNet.

Usage
-----
    python -m life.train
    python -m life.train --epochs 50
    python -m life.train --help
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
from tqdm import tqdm

from .config import BOARD_SIZE, CLASS_NAMES, DATASET_PATH, MODEL_PATH
from .dataset import load_dataset, make_splits, make_loaders
from .model import FateNet


# ANSI helpers
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[92m"
_AMBER = "\033[93m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_FATE_COLORS = ["\033[91m", "\033[94m", "\033[93m", "\033[95m"]


def _color_acc(acc: float) -> str:
    c = _GREEN if acc >= 0.80 else (_AMBER if acc >= 0.60 else _RED)
    return f"{c}{acc:.2%}{_RESET}"


def _sparkline(history: list[float], width: int = 10) -> str:
    """Tiny unicode sparkline from a list of loss values."""
    blocks = " ▁▂▃▄▅▆▇█"
    if len(history) < 2:
        return " " * width
    vals = history[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1e-9
    return "".join(blocks[min(8, int((v - lo) / span * 8))] for v in vals)


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

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}Life Oracle — FateNet training{_RESET}")
    print(f"  device {device}   board {board_size}×{board_size}   "
          f"epochs {epochs}   batch {batch_size}   lr {lr}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"  loading dataset from {data_path} …")
    data = load_dataset(data_path)
    stored_size = int(data.get("board_size", board_size))
    if stored_size != board_size:
        raise ValueError(f"Dataset board_size={stored_size} ≠ {board_size}")

    train_ds, val_ds, test_ds = make_splits(data, seed=seed)
    train_loader, val_loader, _ = make_loaders(
        train_ds, val_ds, test_ds,
        batch_size=batch_size, oversample=oversample,
    )
    print(f"  split  train {len(train_ds):,}  val {len(val_ds):,}  test {len(test_ds):,}\n")

    # class distribution for reference
    label_counts = np.bincount(train_ds.y.numpy(), minlength=len(CLASS_NAMES))
    print(f"  {'class':<14} {'train samples':>14}")
    for i, name in enumerate(CLASS_NAMES):
        bar = _FATE_COLORS[i] + "█" * int(label_counts[i] / label_counts.max() * 20) + _RESET
        print(f"  {name:<14} {label_counts[i]:>6}  {bar}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────
    model = FateNet(board_size=board_size, num_classes=len(CLASS_NAMES)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  FateNet  {n_params:,} params")

    # ── Loss & optimiser ──────────────────────────────────────────────────
    if weighted_loss:
        counts = label_counts.astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        cw = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
        cw = cw / cw.sum() * len(CLASS_NAMES)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=5, min_lr=1e-5)

    # ── Column headers ────────────────────────────────────────────────────
    print()
    print(f"  {'ep':>4}  {'train loss':>10}  {'val loss':>9}  "
          f"{'val acc':>8}  {'lr':>8}  {'loss curve':>12}  ")
    print("  " + "─" * 62)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    best_val_acc      = 0.0
    epochs_no_improve = 0
    val_loss_history:  list[float] = []
    history:           list[dict]  = []

    model_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss   = 0.0
        n_train      = len(train_ds)
        batch_bar    = tqdm(
            train_loader,
            desc=f"  ep {epoch:>3}",
            leave=False,
            unit="batch",
            bar_format="  {desc}  {bar:28}  {n_fmt}/{total_fmt}  loss {postfix}",
            dynamic_ncols=True,
            colour="cyan",
        )
        running_loss = 0.0
        n_seen       = 0
        for xb, yb in batch_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n_seen       += len(yb)
            batch_bar.set_postfix_str(f"{running_loss/n_seen:.4f}", refresh=False)
        train_loss = running_loss / n_train

        # ── Val ────────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits      = model(xb)
                val_loss   += criterion(logits, yb).item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc   = val_correct / len(val_ds)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        val_loss_history.append(val_loss)

        improved = val_loss < best_val_loss
        marker   = f"{_GREEN}✓{_RESET}" if improved else " "

        if improved:
            best_val_loss     = val_loss
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "meta": {
                    "board_size":   board_size,
                    "class_names":  CLASS_NAMES,
                    "train_config": dict(epochs=epochs, batch_size=batch_size,
                                        lr=lr, weighted_loss=weighted_loss,
                                        oversample=oversample, seed=seed),
                },
            }, model_path)
        else:
            epochs_no_improve += 1

        spark = _sparkline(val_loss_history)
        print(f"  {epoch:>4}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
              f"{_color_acc(val_acc):>18}  {current_lr:>8.2e}  "
              f"{_DIM}{spark}{_RESET}  {marker}")

        history.append(dict(epoch=epoch, train_loss=train_loss,
                            val_loss=val_loss, val_acc=val_acc))

        if epochs_no_improve >= patience:
            print(f"\n  {_AMBER}early stop{_RESET} — no improvement for {patience} epochs")
            break

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"\n  {'─'*62}")
    print(f"  {_BOLD}best val loss{_RESET}  {best_val_loss:.4f}   "
          f"{_BOLD}best val acc{_RESET}  {_color_acc(best_val_acc)}")
    print(f"  elapsed  {elapsed:.0f}s   checkpoint → {model_path}\n")

    hist_path = model_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Train FateNet")
    ap.add_argument("--data",           type=Path,  default=DATASET_PATH)
    ap.add_argument("--model",          type=Path,  default=MODEL_PATH)
    ap.add_argument("--size",           type=int,   default=BOARD_SIZE)
    ap.add_argument("--epochs",         type=int,   default=60)
    ap.add_argument("--batch",          type=int,   default=256)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--patience",       type=int,   default=10)
    ap.add_argument("--seed",           type=int,   default=0)
    ap.add_argument("--no-weight-loss", action="store_true")
    ap.add_argument("--no-oversample",  action="store_true")
    args = ap.parse_args()
    train(
        data_path=args.data, model_path=args.model, board_size=args.size,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        patience=args.patience, seed=args.seed,
        weighted_loss=not args.no_weight_loss,
        oversample=not args.no_oversample,
    )


if __name__ == "__main__":
    _cli()
