"""
Evaluation script for FateNet.

Usage
-----
    python -m life.evaluate               # test set metrics + baselines
    python -m life.evaluate --plots       # also save confusion matrix PNG
    python -m life.evaluate --help

Reports
-------
  • Overall accuracy, per-class precision / recall / F1
  • Confusion matrix (text + optional PNG)
  • Breakdown by source generator
  • Breakdown by settle-time bucket
  • Calibration: confidence vs accuracy
  • Baselines: majority-class, hand-crafted features (LogReg), FC-net, CNN
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    BOARD_SIZE, CLASS_NAMES,
    DATASET_PATH, MODEL_PATH,
)
from .dataset import load_dataset, make_splits, make_loaders, FateDataset
from .model import FateNet
from .generate_data import SOURCE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def _per_class_metrics(preds: np.ndarray, labels: np.ndarray, n_cls: int):
    """Return (precision, recall, f1) arrays of shape (n_cls,)."""
    prec = np.zeros(n_cls)
    rec  = np.zeros(n_cls)
    f1   = np.zeros(n_cls)
    for c in range(n_cls):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[c]  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c]   = (2 * prec[c] * rec[c] / (prec[c] + rec[c])
                   if (prec[c] + rec[c]) > 0 else 0.0)
    return prec, rec, f1


def _confusion_matrix(preds: np.ndarray, labels: np.ndarray, n_cls: int) -> np.ndarray:
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def _print_cm(cm: np.ndarray, names: list[str]) -> None:
    w = max(len(n) for n in names)
    header = " " * (w + 2) + "  ".join(f"{n:>{w}}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:>{w}}" for v in row)
        print(f"{names[i]:>{w}}  {cells}")


# ─────────────────────────────────────────────────────────────────────────────
#  Inference on a dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, probs, labels) numpy arrays."""
    all_probs, all_labels = [], []
    model.eval()
    for xb, yb in loader:
        logits = model(xb.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(yb.numpy())
    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds  = probs.argmax(axis=1)
    return preds, probs, labels


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────────────────────────────────────

def _calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> None:
    """Print reliability diagram as text."""
    confs  = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins   = np.linspace(0, 1, n_bins + 1)

    print("\nCalibration (confidence → actual accuracy):")
    print(f"  {'Bin':>14}   {'count':>6}   {'avg conf':>9}   {'accuracy':>9}")
    ece = 0.0
    n   = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        cnt  = mask.sum()
        if cnt == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc  = correct[mask].mean()
        ece     += cnt / n * abs(avg_conf - avg_acc)
        bar = "█" * int(avg_acc * 20)
        print(f"  [{lo:.1f}, {hi:.1f})   {cnt:>6}   {avg_conf:>9.3f}   {avg_acc:>9.3f}  {bar}")
    print(f"  ECE = {ece:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Baselines
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_majority(labels_train: np.ndarray, labels_test: np.ndarray) -> float:
    majority = np.bincount(labels_train).argmax()
    preds    = np.full_like(labels_test, majority)
    return _accuracy(preds, labels_test)


def _hand_features(X: np.ndarray) -> np.ndarray:
    """
    Hand-built features per board:
      0  population (live cells)
      1  density
      2  bounding box height
      3  bounding box width
      4  bounding box area
      5  number of connected components (4-connectivity, approximate)
    """
    n = len(X)
    feats = np.zeros((n, 6), dtype=np.float32)
    for i, g in enumerate(X):
        pop     = g.sum()
        density = pop / g.size
        rows    = np.any(g, axis=1)
        cols    = np.any(g, axis=0)
        rh      = rows.sum()
        cw      = cols.sum()
        feats[i] = [pop, density, rh, cw, rh * cw, 0]
        # crude connected-component count via simple flood fill isn't worth
        # the complexity; use a proxy: isolated 1-cells via erosion
        if pop > 0:
            from scipy.ndimage import label as sp_label
            _, nc = sp_label(g)
            feats[i, 5] = nc
    return feats


def _baseline_logreg(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
        feat_train = _hand_features(X_train)
        feat_test  = _hand_features(X_test)
        clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
        clf.fit(feat_train, y_train)
        preds = clf.predict(feat_test)
        return _accuracy(preds, y_test)
    except ImportError:
        return float("nan")


def _baseline_fc(
    train_ds: FateDataset,
    test_ds:  FateDataset,
    board_size: int,
    n_classes:  int,
    epochs:     int = 20,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Fully-connected baseline on the flattened grid."""
    d = board_size * board_size

    class FCNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(d, 256), nn.ReLU(),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, n_classes),
            )
        def forward(self, x):
            return self.net(x)

    model = FCNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    preds, _, labels = _predict_loader(model, DataLoader(test_ds, batch_size=512), device)
    return _accuracy(preds, labels)


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    data_path:  Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    board_size: int  = BOARD_SIZE,
    seed:       int  = 0,
    plots:      bool = False,
    run_baselines: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path} …")
    data = load_dataset(data_path)
    train_ds, val_ds, test_ds = make_splits(data, seed=seed)
    _, _, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=512, oversample=False)

    n_cls = len(CLASS_NAMES)

    # ── Load model ────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"No model at {model_path}. Run python -m life.train first.")
        return

    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = FateNet(board_size=board_size, num_classes=n_cls).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded model from {model_path}  (epoch {ckpt.get('epoch', '?')}, "
          f"val acc {ckpt.get('val_acc', 0):.2%})\n")

    preds, probs, labels = _predict_loader(model, test_loader, device)

    # ── Overall metrics ───────────────────────────────────────────────────
    acc = _accuracy(preds, labels)
    precision, recall, f1 = _per_class_metrics(preds, labels, n_cls)
    cm = _confusion_matrix(preds, labels, n_cls)

    print("=" * 60)
    print(f"Test accuracy: {acc:.4f}  ({acc:.2%})")
    print("=" * 60)
    print(f"\n{'Class':<14}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Count':>6}")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        cnt = (labels == i).sum()
        print(f"{name:<14}  {precision[i]:>6.3f}  {recall[i]:>6.3f}  "
              f"{f1[i]:>6.3f}  {cnt:>6}")
    macro_f1 = f1.mean()
    print(f"\nMacro F1: {macro_f1:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    _print_cm(cm, [n[:6] for n in CLASS_NAMES])

    # ── Breakdown by source ───────────────────────────────────────────────
    src_arr = data["source"][make_splits.__wrapped_idx if hasattr(make_splits, "__wrapped_idx") else :]
    # Re-derive test indices via same seed
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data["y"]))
    n_test = int(len(data["y"]) * 0.10)
    test_idx = idx[:n_test]
    test_src = data["source"][test_idx]
    test_st  = data.get("settle_time")
    test_st  = data["settle_time"][test_idx] if test_st is not None else None

    print("\nBreakdown by source:")
    print(f"  {'Source':<20}  {'Count':>6}  {'Acc':>7}")
    for s_idx, s_name in enumerate(SOURCE_NAMES):
        mask = test_src == s_idx
        if mask.sum() == 0:
            continue
        a = _accuracy(preds[mask], labels[mask])
        print(f"  {s_name:<20}  {mask.sum():>6}  {a:>7.2%}")

    # ── Breakdown by settle time ──────────────────────────────────────────
    if test_st is not None:
        print("\nBreakdown by settle-time bucket:")
        buckets = [(0, 10), (10, 50), (50, 200), (200, 1001)]
        print(f"  {'Settle gens':<16}  {'Count':>6}  {'Acc':>7}")
        for lo, hi in buckets:
            mask = (test_st >= lo) & (test_st < hi)
            if mask.sum() == 0:
                continue
            a = _accuracy(preds[mask], labels[mask])
            print(f"  [{lo:>4},{hi:>4})      {mask.sum():>6}  {a:>7.2%}")

    # ── Calibration ───────────────────────────────────────────────────────
    _calibration(probs, labels)

    # ── Baselines ─────────────────────────────────────────────────────────
    if run_baselines:
        print("\n" + "=" * 60)
        print("Baselines")
        print("=" * 60)
        X_all = data["X"]
        y_all = data["y"]
        rng2  = np.random.default_rng(seed)
        idx2  = rng2.permutation(len(y_all))
        n_t   = int(len(y_all) * 0.10)
        n_v   = int(len(y_all) * 0.10)
        test_idx2  = idx2[:n_t]
        train_idx2 = idx2[n_t + n_v:]

        X_train, y_train = X_all[train_idx2], y_all[train_idx2]
        X_test,  y_test  = X_all[test_idx2],  y_all[test_idx2]

        print(f"  {'Majority class':.<40} ", end="", flush=True)
        maj = _baseline_majority(y_train, y_test)
        print(f"{maj:.2%}")

        print(f"  {'Logistic Reg (hand features)':.<40} ", end="", flush=True)
        lr_acc = _baseline_logreg(X_train, y_train, X_test, y_test)
        print(f"{lr_acc:.2%}" if not np.isnan(lr_acc) else "n/a (sklearn not installed)")

        print(f"  {'FC network (flat grid)':.<40} ", end="", flush=True)
        fc_acc = _baseline_fc(train_ds, test_ds, board_size, n_cls, device=device)
        print(f"{fc_acc:.2%}")

        print(f"  {'FateNet (CNN)':.<40} {acc:.2%}")
        print(f"\n  CNN advantage over majority: +{(acc - maj)*100:+.1f}pp")

    # ── Optional plots ────────────────────────────────────────────────────
    if plots:
        _save_plots(cm, probs, labels, model_path.parent)


def _save_plots(cm, probs, labels, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#111")

        # Confusion matrix
        ax = axes[0]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels([n[:6] for n in CLASS_NAMES], rotation=30, color="white")
        ax.set_yticklabels([n[:6] for n in CLASS_NAMES], color="white")
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("True", color="white")
        ax.set_title("Confusion Matrix", color="white")
        ax.tick_params(colors="white")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] < cm.max() / 2 else "black", fontsize=9)
        fig.colorbar(im, ax=ax)

        # Calibration curve
        ax2 = axes[1]
        confs   = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(float)
        n_bins  = 15
        bins    = np.linspace(0, 1, n_bins + 1)
        avg_conf_list, avg_acc_list = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            avg_conf_list.append(confs[mask].mean())
            avg_acc_list.append(correct[mask].mean())
        ax2.plot([0, 1], [0, 1], "w--", linewidth=0.8, label="Perfect")
        ax2.plot(avg_conf_list, avg_acc_list, "o-", color="#88aaff", label="Model")
        ax2.set_xlabel("Confidence", color="white")
        ax2.set_ylabel("Accuracy", color="white")
        ax2.set_title("Calibration Curve", color="white")
        ax2.legend(facecolor="#222", labelcolor="white")
        ax2.set_facecolor("#1a1a1a")
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#444")

        fig.tight_layout()
        out = out_dir / "evaluation.png"
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\nPlots → {out}")
    except ImportError:
        print("matplotlib not available — skipping plots.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Evaluate FateNet on the test split")
    ap.add_argument("--data",          type=Path, default=DATASET_PATH)
    ap.add_argument("--model",         type=Path, default=MODEL_PATH)
    ap.add_argument("--size",          type=int,  default=BOARD_SIZE)
    ap.add_argument("--seed",          type=int,  default=0)
    ap.add_argument("--plots",         action="store_true", help="Save evaluation PNGs")
    ap.add_argument("--no-baselines",  action="store_true")
    args = ap.parse_args()

    evaluate(
        data_path     = args.data,
        model_path    = args.model,
        board_size    = args.size,
        seed          = args.seed,
        plots         = args.plots,
        run_baselines = not args.no_baselines,
    )


if __name__ == "__main__":
    _cli()
