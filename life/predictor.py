"""
Inference wrapper for FateNet.

Usage (from app.py)
-------------------
    predictor = Predictor()            # loads model at startup
    probs = predictor.predict(grid)    # np.ndarray shape (4,), sums to 1
    # or None if no model is available

If no checkpoint exists, the constructor raises FileNotFoundError;
the app catches this and shows "no model loaded" in the panel.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import MODEL_PATH, BOARD_SIZE, CLASS_NAMES
from .model import FateNet
from .engine import Grid


class Predictor:
    """Loads a saved FateNet checkpoint and exposes predict()."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No model checkpoint at {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        meta = checkpoint.get("meta", {})

        # Guard against board size mismatch
        ckpt_size = meta.get("board_size", BOARD_SIZE)
        if ckpt_size != BOARD_SIZE:
            raise ValueError(
                f"Model was trained on board_size={ckpt_size}, "
                f"but config has BOARD_SIZE={BOARD_SIZE}."
            )

        self.model = FateNet(board_size=BOARD_SIZE, num_classes=len(CLASS_NAMES))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self._device = torch.device("cpu")

    @torch.no_grad()
    def predict(self, grid: Grid) -> np.ndarray:
        """
        Given a (N, N) uint8 grid return a (4,) float32 probability array.
        """
        t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # 1x1xNxN
        logits = self.model(t)
        probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()
        return probs.astype(np.float32)
