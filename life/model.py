"""
FateNet – small CNN for fate classification of Game of Life patterns.

Architecture
------------
Input: 1 × N × N binary grid (float32)

Conv 3×3 32ch  ReLU  (circular padding)
Conv 3×3 64ch  ReLU  (circular padding)
Conv 3×3 64ch  ReLU  (circular padding)
Conv 3×3 128ch ReLU  (circular padding)
Global avg-pool + global max-pool → concat → 256 features
Linear 256 → 64  ReLU  Dropout(0.3)
Linear 64  → num_classes  (logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPad2d(nn.Module):
    """Circular (toroidal) padding for 2-D convolutions."""

    def __init__(self, pad: int = 1) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad
        return F.pad(x, (p, p, p, p), mode="circular")


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            CircularPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class FateNet(nn.Module):
    """Convolutional fate predictor for a BOARD_SIZE × BOARD_SIZE binary grid."""

    def __init__(self, board_size: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.board_size  = board_size
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  64),
            ConvBlock(64, 128),
        )

        # Global pooling produces a 256-dim vector (avg 128 + max 128)
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, N, N)
        feat = self.conv(x)                    # (B, 128, N, N)
        avg  = feat.mean(dim=(2, 3))           # (B, 128)
        mx   = feat.amax(dim=(2, 3))           # (B, 128)
        pooled = torch.cat([avg, mx], dim=1)   # (B, 256)
        return self.head(pooled)               # (B, num_classes)
