"""
Tests for life.model and life.dataset (shape sanity + overfit check)
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from life.model import FateNet
from life.config import BOARD_SIZE, CLASS_NAMES


def _dummy_batch(n: int = 4, size: int = BOARD_SIZE):
    x = torch.zeros(n, 1, size, size)
    y = torch.randint(0, len(CLASS_NAMES), (n,))
    return x, y


def test_output_shape():
    model = FateNet(board_size=BOARD_SIZE, num_classes=len(CLASS_NAMES))
    x, _ = _dummy_batch(8)
    out = model(x)
    assert out.shape == (8, len(CLASS_NAMES))


def test_probabilities_sum_to_one():
    model = FateNet(board_size=BOARD_SIZE, num_classes=len(CLASS_NAMES))
    x, _ = _dummy_batch(16)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(16), atol=1e-5)


def test_overfit_tiny_batch():
    """
    The model should be able to memorise a tiny batch (basic training sanity).
    """
    import torch.nn as nn

    torch.manual_seed(0)
    model = FateNet(board_size=BOARD_SIZE, num_classes=len(CLASS_NAMES))
    opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit  = nn.CrossEntropyLoss()

    x = torch.rand(8, 1, BOARD_SIZE, BOARD_SIZE)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    model.train()
    for _ in range(200):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    # Should perfectly memorise 8 samples
    assert (preds == y).all(), f"Failed to overfit tiny batch: preds={preds.tolist()}"


def test_circular_padding_equivariance():
    """
    A cyclic shift of the input should not dramatically change predictions
    (rough check — the model should be approximately shift-equivariant
    because circular padding + global pooling makes it fully shift-invariant).
    """
    torch.manual_seed(1)
    model = FateNet(board_size=BOARD_SIZE, num_classes=len(CLASS_NAMES))
    model.eval()

    x = torch.rand(1, 1, BOARD_SIZE, BOARD_SIZE)
    x_shifted = torch.roll(x, shifts=(5, 7), dims=(2, 3))

    with torch.no_grad():
        p1 = torch.softmax(model(x), dim=1)
        p2 = torch.softmax(model(x_shifted), dim=1)

    # With global pooling the outputs should be identical
    assert torch.allclose(p1, p2, atol=1e-4), \
        f"Model is not shift-invariant: max diff {(p1-p2).abs().max():.4f}"
