"""
Tests for life.oracle

Covers:
- Lone cell → EXTINCT
- 2×2 block → STABLE
- Blinker → OSCILLATING, period 2
- Glider on 32×32 wrapped board → OSCILLATING, period 128 (4 × 32)
- Tiny max_generations limit → UNSETTLED
"""

import numpy as np
import pytest

from life.engine import empty_grid
from life.oracle import run_until_settled
from life.config import EXTINCT, STABLE, OSCILLATING, UNSETTLED, BOARD_SIZE


# ── Helpers ───────────────────────────────────────────────────────────────────

def place(size: int, coords: list[tuple[int, int]]) -> np.ndarray:
    g = empty_grid(size)
    for r, c in coords:
        g[r % size, c % size] = 1
    return g


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_lone_cell_extinct():
    grid = place(10, [(5, 5)])
    result = run_until_settled(grid, max_generations=200)
    assert result.label == EXTINCT
    assert result.final_population == 0


def test_block_stable():
    grid = place(10, [(4, 4), (4, 5), (5, 4), (5, 5)])
    result = run_until_settled(grid, max_generations=200)
    assert result.label == STABLE
    assert result.period == 1
    assert result.final_population == 4


def test_blinker_oscillating():
    grid = place(10, [(5, 4), (5, 5), (5, 6)])
    result = run_until_settled(grid, max_generations=200)
    assert result.label == OSCILLATING
    assert result.period == 2


def test_glider_oscillating_period():
    """
    A glider on a 32×32 torus takes 4 × 32 = 128 steps to return to start.
    """
    size = BOARD_SIZE  # 32
    grid = empty_grid(size)
    r0, c0 = 1, 1
    grid[r0 + 0, c0 + 1] = 1
    grid[r0 + 1, c0 + 2] = 1
    grid[r0 + 2, c0 + 0] = grid[r0 + 2, c0 + 1] = grid[r0 + 2, c0 + 2] = 1

    result = run_until_settled(grid, max_generations=200)
    assert result.label == OSCILLATING
    assert result.period == 128, f"Expected period 128, got {result.period}"


def test_unsettled_with_tiny_limit():
    """With max_generations=1, a glider should be UNSETTLED."""
    size = BOARD_SIZE
    grid = empty_grid(size)
    grid[1, 2] = grid[2, 3] = grid[3, 1] = grid[3, 2] = grid[3, 3] = 1
    result = run_until_settled(grid, max_generations=1)
    assert result.label == UNSETTLED
    assert result.generations_run == 1
