"""
Tests for life.engine

Covers:
- Single live cell dies (underpopulation)
- 2×2 block is a still life
- Blinker has period 2
- Glider returns to original shape displaced by (1,1) every 4 gens
- Fast and reference implementations agree on random boards
- Wraparound: a glider crossing an edge reappears on the other side
"""

import numpy as np
import pytest

from life.engine import step, step_reference, empty_grid, random_grid, count_neighbors


def make(rows: list[str]) -> np.ndarray:
    """Build a grid from a list of strings ('O'=alive, '.'=dead)."""
    grid = []
    width = max(len(r) for r in rows)
    for row in rows:
        grid.append([1 if c == "O" else 0 for c in row.ljust(width, ".")])
    return np.array(grid, dtype=np.uint8)


# ── Basic rules ───────────────────────────────────────────────────────────────

def test_single_cell_dies():
    """A lone live cell has no neighbours and should die."""
    grid = empty_grid(5)
    grid[2, 2] = 1
    result = step(grid)
    assert result.sum() == 0


def test_block_still_life():
    """A 2×2 block is a still life — it must not change."""
    grid = empty_grid(6)
    grid[2, 2] = grid[2, 3] = grid[3, 2] = grid[3, 3] = 1
    result = step(grid)
    assert np.array_equal(result, grid)


def test_blinker_period_2():
    """A horizontal blinker becomes vertical, then horizontal again."""
    grid = empty_grid(5)
    grid[2, 1] = grid[2, 2] = grid[2, 3] = 1   # horizontal

    gen1 = step(grid)
    # Expect vertical: (1,2), (2,2), (3,2)
    assert gen1[1, 2] == 1
    assert gen1[2, 2] == 1
    assert gen1[3, 2] == 1
    assert gen1.sum() == 3

    gen2 = step(gen1)
    assert np.array_equal(gen2, grid)


def test_glider_period_4():
    """
    A glider on a large enough board returns to its original shape
    displaced by (+1, +1) after 4 generations.
    """
    size = 20
    grid = empty_grid(size)
    # Standard glider at top-left
    #  .O.
    #  ..O
    #  OOO
    r0, c0 = 1, 1
    grid[r0 + 0, c0 + 1] = 1
    grid[r0 + 1, c0 + 2] = 1
    grid[r0 + 2, c0 + 0] = grid[r0 + 2, c0 + 1] = grid[r0 + 2, c0 + 2] = 1

    g = grid.copy()
    for _ in range(4):
        g = step(g)

    # The glider should have moved (+1, +1)
    expected = np.roll(np.roll(grid, 1, axis=0), 1, axis=1)
    assert np.array_equal(g, expected), "Glider did not displace (1,1) in 4 gens"


def test_fast_matches_reference_many_boards():
    """The vectorised step and the reference step must agree on random boards."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        size = rng.integers(4, 16)
        density = rng.uniform(0.1, 0.7)
        grid = random_grid(int(size), float(density), rng=rng)
        for _ in range(5):
            fast = step(grid)
            ref  = step_reference(grid)
            assert np.array_equal(fast, ref), "Fast and reference step disagree"
            grid = fast


def test_wraparound_glider():
    """A glider that crosses an edge should reappear on the other side."""
    size = 10
    grid = empty_grid(size)
    # Place glider near the bottom-right so it crosses both edges in 4 gens
    r0, c0 = size - 3, size - 2
    grid[(r0 + 0) % size, (c0 + 1) % size] = 1
    grid[(r0 + 1) % size, (c0 + 2) % size] = 1
    grid[(r0 + 2) % size, (c0 + 0) % size] = 1
    grid[(r0 + 2) % size, (c0 + 1) % size] = 1
    grid[(r0 + 2) % size, (c0 + 2) % size] = 1

    g = grid.copy()
    for _ in range(4):
        g = step(g)

    expected = np.roll(np.roll(grid, 1, axis=0), 1, axis=1)
    assert np.array_equal(g, expected), "Glider did not wrap correctly"


def test_neighbor_count_center():
    """Center cell surrounded by all 8 neighbours → count 8."""
    grid = np.ones((3, 3), dtype=np.uint8)
    n = count_neighbors(grid)
    # On a 3×3 torus every cell has all 8 neighbours alive.
    assert (n == 8).all()
