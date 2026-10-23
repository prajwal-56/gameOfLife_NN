"""
Simulation engine for Conway's Game of Life.

Public API
----------
step(grid)            -> ndarray   fast vectorised implementation
step_reference(grid)  -> ndarray   plain Python loop, obviously correct
count_neighbors(grid) -> ndarray   neighbour count array (used by both)

Representation
--------------
A grid is a 2-D NumPy array of shape (N, N), dtype uint8, values 0 or 1.
The board wraps (torus topology): np.roll gives this for free.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Grid = NDArray[np.uint8]


# ── Fast vectorised implementation (used by app and data generator) ──────────

def count_neighbors(grid: Grid) -> NDArray[np.int32]:
    """Return an (N, N) array of live-neighbour counts using np.roll."""
    n = np.zeros_like(grid, dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            n += np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
    return n


def step(grid: Grid) -> Grid:
    """
    Advance the Game of Life one generation.

    Rules (standard B3/S23):
    - A live cell with 2 or 3 neighbours survives.
    - A dead cell with exactly 3 neighbours becomes alive.
    - All others die / stay dead.
    """
    neighbors = count_neighbors(grid)
    alive_next = (neighbors == 3) | (grid.astype(bool) & (neighbors == 2))
    return alive_next.astype(np.uint8)


# ── Reference implementation (slow but obviously correct; used by tests) ─────

def step_reference(grid: Grid) -> Grid:
    """Plain Python loop implementation; used only for testing."""
    n_rows, n_cols = grid.shape
    new_grid = np.zeros_like(grid)
    for r in range(n_rows):
        for c in range(n_cols):
            live = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    live += grid[(r + dr) % n_rows, (c + dc) % n_cols]
            if grid[r, c]:
                new_grid[r, c] = 1 if live in (2, 3) else 0
            else:
                new_grid[r, c] = 1 if live == 3 else 0
    return new_grid


# ── Convenience ──────────────────────────────────────────────────────────────

def empty_grid(size: int) -> Grid:
    """Return a zeroed (size × size) grid."""
    return np.zeros((size, size), dtype=np.uint8)


def random_grid(size: int, density: float = 0.3, rng: np.random.Generator | None = None) -> Grid:
    """Return a random grid with roughly `density` fraction of live cells."""
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random((size, size)) < density).astype(np.uint8)
