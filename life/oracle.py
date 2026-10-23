"""
Fate oracle for Conway's Game of Life.

`run_until_settled(grid, max_generations)` simulates the pattern and returns
a `FateResult` with the fate class, period, settle time and final population.

Fate classes (from config.py):
    EXTINCT     – population reaches 0
    STABLE      – period-1 cycle (still life)
    OSCILLATING – period ≥ 2 repeating cycle
    UNSETTLED   – no repeat found within max_generations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import (
    CLASS_NAMES, EXTINCT, STABLE, OSCILLATING, UNSETTLED, MAX_GENERATIONS,
)
from .engine import step, Grid


@dataclass
class FateResult:
    label: int               # EXTINCT / STABLE / OSCILLATING / UNSETTLED
    period: Optional[int]    # cycle length; None for UNSETTLED/EXTINCT
    settle_time: Optional[int]  # generation of first repeated state; None for UNSETTLED/EXTINCT
    final_population: int    # live cells in the final state
    generations_run: int     # how many generations were simulated

    @property
    def label_name(self) -> str:
        return CLASS_NAMES[self.label]

    def __str__(self) -> str:
        parts = [f"label={self.label_name}"]
        if self.period is not None:
            parts.append(f"period={self.period}")
        if self.settle_time is not None:
            parts.append(f"settle_time={self.settle_time}")
        parts.append(f"pop={self.final_population}")
        parts.append(f"gens={self.generations_run}")
        return f"FateResult({', '.join(parts)})"


def _grid_key(grid: Grid) -> bytes:
    """Stable bytes key for a grid (used as dict key for cycle detection)."""
    return grid.tobytes()


def run_until_settled(
    grid: Grid,
    max_generations: int = MAX_GENERATIONS,
) -> FateResult:
    """
    Simulate `grid` until a cycle is detected or `max_generations` is reached.

    Returns a FateResult describing the fate.
    """
    seen: dict[bytes, int] = {}  # key -> generation index it was first seen

    current = grid.astype(np.uint8)

    for gen in range(max_generations + 1):
        pop = int(current.sum())

        # ── Extinction ────────────────────────────────────────────────────
        if pop == 0:
            return FateResult(
                label=EXTINCT,
                period=None,
                settle_time=None,
                final_population=0,
                generations_run=gen,
            )

        key = _grid_key(current)

        # ── Cycle detected ────────────────────────────────────────────────
        if key in seen:
            first_gen = seen[key]
            period = gen - first_gen
            label = STABLE if period == 1 else OSCILLATING
            return FateResult(
                label=label,
                period=period,
                settle_time=first_gen,
                final_population=pop,
                generations_run=gen,
            )

        seen[key] = gen

        # ── Advance one generation ────────────────────────────────────────
        if gen < max_generations:
            current = step(current)

    # ── Generation limit reached ──────────────────────────────────────────
    pop = int(current.sum())
    return FateResult(
        label=UNSETTLED,
        period=None,
        settle_time=None,
        final_population=pop,
        generations_run=max_generations,
    )
