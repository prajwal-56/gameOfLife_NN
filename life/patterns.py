"""
Built-in pattern library and plaintext (.cells) file I/O.

Pattern format (plaintext / .cells):
    Lines starting with '!' are comments and are ignored.
    '.' means dead, 'O' (capital O) means alive.
    Rows may be shorter than the grid width (trailing dead cells assumed).

Public API
----------
LIBRARY              dict[str, np.ndarray] of named patterns
get_pattern(name)    -> ndarray   (copy)
stamp(grid, pattern, row, col, rotation=0)  -> ndarray   (new grid)
load_cells(path)     -> ndarray
save_cells(path, pattern)
list_patterns()      -> list[str]
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from .engine import Grid


# ── Pattern definitions (plaintext rows) ─────────────────────────────────────

_RAW: dict[str, str] = {
    "block": """\
OO
OO
""",
    "beehive": """\
.OO.
O..O
.OO.
""",
    "loaf": """\
.OO.
O..O
.O.O
..O.
""",
    "blinker": """\
OOO
""",
    "toad": """\
.OOO
OOO.
""",
    "beacon": """\
OO..
OO..
..OO
..OO
""",
    "pulsar": """\
..OOO...OOO..
.............
O....O.O....O
O....O.O....O
O....O.O....O
..OOO...OOO..
.............
..OOO...OOO..
O....O.O....O
O....O.O....O
O....O.O....O
.............
..OOO...OOO..
""",
    "glider": """\
.O.
..O
OOO
""",
    "lwss": """\
O..O.
....O
O...O
.OOOO
""",
    "r_pentomino": """\
.OO
OO.
.O.
""",
    "acorn": """\
.O.....
...O...
OO..OOO
""",
    "diehard": """\
......O.
OO......
.O...OOO
""",
    "pentadecathlon": """\
OO.OOOO.OO
""",
    "gosper_glider_gun": """\
........................O...........
......................O.O...........
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO..............
OO........O...O.OO....O.O...........
..........O.....O.......O...........
...........O...O....................
............OO......................
""",
}


def _parse_rows(raw: str) -> Grid:
    """Convert a multi-line plaintext string to a uint8 numpy array."""
    lines = [l for l in raw.splitlines() if not l.startswith("!")]
    if not lines:
        return np.zeros((1, 1), dtype=np.uint8)
    width = max(len(l) for l in lines)
    rows = []
    for line in lines:
        padded = line.ljust(width, ".")
        rows.append([1 if ch == "O" else 0 for ch in padded])
    return np.array(rows, dtype=np.uint8)


# Build the library eagerly
LIBRARY: dict[str, Grid] = {name: _parse_rows(raw) for name, raw in _RAW.items()}

# Ordered list for keyboard shortcuts
PATTERN_ORDER: list[str] = list(LIBRARY.keys())


def get_pattern(name: str) -> Grid:
    """Return a copy of the named pattern."""
    return LIBRARY[name].copy()


def list_patterns() -> list[str]:
    """Return the ordered list of pattern names."""
    return list(PATTERN_ORDER)


def rotate_pattern(pattern: Grid, times: int = 1) -> Grid:
    """Rotate a pattern 90° clockwise `times` times."""
    return np.rot90(pattern, k=-times % 4)


def stamp(
    grid: Grid,
    pattern: Grid,
    row: int,
    col: int,
    rotation: int = 0,
) -> Grid:
    """
    Return a new grid with `pattern` placed at (row, col) (top-left corner).
    The placement wraps around the board edges.
    rotation: number of 90° clockwise turns.
    """
    pat = rotate_pattern(pattern, rotation)
    new_grid = grid.copy()
    n_rows, n_cols = new_grid.shape
    p_rows, p_cols = pat.shape
    for r in range(p_rows):
        for c in range(p_cols):
            new_grid[(row + r) % n_rows, (col + c) % n_cols] = pat[r, c]
    return new_grid


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_cells(path: str | Path) -> Grid:
    """Load a .cells / plaintext file and return a uint8 ndarray."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return _parse_rows(text)


def save_cells(path: str | Path, pattern: Grid) -> None:
    """Save an ndarray pattern to a .cells file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in pattern:
        lines.append("".join("O" if v else "." for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
