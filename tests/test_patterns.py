"""
Tests for life.patterns (library, stamp, file I/O)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from life.patterns import (
    LIBRARY, get_pattern, stamp, rotate_pattern, load_cells, save_cells,
    list_patterns,
)
from life.oracle import run_until_settled
from life.config import STABLE, OSCILLATING, BOARD_SIZE
from life.engine import empty_grid


def test_library_not_empty():
    assert len(LIBRARY) > 0


def test_get_pattern_returns_copy():
    pat1 = get_pattern("block")
    pat2 = get_pattern("block")
    assert np.array_equal(pat1, pat2)
    pat1[0, 0] = 0
    assert pat2[0, 0] == 1, "get_pattern should return independent copies"


def test_block_is_stable():
    grid = empty_grid(BOARD_SIZE)
    pat  = get_pattern("block")
    grid = stamp(grid, pat, 14, 14)
    result = run_until_settled(grid, max_generations=200)
    assert result.label == STABLE


def test_blinker_is_oscillating():
    grid = empty_grid(BOARD_SIZE)
    pat  = get_pattern("blinker")
    grid = stamp(grid, pat, 15, 14)
    result = run_until_settled(grid, max_generations=200)
    assert result.label == OSCILLATING
    assert result.period == 2


def test_rotate_pattern():
    pat = get_pattern("blinker")   # 1×3 horizontal strip
    rot = rotate_pattern(pat, 1)   # should become 3×1 vertical
    assert rot.shape == (3, 1)


def test_stamp_wraps():
    """Placing a pattern partially outside the board should wrap."""
    grid = empty_grid(4)
    pat  = np.ones((2, 2), dtype=np.uint8)
    result = stamp(grid, pat, 3, 3)  # bottom-right, wraps
    # Expect live cells at (3,3), (3,0), (0,3), (0,0)
    assert result[3, 3] == 1
    assert result[3, 0] == 1
    assert result[0, 3] == 1
    assert result[0, 0] == 1


def test_save_and_load_roundtrip(tmp_path: Path):
    pat  = get_pattern("glider")
    path = tmp_path / "test.cells"
    save_cells(path, pat)
    loaded = load_cells(path)
    assert np.array_equal(pat, loaded)


def test_list_patterns_contains_classics():
    names = list_patterns()
    for name in ("block", "blinker", "glider"):
        assert name in names
