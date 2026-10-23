"""
Dataset generation for Life Oracle.

Generates a labeled dataset of Game of Life starting states and their fates.

Usage
-----
    python -m life.generate_data          # default 30k samples → data/fate_dataset.npz
    python -m life.generate_data --n 5000 # quick pipeline check
    python -m life.generate_data --help

Sources
-------
  1. uniform_soup   – whole board random, density in [0.05, 0.6]
  2. patch_soup     – random live cells in a small centred/offset patch
  3. library_perturb – known patterns placed at random pos/rot + noise
  4. combo          – two or three small patterns placed near each other

All boards are labelled by oracle.run_until_settled.
Deduplication removes identical boards (and their rotations/reflections).
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .config import (
    BOARD_SIZE, MAX_GENERATIONS, CLASS_NAMES,
    DATA_DIR, DATASET_PATH,
)
from .engine import Grid, empty_grid
from .oracle import run_until_settled
from .patterns import LIBRARY, get_pattern, stamp, rotate_pattern


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical key (rotation+reflection invariant)
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_key(grid: Grid) -> bytes:
    """Return the lexicographically smallest bytes over all 8 symmetries."""
    variants = []
    g = grid
    for _ in range(4):
        variants.append(g.tobytes())
        variants.append(np.fliplr(g).tobytes())
        g = np.rot90(g)
    return min(variants)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-sample record
# ─────────────────────────────────────────────────────────────────────────────

class Sample(NamedTuple):
    grid:            Grid
    label:           int
    period:          int     # 0 if unknown
    settle_time:     int     # 0 if unknown
    final_population:int
    source:          int     # 0..3 matching generator index


# ─────────────────────────────────────────────────────────────────────────────
#  Generators
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_NAMES = ["uniform_soup", "patch_soup", "library_perturb", "combo"]


def _gen_uniform_soup(rng: np.random.Generator, size: int) -> Grid:
    density = rng.uniform(0.05, 0.60)
    return (rng.random((size, size)) < density).astype(np.uint8)


def _gen_patch_soup(rng: np.random.Generator, size: int) -> Grid:
    patch_size = int(rng.integers(3, min(size, 17)))
    density    = rng.uniform(0.2, 0.8)
    grid       = empty_grid(size)
    r0 = int(rng.integers(0, size - patch_size + 1))
    c0 = int(rng.integers(0, size - patch_size + 1))
    patch = (rng.random((patch_size, patch_size)) < density).astype(np.uint8)
    grid[r0:r0 + patch_size, c0:c0 + patch_size] = patch
    return grid


def _gen_library_perturb(rng: np.random.Generator, size: int) -> Grid:
    names   = list(LIBRARY.keys())
    name    = names[int(rng.integers(len(names)))]
    pat     = get_pattern(name)
    rot     = int(rng.integers(4))
    pat     = rotate_pattern(pat, rot)
    ph, pw  = pat.shape

    grid = empty_grid(size)
    r0   = int(rng.integers(0, max(1, size - ph)))
    c0   = int(rng.integers(0, max(1, size - pw)))
    grid = stamp(grid, pat, r0, c0)

    # Random noise: flip ~5-15% of cells
    noise_rate = rng.uniform(0.02, 0.15)
    noise_mask = (rng.random((size, size)) < noise_rate).astype(np.uint8)
    grid = np.abs(grid - noise_mask).astype(np.uint8)
    return grid


def _gen_combo(rng: np.random.Generator, size: int) -> Grid:
    names = list(LIBRARY.keys())
    n_pat = int(rng.integers(2, 4))
    grid  = empty_grid(size)
    for _ in range(n_pat):
        name = names[int(rng.integers(len(names)))]
        pat  = rotate_pattern(get_pattern(name), int(rng.integers(4)))
        ph, pw = pat.shape
        r0   = int(rng.integers(0, max(1, size - ph)))
        c0   = int(rng.integers(0, max(1, size - pw)))
        grid = stamp(grid, pat, r0, c0)
    return grid


_GENERATORS = [
    _gen_uniform_soup,
    _gen_patch_soup,
    _gen_library_perturb,
    _gen_combo,
]

# Sampling weights for the four sources
_SOURCE_WEIGHTS = [0.35, 0.30, 0.20, 0.15]


def _generate_one(seed: int, size: int, max_gens: int) -> Sample:
    """Generate and label one sample. Designed for multiprocessing.map."""
    rng = np.random.default_rng(seed)
    weights = np.array(_SOURCE_WEIGHTS)
    src_idx = int(rng.choice(len(_GENERATORS), p=weights / weights.sum()))
    grid    = _GENERATORS[src_idx](rng, size)
    result  = run_until_settled(grid, max_gens)
    return Sample(
        grid             = grid,
        label            = result.label,
        period           = result.period or 0,
        settle_time      = result.settle_time or 0,
        final_population = result.final_population,
        source           = src_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_samples:      int  = 30_000,
    board_size:     int  = BOARD_SIZE,
    max_gens:       int  = MAX_GENERATIONS,
    seed:           int  = 42,
    n_workers:      int  = 0,       # 0 = use all CPU cores
    output_path:    Path = DATASET_PATH,
) -> Path:
    """
    Generate `n_samples` labeled boards and save to `output_path` (.npz).
    Returns the output path.
    """
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_samples * 3).tolist()  # extra for dedup loss

    workers = n_workers or mp.cpu_count()
    print(f"Generating up to {n_samples} samples "
          f"(board={board_size}×{board_size}, workers={workers})…")

    t0 = time.perf_counter()

    fn = partial(_generate_one, size=board_size, max_gens=max_gens)

    seen: set[bytes] = set()
    samples: list[Sample] = []
    label_counts: Counter = Counter()

    chunk = 2048
    seed_idx = 0

    with mp.Pool(processes=workers) as pool:
        while len(samples) < n_samples and seed_idx < len(seeds):
            batch_seeds = seeds[seed_idx:seed_idx + chunk]
            seed_idx += chunk
            batch = pool.map(fn, batch_seeds)

            for s in batch:
                key = _canonical_key(s.grid)
                if key in seen:
                    continue
                seen.add(key)
                samples.append(s)
                label_counts[s.label] += 1
                if len(samples) >= n_samples:
                    break

    elapsed = time.perf_counter() - t0
    total   = len(samples)
    print(f"Done — {total} unique samples in {elapsed:.1f}s")

    # ── Class distribution ────────────────────────────────────────────────
    print("\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        cnt = label_counts[i]
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:<12}  {cnt:6d}  {pct:5.1f}%  {bar}")

    # ── Source distribution ───────────────────────────────────────────────
    src_counts: Counter = Counter(s.source for s in samples)
    print("\nSource distribution:")
    for i, name in enumerate(SOURCE_NAMES):
        cnt = src_counts[i]
        print(f"  {name:<20}  {cnt:6d}  {cnt/total*100:5.1f}%")

    # ── Save ─────────────────────────────────────────────────────────────
    X  = np.stack([s.grid for s in samples]).astype(np.uint8)
    y  = np.array([s.label            for s in samples], dtype=np.int64)
    p  = np.array([s.period           for s in samples], dtype=np.int32)
    st = np.array([s.settle_time      for s in samples], dtype=np.int32)
    fp = np.array([s.final_population for s in samples], dtype=np.int32)
    src= np.array([s.source           for s in samples], dtype=np.int8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X, y=y,
        period=p, settle_time=st, final_population=fp,
        source=src,
        board_size=np.int32(board_size),
        class_names=np.array(CLASS_NAMES),
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved → {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Generate Life Oracle training dataset")
    ap.add_argument("--n",       type=int,  default=30_000, help="Target samples (default 30k)")
    ap.add_argument("--size",    type=int,  default=BOARD_SIZE, help="Board size")
    ap.add_argument("--maxgens", type=int,  default=MAX_GENERATIONS)
    ap.add_argument("--seed",    type=int,  default=42)
    ap.add_argument("--workers", type=int,  default=0, help="CPU workers (0=all)")
    ap.add_argument("--out",     type=Path, default=DATASET_PATH)
    args = ap.parse_args()

    generate_dataset(
        n_samples   = args.n,
        board_size  = args.size,
        max_gens    = args.maxgens,
        seed        = args.seed,
        n_workers   = args.workers,
        output_path = args.out,
    )


if __name__ == "__main__":
    _cli()
