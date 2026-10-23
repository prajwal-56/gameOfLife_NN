"""
Dataset generation for Life Oracle.

Usage
-----
    python -m life.generate_data          # default 30k samples
    python -m life.generate_data --n 5000 # quick pipeline check
    python -m life.generate_data --help
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

from .config import (
    BOARD_SIZE, MAX_GENERATIONS, CLASS_NAMES,
    DATA_DIR, DATASET_PATH,
)
from .engine import Grid, empty_grid
from .oracle import run_until_settled
from .patterns import LIBRARY, get_pattern, stamp, rotate_pattern


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical key (rotation+reflection invariant deduplication)
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_key(grid: Grid) -> bytes:
    variants = []
    g = grid
    for _ in range(4):
        variants.append(g.tobytes())
        variants.append(np.fliplr(g).tobytes())
        g = np.rot90(g)
    return min(variants)


# ─────────────────────────────────────────────────────────────────────────────
#  Sample record
# ─────────────────────────────────────────────────────────────────────────────

class Sample(NamedTuple):
    grid:             Grid
    label:            int
    period:           int
    settle_time:      int
    final_population: int
    source:           int


# ─────────────────────────────────────────────────────────────────────────────
#  Generators
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_NAMES = ["uniform_soup", "patch_soup", "library_perturb", "combo"]
_SOURCE_WEIGHTS = [0.35, 0.30, 0.20, 0.15]


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
    names  = list(LIBRARY.keys())
    name   = names[int(rng.integers(len(names)))]
    pat    = rotate_pattern(get_pattern(name), int(rng.integers(4)))
    ph, pw = pat.shape
    grid   = empty_grid(size)
    r0     = int(rng.integers(0, max(1, size - ph)))
    c0     = int(rng.integers(0, max(1, size - pw)))
    grid   = stamp(grid, pat, r0, c0)
    noise  = (rng.random((size, size)) < rng.uniform(0.02, 0.15)).astype(np.uint8)
    return np.abs(grid - noise).astype(np.uint8)


def _gen_combo(rng: np.random.Generator, size: int) -> Grid:
    names = list(LIBRARY.keys())
    grid  = empty_grid(size)
    for _ in range(int(rng.integers(2, 4))):
        pat    = rotate_pattern(get_pattern(names[int(rng.integers(len(names)))]),
                                int(rng.integers(4)))
        ph, pw = pat.shape
        r0     = int(rng.integers(0, max(1, size - ph)))
        c0     = int(rng.integers(0, max(1, size - pw)))
        grid   = stamp(grid, pat, r0, c0)
    return grid


_GENERATORS = [_gen_uniform_soup, _gen_patch_soup, _gen_library_perturb, _gen_combo]


def _generate_one(seed: int, size: int, max_gens: int) -> Sample:
    rng     = np.random.default_rng(seed)
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
#  Display helpers
# ─────────────────────────────────────────────────────────────────────────────

# ANSI colours (fall back gracefully in dumb terminals)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_COLORS = ["\033[91m", "\033[94m", "\033[93m", "\033[95m"]  # red,blue,amber,purple

def _class_bar(counts: Counter, total: int, width: int = 28) -> str:
    """Compact inline distribution bar for the tqdm postfix."""
    parts = []
    for i, name in enumerate(CLASS_NAMES):
        cnt = counts[i]
        pct = cnt / total * 100 if total else 0
        parts.append(f"{_COLORS[i]}{name[0]}{pct:4.0f}%{_RESET}")
    return "  ".join(parts)


def _print_distribution(counts: Counter, total: int) -> None:
    """Final pretty-printed class + source distribution table."""
    bar_width = 40
    print(f"\n{_BOLD}  Class distribution:{_RESET}")
    for i, name in enumerate(CLASS_NAMES):
        cnt  = counts[i]
        pct  = cnt / total * 100 if total else 0
        fill = int(pct / 100 * bar_width)
        bar  = _COLORS[i] + "█" * fill + _RESET + _DIM + "░" * (bar_width - fill) + _RESET
        print(f"  {name:<14} {bar}  {cnt:>6}  {pct:5.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
#  Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_samples:   int  = 30_000,
    board_size:  int  = BOARD_SIZE,
    max_gens:    int  = MAX_GENERATIONS,
    seed:        int  = 42,
    n_workers:   int  = 0,
    output_path: Path = DATASET_PATH,
) -> Path:
    workers = n_workers or mp.cpu_count()
    rng     = np.random.default_rng(seed)
    # generate 3× seeds to have slack after dedup loss
    seeds   = rng.integers(0, 2**31, size=n_samples * 3).tolist()

    print(f"\n{_BOLD}Life Oracle — dataset generation{_RESET}")
    print(f"  board {board_size}×{board_size}   target {n_samples:,}   workers {workers}")
    print()

    fn      = partial(_generate_one, size=board_size, max_gens=max_gens)
    seen:   set[bytes] = set()
    samples: list[Sample] = []
    counts:  Counter = Counter()
    dupes    = 0
    chunk    = min(512, n_samples)
    seed_idx = 0
    t0       = time.perf_counter()

    bar = tqdm(
        total=n_samples,
        unit="sample",
        bar_format=(
            "  {percentage:3.0f}%  {bar:35}  {n_fmt}/{total_fmt}"
            "  [{elapsed}<{remaining}  {rate_fmt}]"
        ),
        colour="white",
        dynamic_ncols=True,
    )

    with mp.Pool(processes=workers) as pool:
        while len(samples) < n_samples and seed_idx < len(seeds):
            batch = pool.map(fn, seeds[seed_idx:seed_idx + chunk])
            seed_idx += chunk

            for s in batch:
                key = _canonical_key(s.grid)
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                samples.append(s)
                counts[s.label] += 1

                bar.update(1)
                bar.set_postfix_str(_class_bar(counts, len(samples)), refresh=False)

                if len(samples) >= n_samples:
                    break

    bar.close()
    total   = len(samples)
    elapsed = time.perf_counter() - t0

    print(f"\n  {_BOLD}Done{_RESET}  —  {total:,} samples  |  "
          f"{dupes:,} dupes dropped  |  {elapsed:.1f}s  "
          f"({total/elapsed:.0f} samples/s)\n")

    _print_distribution(counts, total)

    # Source breakdown
    src_counts: Counter = Counter(s.source for s in samples)
    print(f"\n{_BOLD}  Source distribution:{_RESET}")
    for i, name in enumerate(SOURCE_NAMES):
        cnt = src_counts[i]
        pct = cnt / total * 100
        print(f"  {name:<20}  {cnt:>6}  {pct:5.1f}%")

    # Save
    X   = np.stack([s.grid             for s in samples]).astype(np.uint8)
    y   = np.array([s.label            for s in samples], dtype=np.int64)
    p   = np.array([s.period           for s in samples], dtype=np.int32)
    st  = np.array([s.settle_time      for s in samples], dtype=np.int32)
    fp  = np.array([s.final_population for s in samples], dtype=np.int32)
    src = np.array([s.source           for s in samples], dtype=np.int8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X, y=y, period=p, settle_time=st, final_population=fp, source=src,
        board_size=np.int32(board_size),
        class_names=np.array(CLASS_NAMES),
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n  saved → {output_path}  ({size_mb:.1f} MB)\n")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Generate Life Oracle training dataset")
    ap.add_argument("--n",       type=int,  default=30_000)
    ap.add_argument("--size",    type=int,  default=BOARD_SIZE)
    ap.add_argument("--maxgens", type=int,  default=MAX_GENERATIONS)
    ap.add_argument("--seed",    type=int,  default=42)
    ap.add_argument("--workers", type=int,  default=0)
    ap.add_argument("--out",     type=Path, default=DATASET_PATH)
    args = ap.parse_args()
    generate_dataset(
        n_samples=args.n, board_size=args.size, max_gens=args.maxgens,
        seed=args.seed, n_workers=args.workers, output_path=args.out,
    )


if __name__ == "__main__":
    _cli()
