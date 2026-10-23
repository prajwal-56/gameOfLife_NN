# Life Oracle

> An interactive desktop Conway's Game of Life simulator paired with a convolutional neural network that predicts the eventual fate of cellular patterns before simulation.

[![Tests](https://img.shields.io/badge/tests-27%20passed-brightgreen.svg)](#test-suite)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## Overview

**Life Oracle** is a self-contained Python desktop application that explores the boundary between deterministic cellular automata and deep learning. You paint initial cell configurations onto a toroidal grid, watch them evolve, or consult a neural network trained to classify the eventual fate of the pattern directly from its starting state.

### Key Capabilities
- **Minimalist Aesthetic GUI**: High-contrast, distraction-free monochrome canvas built with `pygame-ce`.
- **Dynamic Board Geometry**: Full window resizing, scroll-wheel zoom (4px to 40px/cell), and live board scaling (8×8 to 128×128).
- **Exact Fate Oracle**: Ground-truth engine using state-hash cycle detection to identify exact periods, settle times, and final populations.
- **Neural Fate Predictor**: Custom `FateNet` CNN with circular convolutions matching toroidal topology.
- **Pattern Library & Text I/O**: 14 classic Conway archetypes with rotation, wrapping stamps, and plaintext `.cells` save/load.
- **Live Ground-Truth Verification**: Instant `Verify` mode comparing predicted probability distributions against actual simulation trajectories.

---

## Deep-Dive Documentation

For thorough explanations of the mathematical principles, architecture, and training pipeline, see the specialized guides in [`docs/`](file:///home/prawmathean/playground/Projects/gameOfLife/docs/):

- 📐 [**System Architecture**](file:///home/prawmathean/playground/Projects/gameOfLife/docs/ARCHITECTURE.md): End-to-end dataflow, module responsibilities, and system design.
- 🔄 [**Simulation & Oracle Mechanics**](file:///home/prawmathean/playground/Projects/gameOfLife/docs/SIMULATION_AND_ORACLE.md): B3/S23 rules, 2-torus ($T^2$) topology, and exact cycle-detection algorithms.
- 🧠 [**Neural Predictor Theory**](file:///home/prawmathean/playground/Projects/gameOfLife/docs/NEURAL_PREDICTOR.md): Turing undecidability, circular convolutions, and translation invariance.
- 📊 [**Dataset Synthesis & Training**](file:///home/prawmathean/playground/Projects/gameOfLife/docs/DATASET_AND_TRAINING.md): Multi-source generation, canonical dihedral deduplication, and baseline benchmarks.

---

## Quickstart

### 1. Environment Setup

```bash
git clone <repo-url> && cd gameOfLife
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
python -m life.app
```

> **Note**: The application functions immediately as a standalone Game of Life sandbox. Once you train the model (below), the prediction panel activates automatically.

---

## The Neural Pipeline

### Step 1: Generate Training Data
Synthesizes a balanced, canonical-deduplicated dataset across four pattern generators (uniform soups, localized patches, library archetypes, and multi-pattern collisions):

```bash
python -m life.generate_data --n 30000
```

### Step 2: Train FateNet
Trains the CNN with circular convolutions, dynamic dihedral augmentations, and early stopping:

```bash
python -m life.train --epochs 60 --batch 256
```

### Step 3: Evaluate Metrics & Baselines
Computes confusion matrices, expected calibration error (ECE), and compares against majority, logistic regression, and dense MLP baselines:

```bash
python -m life.evaluate --plots
```

---

## Desktop Controls

| Key / Input | Action |
|---|---|
| `Left Click` / `Drag` | Toggle cell state (draw / erase) |
| `Right Click` / `Drag` | Erase cells |
| `Space + Drag` or `Middle Drag` | Pan camera across the infinite canvas |
| `Mouse Wheel` | Smooth zoom anchored to mouse cursor |
| `0` or `H` | Fit and center board in viewport |
| `Space` | Play / Pause simulation |
| `→` or `N` | Advance single generation |
| `↑` / `↓` | Double / halve simulation speed |
| `[` / `]` | Zoom in / out (stepwise) |
| `<` / `>` (`,` / `.`) | Decrease / increase board dimensions ($\pm 4$) |
| `C` | Clear grid to empty state |
| `R` | Populate grid with random soup |
| `G` | Toggle grid lines |
| `1` – `9` | Select stamp from library (Glider, Gosper Gun, Pulsar, etc.) |
| `Q` / `E` | Rotate active stamp $90^\circ$ CCW / CW |
| `P` | Toggle live background neural prediction |
| `V` | Run Fate Oracle to verify ground truth |
| `F` | Fast-forward board directly to settled state |
| `S` / `L` | Save / load pattern (`.cells` format) |
| `Esc` | Cancel active stamp / Quit application |

---

## Fate Classes

| Fate Class | Description | Detection Condition |
|---|---|---|
| `EXTINCT` | All cells perish | Total active population drops to 0 |
| `STABLE` | Still life | Grid state repeats with period 1 and population $> 0$ |
| `OSCILLATING` | Periodic oscillator | Grid state repeats with period $\ge 2$ |
| `UNSETTLED` | Unsettled transient | No repeat found within 1,000 generation limit |

---

## Test Suite

Run the full pytest suite (engine unit tests, cycle detection, shape equivariance, and end-to-end integration):

```bash
pytest
```
