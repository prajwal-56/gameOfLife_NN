# System Architecture

Life Oracle combines a classical cellular automata simulation engine, an exact cycle-detection ground-truth oracle, and a convolutional neural network designed to predict the asymptotic fate of patterns.

---

## 1. High-Level System Architecture

```text
                     +----------------------------+
                     |         engine.py          |
                     |  Vectorized NumPy Rules    |
                     +--------------+-------------+
                                    |
             +----------------------+----------------------+
             |                                             |
+------------v------------+                   +------------v------------+
|        oracle.py        |                   |         app.py          |
| Exact Cycle Detection   |                   | Minimal Pygame Desktop  |
| State Hash Trajectories |                   | Interactive Simulator   |
+------------+------------+                   +------------+------------+
             |                                             ^
             | Ground Truth Labels                         | Inferred Probs
+------------v------------+                   +------------+------------+
|    generate_data.py     |                   |       predictor.py      |
| Parallel Multi-Source   |                   | Model Loading, Inference|
| Canonical Deduplication |                   | Debounced Background Run|
+------------+------------+                   +------------^------------+
             | fate_dataset.npz                            | fate_net.pt
+------------v------------+                   +------------+------------+
|        train.py         |------------------>|        model.py         |
| Toroidal Augmentation   | Checkpoint Export | FateNet CNN Architecture|
| Class-Balanced Sampling |                   | Circular Convolutions   |
+------------+------------+                   +-------------------------+
             |
+------------v------------+
|       evaluate.py       |
| ECE, Confusion Matrix,  |
| Baselines & Breakdowns  |
+-------------------------+
```

---

## 2. Component Directory & File Layout

```text
gameOfLife/
├── life/
│   ├── config.py         # Central system constants: board dimension, classes, paths, timing
│   ├── engine.py         # Pure array simulation engine (vectorized & reference implementations)
│   ├── oracle.py         # Exact trajectory analysis and fate classification (ground truth)
│   ├── patterns.py       # Built-in pattern catalog, rotation, toroidal stamping, .cells I/O
│   ├── app.py            # Minimalist monochrome Pygame GUI with zoom, resize, and live stats
│   ├── model.py          # FateNet CNN definition utilizing circular 2D padding
│   ├── predictor.py      # Low-latency CPU inference wrapper for real-time app integration
│   ├── generate_data.py  # Parallel multi-core synthetic dataset generation pipeline
│   ├── dataset.py        # PyTorch Dataset, dihedral augmentations, stratified splits
│   ├── train.py          # Training loop with lr scheduler, sparklines, and early stopping
│   └── evaluate.py       # Comprehensive test evaluation, baseline models, and calibration
├── docs/                 # In-depth architectural, theoretical, and mathematical guides
├── data/                 # Compressed dataset binaries (.npz)
├── models/               # Checkpoint weights (.pt) and training telemetry history (.json)
├── patterns/             # User-saved plaintext .cells pattern archives
└── tests/                # Pytest test suite covering unit behavior and pipeline integration
```

---

## 3. Separation of Concerns & Design Principles

1. **Pure Functional Core (`engine.py`)**:
   - The simulation rules know nothing about Pygame, PyTorch, files, or state. It accepts a `(N, N)` NumPy array of `uint8` and returns a new `(N, N)` NumPy array. This guarantees thread safety, straightforward vectorization, and deterministic reproducibility.

2. **Decoupled Verification (`oracle.py`)**:
   - Serves as the single source of truth for both offline dataset generation and the in-app `Verify` command. Because the rules and toroidal boundaries are identical across both contexts, the training data and live app truth are mathematically equivalent.

3. **Graceful ML Degradation (`app.py` & `predictor.py`)**:
   - The desktop simulator functions independently as a standalone Game of Life sandbox even when no trained neural weights exist. The predictor is dynamically attached via a clean fallback mechanism.

4. **Toroidal Geometry Invariance**:
   - Conway's Game of Life on a flat grid suffers from edge clipping artifacts. Life Oracle simulates an unwrapped 2-torus ($T^2$). This mathematical property is carried directly into the neural network architecture through custom circular convolution padding.
