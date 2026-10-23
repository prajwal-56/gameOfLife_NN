# Neural Network Fate Predictor

This document explains the mathematical motivation, theoretical limitations, and neural architecture design behind **FateNet**.

---

## 1. Theoretical Foundations: The Prediction Paradox

### Turing Completeness and Undecidability
Conway's Game of Life is proven to be **Turing complete**. It is capable of simulating logic gates, registers, memory banks, and universal Turing machines.

Because universal computation can be embedded within a Game of Life configuration, predicting whether an arbitrary initial pattern will ever settle, cycle, or reach a specific state is formally reducible to the **Halting Problem**:

$$\text{HALT}(M, w) \iff \text{Pattern Settles}(P_{M, w})$$

Consequently, **no algorithm can predict the fate of all arbitrary Game of Life patterns without simulating them step by step**.

### The Role of the Neural Network
The neural network does not solve the undecidable halting problem. Instead, it solves an **empirical statistical classification problem**:

$$\hat{y} = \arg\max_{c \in \{0, 1, 2, 3\}} P(Y = c \mid X = \text{grid})$$

For typical patterns—sparse structures, localized clusters, or small interactive perturbations—the system exhibits strong statistical regularities. FateNet acts as an intuitive heuristic that recognizes structural motifs (isolated blocks, blinker precursors, exploding pentominoes, decaying gliders) and predicts their probabilistic outcome instantly.

---

## 2. Why Convolutional Neural Networks?

The rules of the Game of Life possess two fundamental geometric properties:
1. **Locality**: A cell's next state depends strictly on its immediate $3 \times 3$ Moore neighborhood.
2. **Translation Invariance**: The transition function $f(c, N)$ is identical everywhere on the grid.

Convolutions explicitly encode both assumptions:
- A $3 \times 3$ convolution filter operates on the exact receptive radius of a cell's neighborhood.
- Weight sharing guarantees that a motif is recognized regardless of its spatial location $(r, c)$.
- Stacking $L$ convolutional layers expands the receptive field linearly, allowing information to propagate across the grid in the same manner that glider signals and debris fronts expand over generations.

---

## 3. Architecture Specification: `FateNet`

Standard convolutional networks use zero-padding, which distorts spatial relationships along board boundaries. Because our simulation operates on a **torus**, FateNet employs custom **circular padding** (`CircularPad2d`).

```text
Input Tensor: (B, 1, 32, 32)
   │
   ├─► CircularPad2d(1) -> Conv2D(1 -> 32, 3x3) -> BatchNorm -> ReLU
   │   (Receptive field: 3x3)
   │
   ├─► CircularPad2d(1) -> Conv2D(32 -> 64, 3x3) -> BatchNorm -> ReLU
   │   (Receptive field: 5x5)
   │
   ├─► CircularPad2d(1) -> Conv2D(64 -> 64, 3x3) -> BatchNorm -> ReLU
   │   (Receptive field: 7x7)
   │
   ├─► CircularPad2d(1) -> Conv2D(64 -> 128, 3x3) -> BatchNorm -> ReLU
   │   (Receptive field: 9x9)
   │
   ├─► Dual Global Pooling:
   │     ├─ Global Average Pooling: (B, 128) [Measures background activity]
   │     └─ Global Maximum Pooling: (B, 128) [Detects localized energetic structures]
   │     └─ Concatenate -> (B, 256)
   │
   ├─► Dense(256 -> 64) -> ReLU -> Dropout(p=0.3)
   │
   └─► Dense(64 -> 4) -> Logits [EXTINCT, STABLE, OSCILLATING, UNSETTLED]
```

---

## 4. Toroidal Invariance & Dual Global Pooling

### Translation Invariance on a Torus
Because the board wraps around seamlessly, shifting any pattern by $(\Delta r, \Delta c)$ results in an identical physical trajectory modulo translation.

By combining:
1. **Circular Convolutional Layers** (which are cyclic shift-equivariant)
2. **Global Spatial Pooling** (which collapses the $(H, W)$ dimensions into scalar statistics)

FateNet achieves **strict cyclic shift invariance**:
$$f(\text{roll}(X, \Delta r, \Delta c)) \equiv f(X)$$

### Dual Pooling Rationale
- **Average Pooling**: Measures the overall density of feature activations across the board (useful for distinguishing widespread chaotic growth from sparse still-lifes).
- **Max Pooling**: Fires if a single high-salience structural motif (such as an indestructible block or an active oscillator) exists anywhere on the board, even if the rest of the board is completely empty.

---

## 5. Symmetry Preservation & Data Augmentation

On a toroidal board, any starting grid $X$ has an equivalence class of symmetric configurations that share the **exact same fate**:
- 4 rotations: $0^\circ, 90^\circ, 180^\circ, 270^\circ$
- 2 reflections: horizontal / vertical flips
- $N \times N$ cyclic translations: $\text{roll}(X, \Delta r, \Delta c)$

During training, each batch undergoes stochastic on-the-fly dihedral transformations and random coordinate rolls, enabling the network to learn robust, rotation-invariant representations without requiring excessive model parameter capacity.
