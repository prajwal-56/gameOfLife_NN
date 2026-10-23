# Life Oracle: A Game of Life Simulator with a Neural Network Fate Predictor

*(Working title. Rename freely.)*

---

## 1. Overview

Life Oracle is a native desktop application for Conway's Game of Life. You draw a pattern on a grid, press play, and watch it evolve. A built-in neural network looks at the pattern you drew and predicts its eventual fate before the simulation plays out. A ground-truth checker can then run the pattern to the end and show whether the network was right.

The project has three connected parts:

1. **A simulation engine** that implements the Game of Life rules.
2. **A fate oracle** that runs a pattern until its behavior is known and labels it. This generates the training data and also serves as the ground truth inside the app.
3. **A neural network** (a small convolutional network) trained on those labeled patterns to predict the fate directly from the starting grid, without running the simulation.

Everything runs locally in a single Python process. There is no server, no network calls and no database.

---

## 2. What the finished product does

- Shows a grid you can paint on with the mouse.
- Plays, pauses, single-steps and speed-controls the simulation.
- Provides a library of classic patterns (glider, blinker, block, R-pentomino, acorn, and others) you can stamp onto the board.
- Saves and loads patterns as plain text files.
- Shows a live prediction panel: the probability the pattern will die out, settle into a still life, become an oscillator, or keep evolving beyond the checking limit.
- Has a "Verify" action that runs the real simulation to its conclusion and compares the result with the prediction.
- Keeps a session scoreboard of how often the network has been right.
- Works as a plain Game of Life app even if no trained model is present.

---

## 3. Background

### 3.1 The rules

The world is a grid of cells, each alive or dead. At each generation, every cell looks at its 8 neighbors (horizontal, vertical, diagonal) and all cells update simultaneously:

1. A live cell with fewer than 2 live neighbors dies (underpopulation).
2. A live cell with 2 or 3 live neighbors survives.
3. A live cell with more than 3 live neighbors dies (overpopulation).
4. A dead cell with exactly 3 live neighbors becomes alive (reproduction).

### 3.2 Board topology

The board is a fixed-size square (default 32 x 32) that **wraps around** at the edges (a torus): the top edge connects to the bottom, the left edge to the right. Reasons:

- Every cell has exactly 8 neighbors, so no edge special-casing.
- Patterns like gliders never "fall off" the world.
- The board has a finite number of possible states, so every pattern is guaranteed to eventually repeat a state. This makes exact cycle detection possible.

The network is trained for one specific board size. The app board and the training data must use the same size and the same wrapping rule.

### 3.3 Fate classes

Because the board is finite and wraps, every pattern eventually enters a cycle. Patterns are classified by that cycle:

| Class | Meaning | How it is detected |
|---|---|---|
| `EXTINCT` | All cells die | Population reaches 0 |
| `STABLE` | Settles into a still life (period 1) | A state repeats with period 1 and population > 0 |
| `OSCILLATING` | Settles into a repeating cycle of period 2 or more | A state repeats with period >= 2 |
| `UNSETTLED` | No repeat found within the generation limit | Limit reached without a repeat |

In plain terms: "dies out" = `EXTINCT`, "stabilizes" = `STABLE`, and "survives" is split into `OSCILLATING` (alive and periodic) and `UNSETTLED` (alive, still evolving after the limit, often chaotic or with long-travelling gliders).

Extra values recorded for each pattern: `settle_time` (the generation at which it first entered its final cycle), `period`, and `final_population`.

---

## 4. Architecture

```text
                +---------------------+
                |     engine.py       |   step(grid) -> grid
                |  (rules, NumPy)     |
                +----------+----------+
                           |
            +--------------+---------------+
            |                              |
   +--------v---------+          +---------v---------+
   |    oracle.py     |          |      app.py       |
   | run_until_settled|<---------|  pygame desktop   |
   | -> label, period |  Verify  |  UI + simulation  |
   +--------+---------+          +---------+---------+
            |                              ^
            | labels                       | probabilities
   +--------v---------+          +---------+---------+
   |  generate_data   |          |   predictor.py    |
   |  (random soups,  |          |  loads model,     |
   |   stamps, etc.)  |          |  grid -> probs    |
   +--------+---------+          +---------^---------+
            | dataset.npz                  | weights (.pt)
   +--------v---------+          +---------+---------+
   |    train.py      +--------->|     model.py      |
   |  train/evaluate  |          |  CNN definition   |
   +------------------+          +-------------------+
```

### Tech stack

- **Python 3.11+**
- **NumPy**: grid representation and fast vectorized simulation
- **pygame** (or pygame-ce): window, drawing, mouse and keyboard input
- **PyTorch**: model definition, training, inference (CPU is sufficient)
- **scikit-learn** (optional): baseline models and evaluation metrics
- **matplotlib** (optional): training curves and confusion matrix plots

### Project structure

```text
life-oracle/
├── life/
│   ├── config.py          # board size, generation limit, class names, paths
│   ├── engine.py          # step(), count_neighbors(), reference implementation
│   ├── oracle.py          # run_until_settled() -> FateResult
│   ├── patterns.py        # named pattern library + text pattern loader/saver
│   ├── generate_data.py   # builds the labeled dataset
│   ├── dataset.py         # loading, splitting, augmentation, PyTorch Dataset
│   ├── model.py           # FateNet (CNN)
│   ├── train.py           # training loop, checkpointing
│   ├── evaluate.py        # metrics, confusion matrix, baselines
│   ├── predictor.py       # inference wrapper used by the app
│   └── app.py             # pygame application (entry point)
├── data/                  # generated datasets (not committed)
├── models/                # trained weights (.pt)
├── patterns/              # saved patterns (.cells text files)
├── tests/
├── requirements.txt
└── README.md
```

### Commands

```text
python -m life.app             # run the desktop app
python -m life.generate_data   # create the labeled dataset
python -m life.train           # train the model
python -m life.evaluate        # evaluate model + baselines
pytest                         # run tests
```

---

## 5. Component 1: Simulation engine (`engine.py`)

**Representation:** a 2D NumPy array of shape `(N, N)`, dtype `uint8`, values 0 (dead) or 1 (alive).

**Core function:** `step(grid) -> grid` computes the next generation.

Two implementations are kept:

1. **Reference implementation:** plain Python loops over cells with explicit neighbor counting and modulo wraparound. Slow but obviously correct.
2. **Fast implementation:** vectorized NumPy. The neighbor count for every cell at once is the sum of the grid shifted in 8 directions (using `np.roll`, which gives wraparound for free). The rules are then applied as boolean array operations:
   - `alive_next = (neighbors == 3) | (grid & (neighbors == 2))`

The fast version is what the app and the data generator use. Tests verify that both implementations agree on many random boards over many generations.

Design rule: the engine is pure functions on arrays. It knows nothing about pygame, PyTorch or files.

---

## 6. Component 2: Fate oracle (`oracle.py`)

`run_until_settled(grid, max_generations)` simulates a pattern and returns a result containing: `label`, `period`, `settle_time`, `final_population`.

**Algorithm:**

1. Maintain a dictionary mapping `hash(grid bytes) -> generation index`.
2. At each generation:
   - If population is 0, return `EXTINCT`.
   - If the current state's hash is already in the dictionary, a cycle exists. `period = current_gen - first_seen_gen` and `settle_time = first_seen_gen`. Return `STABLE` if period is 1, otherwise `OSCILLATING`.
   - Otherwise record the state and call `step`.
3. If `max_generations` is reached, return `UNSETTLED`.

Because the board is finite and wraps, the cycle detection is exact (no heuristics). Hash collisions are avoided by storing the grid bytes as the key (or a strong hash).

The default generation limit is around 1000, which is a configuration value. The `UNSETTLED` label always means "not settled within this limit", never "never settles".

The oracle has two jobs: it labels training data, and in the app it provides the ground truth behind the "Verify" button.

---

## 7. Component 3: Desktop app (`app.py`)

A single pygame window, roughly 1000 x 700 pixels. The board is on the left (32 x 32 cells at about 20 px each) and a side panel is on the right.

### 7.1 Main loop

Each frame:

1. Handle events (mouse, keyboard, window close).
2. If running and enough time has passed for the chosen speed, advance one generation with `engine.step`.
3. Redraw the board and the side panel.
4. Cap the frame rate (60 FPS).

Simulation speed (generations per second) is decoupled from frame rate, so the UI stays responsive at any speed.

### 7.2 Controls

| Input | Action |
|---|---|
| Left click / drag | Paint live cells |
| Right click / drag | Erase cells |
| Space | Play / pause |
| Right arrow or N | Advance one generation |
| Up / Down arrows | Increase / decrease speed |
| C | Clear board |
| R | Fill with a random soup |
| G | Toggle grid lines |
| 1-9 | Select a pattern from the library to stamp |
| Q / E | Rotate the selected stamp |
| Click while a stamp is selected | Place the stamp at the cursor |
| P | Toggle live prediction |
| V | Verify: run the oracle on the current board |
| F | Fast-forward: jump to the settled state |
| S / L | Save / load a pattern file |
| Esc | Cancel stamp, then quit |

Editing the board while the simulation is running is allowed.

### 7.3 Side panel

- Generation counter and live population.
- Run state and speed.
- **Prediction block:** four probability bars (Extinct, Stable, Oscillating, Unsettled), with the top class highlighted.
- **Reality block** (after Verify): the actual fate, period, and the generation at which it settled, with a clear match / mismatch indicator.
- **Scoreboard:** predictions checked this session and how many were correct.
- A short controls reminder.

### 7.4 Pattern files

Patterns are saved as plain text, one row per line, `.` for dead and `O` for alive (the "plaintext" Life convention), so they are easy to read and edit by hand. The built-in library uses the same format.

---

## 8. Component 4: Dataset generation (`generate_data.py`)

The network is only as good as its training data, and the data must resemble what a person will actually draw.

### 8.1 Pattern sources

A mixture of generators:

1. **Uniform random soups:** the whole board filled randomly, with the density (fraction of live cells) itself sampled randomly per sample, for example between 0.05 and 0.6.
2. **Sparse patch soups:** random cells only inside a small centered or randomly placed patch (for example 3 x 3 up to 16 x 16). These resemble small hand-drawn patterns and often die or settle quickly.
3. **Library patterns with perturbations:** known patterns (gliders, spaceships, methuselahs, and so on) placed at random positions and orientations, with random extra cells or removed cells added.
4. **Combinations:** two or three small patterns placed near each other, so they interact.

The mix is configurable. Sources 2-4 matter because a model trained only on dense random soups will fail on the sparse, structured patterns people actually draw.

### 8.2 Labeling

Each generated board is run through `oracle.run_until_settled`. The stored record is:

- `X`: the starting grid (`uint8`, shape `(N, N)`)
- `y`: the class index
- `period`, `settle_time`, `final_population`: extra targets, kept for analysis and optional extensions
- `source`: which generator produced it (used for per-source evaluation)

### 8.3 Practicalities

- Generation uses `multiprocessing` so all CPU cores label samples in parallel.
- Target size: tens of thousands to roughly 100k samples. Start small (5k) to check the pipeline end to end.
- Output: a compressed `.npz` file in `data/`. Boards may be bit-packed to save space.
- **Deduplication:** identical boards (including rotated and mirrored versions) are removed so the same pattern does not appear in both training and test data.
- **Class balance:** the class distribution will be uneven and must be measured and reported. It is handled with class-weighted loss or resampling.
- A fixed random seed makes the dataset reproducible.

---

## 9. Component 5: The neural network (`model.py`)

### 9.1 Task

Input: one `N x N` binary grid. Output: probabilities over the four fate classes. This is image classification on a tiny binary image, so a convolutional neural network (CNN) is the natural fit.

### 9.2 Why a CNN

Game of Life rules are local (each cell depends only on its 3 x 3 neighborhood) and the same everywhere on the board. Convolutions share weights across positions and look at local neighborhoods, so the architecture matches the problem's structure.

### 9.3 Architecture (`FateNet`)

```text
Input            1 x 32 x 32
Conv 3x3, 32 ch   + ReLU     (circular padding)
Conv 3x3, 64 ch   + ReLU     (circular padding)
Conv 3x3, 64 ch   + ReLU     (circular padding)
Conv 3x3, 128 ch  + ReLU     (circular padding)
Global pooling (average and max, concatenated)
Linear 256 -> 64 + ReLU + Dropout
Linear 64 -> 4              (logits)
```

Key decisions:

- **Circular padding** in the convolutions, so the network sees the same wraparound topology as the simulation.
- **Global pooling** at the end so the classification does not depend on where on the board a pattern sits.
- Small enough to train on a CPU in minutes to an hour. Depth and width are tunable; the receptive field should grow large enough to relate distant parts of a pattern.

### 9.4 Training (`train.py`)

- **Loss:** cross-entropy (optionally class-weighted).
- **Optimizer:** Adam, learning rate about 1e-3, with a scheduler (reduce on plateau or cosine decay).
- **Batch size:** around 256. Epochs: tens, with early stopping on validation loss.
- **Split:** 80 / 10 / 10 train / validation / test, done **before** augmentation and **after** deduplication, so no pattern leaks across splits.
- **Augmentation (training set only):** random rotations by 90 degrees, flips, and cyclic shifts of the board. These transformations do not change the fate on a wrapped board, so the labels stay valid and the model learns the symmetry.
- **Checkpointing:** the best-validation checkpoint is saved to `models/` together with metadata (board size, class names, training config) so the app can verify the model matches its board.
- Seeds are fixed for reproducibility.

### 9.5 Evaluation (`evaluate.py`)

Run on the held-out test set only:

- Overall accuracy, per-class precision, recall and F1, and a confusion matrix.
- Accuracy broken down by source generator (random soups vs sparse patches vs library patterns).
- Accuracy broken down by how long the pattern takes to settle. Patterns with long transients are expected to be the hardest.
- **Calibration:** whether a predicted 80% confidence is actually correct about 80% of the time, since the app shows probabilities to the user.
- **Baselines** to compare against, so the network's value is measured rather than assumed:
  1. Always predict the most common class.
  2. A simple classifier on hand-built features (population, density, bounding box size, number of connected clusters).
  3. A fully connected network on the flattened grid.
  4. The CNN.

---

## 10. Component 6: Inference integration (`predictor.py`)

`Predictor` loads the saved checkpoint once at startup, switches the model to evaluation mode, and exposes `predict(grid) -> probabilities`.

- Inference on a single 32 x 32 board is a few milliseconds on a CPU, so no GPU, server or API is needed. The app calls the predictor as a normal function.
- If no checkpoint exists or the board size does not match, the prediction panel shows "no model loaded" and the rest of the app works normally.
- **When predictions run:** after the board is edited (with a short debounce so it is not recomputed on every pixel of a drag), and, if live prediction is on, periodically while the simulation runs.

### 10.1 A useful consistency check

The fate of a pattern is the same as the fate of any later state on its own trajectory, since both follow the same path. So while the simulation is running, the prediction should stay roughly stable from generation to generation. Large swings in predicted probabilities along a single trajectory reveal where the model is unreliable. The app makes this visible simply by predicting live while running.

### 10.2 Verify

Pressing V copies the current board and runs the oracle (typically tens of milliseconds, fast enough to run directly in the UI loop; a worker thread is only needed if it proves too slow). It shows the true fate, period and settle time next to the prediction, and updates the scoreboard.

---

## 11. Configuration (`config.py`)

A single place for shared constants so the app, data generator and model cannot drift apart:

```text
BOARD_SIZE        = 32
MAX_GENERATIONS   = 1000
CLASS_NAMES       = [EXTINCT, STABLE, OSCILLATING, UNSETTLED]
DATASET_PATH      = data/fate_dataset.npz
MODEL_PATH        = models/fate_net.pt
CELL_PIXELS       = 20
DEFAULT_SPEED     = 10 generations per second
```

If `BOARD_SIZE` changes, the dataset must be regenerated and the model retrained.

---

## 12. Testing and validation

**Engine**
- A single live cell dies in one step.
- A 2 x 2 block never changes.
- A blinker has period 2 (vertical to horizontal and back).
- A glider returns to its original shape displaced by (1, 1) every 4 generations.
- Fast and reference implementations agree on random boards for many generations.
- Wraparound: a glider crossing an edge reappears on the other side.

**Oracle**
- Lone cell -> `EXTINCT`, block -> `STABLE`, blinker -> `OSCILLATING` with period 2.
- A glider on an N x N wrapped board -> `OSCILLATING` with period 4N (128 on a 32 x 32 board).
- A pattern given a tiny generation limit -> `UNSETTLED`.

**Data**
- Dataset shapes and label ranges are correct; the class distribution is logged.
- No duplicate boards across train, validation and test splits.
- Rotating or flipping a board leaves its oracle label unchanged.

**Model**
- Output shape and probabilities summing to 1.
- The model can overfit a tiny batch (a basic training sanity check).
- Test accuracy is clearly above the baselines in section 9.5.

---

## 13. Limitations and what to expect

- **Prediction in general is provably hard.** The Game of Life is Turing complete, so no method can predict the fate of every possible pattern without simulating it. The network learns statistical regularities, not a general solution, and it will make mistakes, especially on patterns with long, chaotic transients.
- **Labels are defined relative to the setup:** this board size, wraparound topology and generation limit. Results do not transfer to an infinite board.
- **`UNSETTLED` is a catch-all** for "did not settle within the limit".
- **Distribution shift:** the model is strongest on the kinds of patterns it was trained on. Unusual hand-drawn patterns may be predicted poorly, which is why the dataset mixes several sources.
- **Probabilities are estimates.** Calibration is measured, not assumed.

The mismatches are a feature, not just a flaw: they show which patterns the network can and cannot "intuit".

---

## 14. Build order

1. **Engine:** reference `step`, then the vectorized version, with tests.
2. **App, part 1:** window, grid drawing, mouse painting, play/pause/step/speed/clear/random.
3. **Oracle:** `run_until_settled` with cycle detection and tests.
4. **App, part 2:** pattern library and stamping, save/load, Verify and Fast-forward.
5. **Data generation:** generators, parallel labeling, deduplication, dataset file, class distribution report.
6. **Baselines:** simple feature classifier and majority baseline, to set the bar.
7. **Model:** `FateNet`, training loop, augmentation, checkpointing.
8. **Evaluation:** metrics, confusion matrix, breakdowns, calibration.
9. **Integration:** `Predictor`, live prediction panel, scoreboard.
10. **Polish:** README, requirements, error messages, performance pass.

Steps 1-4 produce a complete, playable Game of Life application. Steps 5-9 add the intelligence.

---

## 15. Possible extensions

- Extra output heads predicting time to settle and final population (the data already contains these).
- Cell age coloring or a fading trail of recently died cells in the UI.
- Larger boards with a model trained for that size.
- Importing and exporting standard RLE pattern files.
- Visualizing what the network attends to (for example, which cells most affect the prediction).
- A "stability explorer" mode that adds or removes single cells and shows how the predicted fate changes.
- Comparing CNN depth and width, or trying a small residual network.