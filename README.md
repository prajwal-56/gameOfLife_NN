# Life Oracle

A native desktop Game of Life simulator with a neural-network fate predictor.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m life.app          # play the game
```

## Commands

| Command | Description |
|---|---|
| `python -m life.app` | Launch the desktop app |
| `python -m life.generate_data` | Generate labeled training dataset (default 30k) |
| `python -m life.generate_data --n 5000` | Quick 5k dataset for testing |
| `python -m life.train` | Train FateNet (60 epochs, early stopping) |
| `python -m life.train --epochs 30 --batch 128` | Custom training config |
| `python -m life.evaluate` | Evaluate model + baselines on test split |
| `python -m life.evaluate --plots` | Also save confusion matrix + calibration PNG |
| `pytest` | Run all tests |

## Full ML pipeline (after app is working)

```bash
# 1. Generate dataset (~5-10 min for 30k samples)
python -m life.generate_data

# 2. Train (~10-30 min on CPU depending on dataset size)
python -m life.train

# 3. Evaluate
python -m life.evaluate --plots

# 4. Launch app — predictor loads automatically
python -m life.app
```

## Controls

| Key / Mouse | Action |
|---|---|
| Left click/drag | Paint live cells |
| Right click/drag | Erase cells |
| Space | Play / pause |
| → or N | Step one generation |
| ↑ / ↓ | Increase / decrease speed |
| C | Clear board |
| R | Random soup |
| G | Toggle grid lines |
| 1–9 | Select stamp pattern |
| Q / E | Rotate stamp CCW / CW |
| Click (stamp mode) | Place stamp |
| P | Toggle live prediction |
| V | Verify: run oracle on current board |
| F | Fast-forward to settled state |
| S / L | Save / load pattern |
| Esc | Cancel stamp / quit |

## Project layout

```
life/
  config.py       shared constants
  engine.py       simulation (vectorised + reference)
  oracle.py       fate detection with cycle detection
  patterns.py     built-in pattern library + file I/O
  predictor.py    inference wrapper (needs trained model)
  model.py        FateNet CNN definition
  app.py          pygame UI (entry point)
data/             generated datasets (not committed)
models/           trained checkpoints
patterns/         saved .cells files
tests/            pytest test suite
```

## Fate classes

| Class | Meaning |
|---|---|
| EXTINCT | All cells die |
| STABLE | Settles into a still life |
| OSCILLATING | Repeating cycle of period ≥ 2 |
| UNSETTLED | No repeat within generation limit |
