"""Shared configuration constants for Life Oracle."""

from pathlib import Path

# ── Board ───────────────────────────────────────────────────────────────────
BOARD_SIZE: int = 32          # cells per side; changing requires regen + retrain
MAX_GENERATIONS: int = 1000   # oracle generation limit

# ── Fate classes ────────────────────────────────────────────────────────────
CLASS_NAMES = ["EXTINCT", "STABLE", "OSCILLATING", "UNSETTLED"]
EXTINCT     = 0
STABLE      = 1
OSCILLATING = 2
UNSETTLED   = 3

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
PATTERNS_DIR = ROOT / "patterns"

DATASET_PATH = DATA_DIR  / "fate_dataset.npz"
MODEL_PATH   = MODELS_DIR / "fate_net.pt"

# ── UI ───────────────────────────────────────────────────────────────────────
CELL_PIXELS    = 20          # px per cell
DEFAULT_SPEED  = 10          # generations per second
TARGET_FPS     = 60
PREDICT_DEBOUNCE_MS = 300   # ms to wait before rerunning predictor after a brush stroke
