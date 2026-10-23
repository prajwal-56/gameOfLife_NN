"""
Life Oracle – pygame desktop application.
Entry point: python -m life.app

Layout (resizable)
------------------
+--------------------+-------+
|                    |       |
|      BOARD         | PANEL |
|                    |       |
+--------------------+-------+

All geometry is recomputed each frame from the live window size.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

from .config import (
    DEFAULT_SPEED, TARGET_FPS,
    PATTERNS_DIR, PREDICT_DEBOUNCE_MS, CLASS_NAMES,
    BOARD_SIZE as _CFG_BOARD_SIZE,
    CELL_PIXELS as _CFG_CELL_PX,
)
from .engine import step, empty_grid, random_grid, Grid
from .oracle import run_until_settled, FateResult
from .patterns import (
    PATTERN_ORDER, get_pattern, stamp, rotate_pattern,
    load_cells, save_cells,
)

# ── Optional predictor ────────────────────────────────────────────────────────
try:
    from .predictor import Predictor
    _PREDICTOR_AVAILABLE = True
except Exception:
    _PREDICTOR_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  Minimal monochrome palette
# ─────────────────────────────────────────────────────────────────────────────
C_BG         = (15,  15,  15)   # near-black canvas
C_ALIVE      = (220, 220, 215)  # warm off-white cells
C_DEAD       = (15,  15,  15)   # same as canvas
C_GRID       = (35,  35,  35)   # very subtle grid lines
C_PANEL_BG   = (22,  22,  22)   # slightly lighter than canvas
C_DIVIDER    = (45,  45,  45)   # section separators
C_TEXT       = (180, 180, 175)  # primary text
C_DIM        = (80,  80,  75)   # de-emphasised text
C_ACCENT     = (200, 200, 195)  # headers / highlighted items
C_GOOD       = (140, 190, 140)  # muted green
C_BAD        = (190, 120, 115)  # muted red
C_STAMP_PRE  = (180, 180, 175, 80)  # semi-transparent stamp ghost

# Fate class colours — muted, not neon
C_FATE = {
    0: (160, 90,  85),   # EXTINCT    – dusty red
    1: (85,  140, 170),  # STABLE     – slate blue
    2: (170, 145, 75),   # OSCILLATING – warm amber
    3: (130, 95,  165),  # UNSETTLED  – muted violet
}

# Panel width (fixed); everything else scales with window
PANEL_W     = 260
MIN_WIN_W   = 600
MIN_WIN_H   = 480
INIT_WIN_W  = _CFG_BOARD_SIZE * _CFG_CELL_PX + PANEL_W
INIT_WIN_H  = max(_CFG_BOARD_SIZE * _CFG_CELL_PX, 600)

# Cell-size and board-size limits
CELL_PX_MIN, CELL_PX_MAX   = 4, 40
BOARD_SIZE_MIN, BOARD_SIZE_MAX = 8, 128


# ─────────────────────────────────────────────────────────────────────────────
#  Small drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _txt(surf: pygame.Surface, text: str, font: pygame.font.Font,
         x: int, y: int, color: tuple = C_TEXT) -> int:
    """Blit one line and return y of the next line."""
    surf.blit(font.render(text, True, color), (x, y))
    return y + font.get_linesize()


def _divider(surf: pygame.Surface, x: int, y: int, w: int) -> int:
    pygame.draw.line(surf, C_DIVIDER, (x, y + 3), (x + w, y + 3))
    return y + 8


def _bar(surf: pygame.Surface, x: int, y: int, w: int, h: int,
         frac: float, color: tuple, label: str, font: pygame.font.Font) -> int:
    """Labelled probability bar."""
    pygame.draw.rect(surf, (40, 40, 40), (x, y, w, h))
    fill = max(1, int(frac * w))
    pygame.draw.rect(surf, color, (x, y, fill, h))
    ts = font.render(f"{label}  {frac:.0%}", True, C_TEXT)
    surf.blit(ts, (x + 4, y + (h - ts.get_height()) // 2))
    return y + h + 2


# ─────────────────────────────────────────────────────────────────────────────
#  Application
# ─────────────────────────────────────────────────────────────────────────────

class App:
    """Main application object."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Life Oracle")

        self.screen = pygame.display.set_mode(
            (INIT_WIN_W, INIT_WIN_H),
            pygame.RESIZABLE,
        )
        self.clock = pygame.time.Clock()

        # Fonts (monospace keeps the panel tidy)
        self.font_sm  = pygame.font.SysFont("monospace", 11)
        self.font_med = pygame.font.SysFont("monospace", 13)
        self.font_lg  = pygame.font.SysFont("monospace", 14, bold=True)

        # ── Configurable display settings ────────────────────────────────
        self.cell_px:   int = _CFG_CELL_PX    # zoom: pixels per cell
        self.board_size: int = _CFG_BOARD_SIZE  # number of cells per side

        # ── Simulation state ─────────────────────────────────────────────
        self.grid: Grid = empty_grid(self.board_size)
        self.generation: int = 0
        self.running: bool   = False
        self.speed: float    = DEFAULT_SPEED
        self._last_step_t: float = time.perf_counter()

        # ── Drawing / editing state ──────────────────────────────────────
        self.show_grid: bool = True
        # paint mode: None = not dragging; int = value being painted (0 or 1)
        self._painting: Optional[int] = None
        # The cell that was toggled on mouse-down (used to avoid re-toggling
        # during drag over the same cell)
        self._paint_cell: Optional[tuple[int, int]] = None

        # ── Stamp state ──────────────────────────────────────────────────
        self._stamp_name: Optional[str]  = None
        self._stamp_pat:  Optional[Grid] = None
        self._stamp_rot:  int = 0

        # ── Prediction state ─────────────────────────────────────────────
        self.live_predict: bool = True
        self.probs: Optional[np.ndarray] = None
        self._last_edit_t: float = 0.0
        self._predict_scheduled: bool = False

        # ── Oracle / verify state ────────────────────────────────────────
        self.fate_result: Optional[FateResult] = None
        self._verify_running: bool = False

        # ── Scoreboard ───────────────────────────────────────────────────
        self.score_total:   int = 0
        self.score_correct: int = 0

        # ── Status message (transient) ───────────────────────────────────
        self._status_msg: str = ""
        self._status_t:   float = 0.0

        # ── Predictor (optional) ─────────────────────────────────────────
        self.predictor = None
        if _PREDICTOR_AVAILABLE:
            try:
                from .predictor import Predictor
                self.predictor = Predictor()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Geometry helpers (all dynamic)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def board_px(self) -> int:
        """Width/height of the board area in pixels."""
        return self.board_size * self.cell_px

    @property
    def panel_x(self) -> int:
        return self.screen.get_width() - PANEL_W

    def _cell_at(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        """Return (row, col) for a pixel position, or None if outside board."""
        x, y = pos
        bp = self.board_px
        if x < 0 or x >= bp or y < 0 or y >= bp:
            return None
        return y // self.cell_px, x // self.cell_px

    # ─────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        while True:
            self.clock.tick(TARGET_FPS)
            self._handle_events()
            self._maybe_step()
            self._maybe_predict()
            self._draw()
            pygame.display.flip()

    # ─────────────────────────────────────────────────────────────────────────
    #  Event handling
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()

            elif event.type == pygame.KEYDOWN:
                self._on_key(event.key, event.mod)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 3):
                    self._on_mouse_down(event.pos, event.button)
                elif event.button == 4:   # scroll up → zoom in
                    self._zoom(+1, event.pos)
                elif event.button == 5:   # scroll down → zoom out
                    self._zoom(-1, event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    self._painting   = None
                    self._paint_cell = None

            elif event.type == pygame.MOUSEMOTION:
                if self._painting is not None:
                    self._on_mouse_drag(event.pos)

            elif event.type == pygame.VIDEORESIZE:
                # pygame.RESIZABLE handles the surface automatically in pygame2+
                pass

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _on_mouse_down(self, pos: tuple[int, int], button: int) -> None:
        cell = self._cell_at(pos)
        if cell is None:
            return

        # Stamp placement (left-click only)
        if self._stamp_name and button == 1:
            r, c = cell
            self.grid = stamp(self.grid, self._stamp_pat, r, c, self._stamp_rot)
            self.generation = 0
            self._on_board_edit()
            return

        r, c = cell

        if button == 1:
            # Toggle the cell that was clicked; subsequent drag paints same value
            new_val = 1 - int(self.grid[r, c])
            self._painting   = new_val
            self._paint_cell = (r, c)
            self.grid[r, c]  = new_val
        else:
            # Right-click always erases
            self._painting   = 0
            self._paint_cell = (r, c)
            self.grid[r, c]  = 0

        self.generation = 0
        self._on_board_edit()

    def _on_mouse_drag(self, pos: tuple[int, int]) -> None:
        if self._stamp_name:
            return
        cell = self._cell_at(pos)
        if cell is None:
            return
        if cell == self._paint_cell:
            return   # already handled on mouse-down
        r, c = cell
        if self.grid[r, c] != self._painting:
            self.grid[r, c]  = self._painting
            self._paint_cell = cell
            self._on_board_edit()

    def _on_board_edit(self) -> None:
        self._last_edit_t = time.perf_counter()
        self.fate_result  = None

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _on_key(self, key: int, mod: int) -> None:
        # ── Simulation controls ───────────────────────────────────────────
        if key == pygame.K_SPACE:
            self.running = not self.running

        elif key in (pygame.K_RIGHT, pygame.K_n):
            self._advance()

        elif key == pygame.K_UP:
            self.speed = min(self.speed * 2, 200.0)

        elif key == pygame.K_DOWN:
            self.speed = max(self.speed / 2, 0.5)

        elif key == pygame.K_c:
            self.grid = empty_grid(self.board_size)
            self.generation = 0
            self.fate_result = None
            self._on_board_edit()

        elif key == pygame.K_r:
            self.grid = random_grid(self.board_size)
            self.generation = 0
            self.fate_result = None
            self._on_board_edit()

        elif key == pygame.K_g:
            self.show_grid = not self.show_grid

        elif key == pygame.K_p:
            self.live_predict = not self.live_predict

        elif key == pygame.K_v:
            self._verify()

        elif key == pygame.K_f:
            self._fast_forward()

        elif key == pygame.K_s:
            self._save_dialog()

        elif key == pygame.K_l:
            self._load_dialog()

        # ── Zoom ([ and ]) ────────────────────────────────────────────────
        elif key == pygame.K_LEFTBRACKET:
            self._zoom(-1)

        elif key == pygame.K_RIGHTBRACKET:
            self._zoom(+1)

        # ── Board size (< and >) ──────────────────────────────────────────
        elif key == pygame.K_COMMA:   # < (without shift on most layouts)
            self._resize_board(self.board_size - 4)

        elif key == pygame.K_PERIOD:  # >
            self._resize_board(self.board_size + 4)

        # ── Stamp rotation ────────────────────────────────────────────────
        elif key == pygame.K_q and self._stamp_name:
            self._stamp_rot = (self._stamp_rot - 1) % 4
            self._stamp_pat = rotate_pattern(get_pattern(self._stamp_name), self._stamp_rot)

        elif key == pygame.K_e and self._stamp_name:
            self._stamp_rot = (self._stamp_rot + 1) % 4
            self._stamp_pat = rotate_pattern(get_pattern(self._stamp_name), self._stamp_rot)

        # ── Esc ───────────────────────────────────────────────────────────
        elif key == pygame.K_ESCAPE:
            if self._stamp_name:
                self._stamp_name = None
                self._stamp_pat  = None
                self._stamp_rot  = 0
            else:
                self._quit()

        # ── 1-9: pattern library ──────────────────────────────────────────
        elif pygame.K_1 <= key <= pygame.K_9:
            idx = key - pygame.K_1
            if idx < len(PATTERN_ORDER):
                name = PATTERN_ORDER[idx]
                self._stamp_name = name
                self._stamp_rot  = 0
                self._stamp_pat  = get_pattern(name)

    # ── Zoom & resize ─────────────────────────────────────────────────────────

    def _zoom(self, delta: int, mouse_pos: tuple[int, int] | None = None) -> None:
        new_px = max(CELL_PX_MIN, min(CELL_PX_MAX, self.cell_px + delta))
        self.cell_px = new_px

    def _resize_board(self, new_size: int) -> None:
        new_size = max(BOARD_SIZE_MIN, min(BOARD_SIZE_MAX, new_size))
        if new_size == self.board_size:
            return
        # Crop or pad with zeros
        old = self.grid
        new = empty_grid(new_size)
        copy_r = min(old.shape[0], new_size)
        copy_c = min(old.shape[1], new_size)
        new[:copy_r, :copy_c] = old[:copy_r, :copy_c]
        self.grid       = new
        self.board_size = new_size
        self.generation = 0
        self.fate_result = None
        self._on_board_edit()
        self._status(f"Board: {new_size}×{new_size}")

    def _status(self, msg: str, duration: float = 2.0) -> None:
        self._status_msg = msg
        self._status_t   = time.perf_counter() + duration

    # ─────────────────────────────────────────────────────────────────────────
    #  Simulation
    # ─────────────────────────────────────────────────────────────────────────

    def _advance(self) -> None:
        self.grid = step(self.grid)
        self.generation += 1
        self.fate_result = None
        if self.live_predict:
            self._last_edit_t = time.perf_counter()

    def _maybe_step(self) -> None:
        if not self.running:
            return
        now      = time.perf_counter()
        interval = 1.0 / self.speed
        if now - self._last_step_t >= interval:
            self._advance()
            self._last_step_t = now

    # ─────────────────────────────────────────────────────────────────────────
    #  Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_predict(self) -> None:
        if not self.live_predict or self.predictor is None:
            return
        if self._last_edit_t == 0.0:
            return
        elapsed_ms = (time.perf_counter() - self._last_edit_t) * 1000
        if elapsed_ms >= PREDICT_DEBOUNCE_MS and not self._predict_scheduled:
            self._predict_scheduled = True
            snap = self.grid.copy()
            def _run():
                p = self.predictor.predict(snap)
                self.probs = p
                self._predict_scheduled = False
                self._last_edit_t = 0.0
            threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Oracle / verify
    # ─────────────────────────────────────────────────────────────────────────

    def _verify(self) -> None:
        if self._verify_running:
            return
        self._verify_running = True
        snap = self.grid.copy()
        def _run():
            result = run_until_settled(snap)
            self.fate_result = result
            if self.probs is not None:
                pred = int(np.argmax(self.probs))
                self.score_total  += 1
                if pred == result.label:
                    self.score_correct += 1
            self._verify_running = False
        threading.Thread(target=_run, daemon=True).start()

    def _fast_forward(self) -> None:
        result = run_until_settled(self.grid.copy())
        self.fate_result = result
        g = self.grid.copy()
        target = result.settle_time if result.settle_time is not None else result.generations_run
        for _ in range(target):
            g = step(g)
        self.grid = g
        self.generation = target
        self._on_board_edit()

    # ─────────────────────────────────────────────────────────────────────────
    #  Save / load
    # ─────────────────────────────────────────────────────────────────────────

    def _save_dialog(self) -> None:
        PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
        idx  = len(list(PATTERNS_DIR.glob("*.cells")))
        path = PATTERNS_DIR / f"pattern_{idx:03d}.cells"
        save_cells(path, self.grid)
        self._status(f"Saved → {path.name}")

    def _load_dialog(self) -> None:
        files = sorted(PATTERNS_DIR.glob("*.cells"))
        if not files:
            self._status("No saved patterns found.")
            return
        path = files[-1]
        pat  = load_cells(path)
        ph, pw = pat.shape
        r0 = max(0, (self.board_size - ph) // 2)
        c0 = max(0, (self.board_size - pw) // 2)
        self.grid = empty_grid(self.board_size)
        self.grid = stamp(self.grid, pat, r0, c0)
        self.generation  = 0
        self.fate_result = None
        self._on_board_edit()
        self._status(f"Loaded ← {path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Drawing
    # ─────────────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.screen.fill(C_BG)
        self._draw_board()
        self._draw_stamp_preview()
        self._draw_panel()
        self._draw_status()

    def _draw_board(self) -> None:
        cp  = self.cell_px
        bs  = self.board_size
        bp  = self.board_px
        surf = self.screen

        # Dead background for the board area
        pygame.draw.rect(surf, C_DEAD, (0, 0, bp, bp))

        # Live cells
        for r in range(bs):
            for c in range(bs):
                if self.grid[r, c]:
                    pygame.draw.rect(surf, C_ALIVE,
                                     (c * cp + 1, r * cp + 1, cp - 1, cp - 1))

        # Grid lines
        if self.show_grid and cp >= 6:
            for i in range(bs + 1):
                pygame.draw.line(surf, C_GRID, (i * cp, 0), (i * cp, bp))
                pygame.draw.line(surf, C_GRID, (0, i * cp), (bp, i * cp))

    def _draw_stamp_preview(self) -> None:
        if self._stamp_pat is None:
            return
        mx, my = pygame.mouse.get_pos()
        bp = self.board_px
        if mx >= bp or my >= bp:
            return

        cp = self.cell_px
        bs = self.board_size
        cr, cc = my // cp, mx // cp

        overlay = pygame.Surface((bp, bp), pygame.SRCALPHA)
        ph, pw  = self._stamp_pat.shape
        for r in range(ph):
            for c in range(pw):
                if self._stamp_pat[r, c]:
                    dr = (cr + r) % bs
                    dc = (cc + c) % bs
                    pygame.draw.rect(overlay, C_STAMP_PRE,
                                     (dc * cp, dr * cp, cp, cp))
        self.screen.blit(overlay, (0, 0))

    def _draw_panel(self) -> None:
        surf   = self.screen
        win_h  = surf.get_height()
        px     = self.panel_x
        pw     = PANEL_W
        pad    = 12            # left/right padding inside panel
        x      = px + pad
        w      = pw - pad * 2
        y      = 12

        # Panel background
        pygame.draw.rect(surf, C_PANEL_BG, (px, 0, pw, win_h))
        # Left border line
        pygame.draw.line(surf, C_DIVIDER, (px, 0), (px, win_h))

        # ── Title ─────────────────────────────────────────────────────────
        y = _txt(surf, "Life Oracle", self.font_lg, x, y, C_ACCENT)
        y += 2
        y = _divider(surf, x, y, w)

        # ── Sim stats ──────────────────────────────────────────────────────
        pop    = int(self.grid.sum())
        status = "run" if self.running else "pause"
        y = _txt(surf, f"gen {self.generation}   pop {pop}", self.font_sm, x, y, C_TEXT)
        y = _txt(surf, f"{status}   {self.speed:.1f} g/s", self.font_sm, x, y, C_GOOD if self.running else C_DIM)
        y += 6

        # ── Settings ──────────────────────────────────────────────────────
        y = _divider(surf, x, y, w)
        y = _txt(surf, "settings", self.font_lg, x, y, C_ACCENT)
        y = _txt(surf, f"zoom  {self.cell_px}px/cell   [/]", self.font_sm, x, y, C_DIM)
        y = _txt(surf, f"grid  {self.board_size}×{self.board_size}   </> (±4)", self.font_sm, x, y, C_DIM)
        y = _txt(surf, f"lines {'on' if self.show_grid else 'off'}   G", self.font_sm, x, y, C_DIM)
        y += 6

        # ── Prediction ────────────────────────────────────────────────────
        y = _divider(surf, x, y, w)
        pred_hdr = "predict" + ("" if self.live_predict else " [off]")
        y = _txt(surf, pred_hdr, self.font_lg, x, y, C_ACCENT if self.live_predict else C_DIM)

        if self.predictor is None:
            y = _txt(surf, "no model — train first", self.font_sm, x, y, C_DIM)
        elif self.probs is None:
            y = _txt(surf, "draw something…", self.font_sm, x, y, C_DIM)
        else:
            top = int(np.argmax(self.probs))
            for i, name in enumerate(CLASS_NAMES):
                y = _bar(surf, x, y, w, 16, float(self.probs[i]),
                         C_FATE[i], name.lower(), self.font_sm)
            y += 2
            y = _txt(surf, f"→ {CLASS_NAMES[top].lower()}", self.font_med, x, y, C_FATE[top])
        y += 6

        # ── Verify ────────────────────────────────────────────────────────
        y = _divider(surf, x, y, w)
        y = _txt(surf, "verify  V", self.font_lg, x, y, C_ACCENT)

        if self._verify_running:
            y = _txt(surf, "running…", self.font_sm, x, y, C_DIM)
        elif self.fate_result is None:
            y = _txt(surf, "press V to run oracle", self.font_sm, x, y, C_DIM)
        else:
            fr = self.fate_result
            fc = C_FATE[fr.label]
            y = _txt(surf, fr.label_name.lower(), self.font_med, x, y, fc)
            if fr.period is not None:
                y = _txt(surf, f"period {fr.period}", self.font_sm, x, y, C_DIM)
            if fr.settle_time is not None:
                y = _txt(surf, f"settled gen {fr.settle_time}", self.font_sm, x, y, C_DIM)
            y = _txt(surf, f"pop {fr.final_population}", self.font_sm, x, y, C_DIM)

            if self.probs is not None:
                pred_lbl = int(np.argmax(self.probs))
                match    = pred_lbl == fr.label
                y = _txt(surf,
                         "match ✓" if match else "mismatch ✗",
                         self.font_med, x, y,
                         C_GOOD if match else C_BAD)
        y += 6

        # ── Scoreboard ────────────────────────────────────────────────────
        y = _divider(surf, x, y, w)
        y = _txt(surf, "score", self.font_lg, x, y, C_ACCENT)
        if self.score_total == 0:
            y = _txt(surf, "—", self.font_sm, x, y, C_DIM)
        else:
            acc = self.score_correct / self.score_total
            y = _txt(surf,
                     f"{self.score_correct}/{self.score_total}  {acc:.0%}",
                     self.font_med, x, y,
                     C_GOOD if acc >= 0.7 else C_BAD)
        y += 6

        # ── Pattern library ───────────────────────────────────────────────
        y = _divider(surf, x, y, w)
        y = _txt(surf, "patterns  1-9  Q/E rotate", self.font_lg, x, y, C_ACCENT)
        for i, name in enumerate(PATTERN_ORDER[:9]):
            col = C_TEXT if name == self._stamp_name else C_DIM
            y   = _txt(surf, f"{i+1} {name}", self.font_sm, x, y, col)
            if y > win_h - 80:
                break
        y += 6

        # ── Controls ──────────────────────────────────────────────────────
        if y < win_h - 20:
            y = _divider(surf, x, y, w)
            y = _txt(surf, "controls", self.font_lg, x, y, C_ACCENT)
            for line in [
                "space  play/pause",
                "→ / n  step",
                "↑ ↓    speed",
                "c      clear",
                "r      random",
                "f      fast-forward",
                "s / l  save / load",
                "scroll zoom",
                "esc    quit",
            ]:
                if y + 14 > win_h - 4:
                    break
                y = _txt(surf, line, self.font_sm, x, y, C_DIM)

    def _draw_status(self) -> None:
        """Transient status message drawn at bottom of board area."""
        if not self._status_msg:
            return
        if time.perf_counter() > self._status_t:
            self._status_msg = ""
            return
        surf = self.screen
        label = self.font_sm.render(self._status_msg, True, C_ACCENT)
        bx, by = 8, self.board_px - label.get_height() - 6
        by = max(4, by)
        # subtle dark pill behind the text
        pygame.draw.rect(surf, (20, 20, 20),
                         (bx - 4, by - 3, label.get_width() + 10, label.get_height() + 6),
                         border_radius=3)
        surf.blit(label, (bx, by))

    # ─────────────────────────────────────────────────────────────────────────
    #  Quit
    # ─────────────────────────────────────────────────────────────────────────

    def _quit(self) -> None:
        pygame.quit()
        sys.exit(0)


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
