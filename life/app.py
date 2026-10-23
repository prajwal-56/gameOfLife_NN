"""
Life Oracle – modern minimal desktop application.
Entry point: python -m life.app

Layout (resizable)
------------------
+-----------------------------------+-------------------+
|                                   |                   |
|        SIMULATION BOARD           |    SIDE PANEL     |
|       (toroidal canvas)           | (cards & metrics) |
|                                   |                   |
+-----------------------------------+-------------------+

Dynamic geometry recalculation ensures responsiveness across window resizes.
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
#  Modern Minimalist Palette (High-Contrast, Refined Dark Mode)
# ─────────────────────────────────────────────────────────────────────────────
C_CANVAS      = (13,  14,  16)    # deep matte obsidian canvas
C_ALIVE       = (242, 244, 248)   # clean porcelain white cells
C_DEAD        = (13,  14,  16)    # same as canvas
C_GRID_LINE   = (27,  29,  34)    # subtle unobtrusive grid line

C_PANEL_BG    = (20,  22,  27)    # panel background
C_PANEL_EDGE  = (36,  40,  49)    # separator border

C_CARD_BG     = (26,  29,  36)    # card background
C_CARD_EDGE   = (42,  46,  56)    # card border outline
C_CARD_ACTIVE = (55,  62,  75)    # active item border

C_TEXT_WHITE  = (255, 255, 255)   # highest contrast pure white
C_TEXT_MAIN   = (225, 230, 238)   # primary high-contrast text
C_TEXT_MUTED  = (165, 174, 186)   # clear secondary silver/gray
C_TEXT_FAINT  = (115, 125, 140)   # subdued hints / brackets

C_CHIP_BG     = (34,  38,  47)    # key tag badge background
C_CHIP_EDGE   = (48,  54,  66)    # key tag border

C_STAMP_GHOST = (242, 244, 248, 85) # translucent stamp placement ghost

# Status Indicators
C_RUN_BG      = (22,  48,  34)
C_RUN_TEXT    = (75,  215, 130)
C_RUN_EDGE    = (35,  85,  55)

C_PAUSE_BG    = (34,  37,  45)
C_PAUSE_TEXT  = (180, 186, 198)
C_PAUSE_EDGE  = (52,  56,  68)

C_MATCH_BG    = (20,  45,  30)
C_MATCH_TEXT  = (80,  220, 130)
C_MISMATCH_BG = (48,  25,  25)
C_MISMATCH_TEXT = (240, 110, 110)

# Refined Fate Colors (Matte Modern Tones)
C_FATE = {
    0: (235, 105, 105),   # EXTINCT    – rose coral
    1: (95,  175, 245),   # STABLE     – steel sky blue
    2: (245, 185, 75),    # OSCILLATING – warm amber gold
    3: (185, 135, 245),   # UNSETTLED  – amethyst violet
}

# Dimensions
PANEL_W        = 320
MIN_WIN_W      = 720
MIN_WIN_H      = 540
INIT_WIN_W     = _CFG_BOARD_SIZE * _CFG_CELL_PX + PANEL_W
INIT_WIN_H     = max(_CFG_BOARD_SIZE * _CFG_CELL_PX, 640)

CELL_PX_MIN, CELL_PX_MAX       = 4, 40
BOARD_SIZE_MIN, BOARD_SIZE_MAX = 8, 128


# ─────────────────────────────────────────────────────────────────────────────
#  Font Loader with Modern Sans-Serif and Monospace Fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(family_names: str, size: int, bold: bool = False) -> pygame.font.Font:
    """Attempt to load the cleanest available font from candidate families."""
    for name in family_names.split(","):
        name = name.strip()
        matched = pygame.font.match_font(name, bold=bold)
        if matched:
            try:
                return pygame.font.Font(matched, size)
            except Exception:
                pass
    return pygame.font.SysFont(family_names.split(",")[0].strip(), size, bold=bold)


# ─────────────────────────────────────────────────────────────────────────────
#  Application
# ─────────────────────────────────────────────────────────────────────────────

class App:
    """Main application object."""

    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Life Oracle")

        self.screen = pygame.display.set_mode(
            (INIT_WIN_W, INIT_WIN_H),
            pygame.RESIZABLE,
        )
        self.clock = pygame.time.Clock()

        # Load modern typography
        sans = "adwaitasans, notosans, dejavusans, freesans, helvetica, arial, sans-serif"
        mono = "adwaitamono, hack, freemono, dejavusansmono, monospace"

        self.font_title   = _load_font(sans, 16, bold=True)
        self.font_card_h  = _load_font(sans, 13, bold=True)
        self.font_body    = _load_font(sans, 13, bold=False)
        self.font_small   = _load_font(sans, 11, bold=False)
        self.font_mono    = _load_font(mono, 13, bold=False)
        self.font_mono_sm = _load_font(mono, 11, bold=False)
        self.font_key     = _load_font(mono, 11, bold=True)

        # ── Configurable display settings ────────────────────────────────
        self.cell_px:   int = _CFG_CELL_PX      # zoom: pixels per cell
        self.board_size: int = _CFG_BOARD_SIZE   # grid dimension

        # ── Simulation state ─────────────────────────────────────────────
        self.grid: Grid = empty_grid(self.board_size)
        self.generation: int = 0
        self.running: bool   = False
        self.speed: float    = DEFAULT_SPEED
        self._last_step_t: float = time.perf_counter()

        # ── Drawing state ────────────────────────────────────────────────
        self.show_grid: bool = True
        self._painting: Optional[int] = None
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

        # ── Floating status notification ─────────────────────────────────
        self._status_msg: str   = ""
        self._status_t:   float = 0.0

        # ── Predictor instance (optional) ────────────────────────────────
        self.predictor = None
        if _PREDICTOR_AVAILABLE:
            try:
                from .predictor import Predictor
                self.predictor = Predictor()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Geometry Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def board_px(self) -> int:
        return self.board_size * self.cell_px

    @property
    def panel_x(self) -> int:
        return self.screen.get_width() - PANEL_W

    def _cell_at(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        x, y = pos
        bp = self.board_px
        if x < 0 or x >= bp or y < 0 or y >= bp:
            return None
        return y // self.cell_px, x // self.cell_px

    # ─────────────────────────────────────────────────────────────────────────
    #  Main Loop
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
    #  Events
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
                elif event.button == 4:   # Scroll up -> Zoom in
                    self._zoom(+1)
                elif event.button == 5:   # Scroll down -> Zoom out
                    self._zoom(-1)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    self._painting   = None
                    self._paint_cell = None

            elif event.type == pygame.MOUSEMOTION:
                if self._painting is not None:
                    self._on_mouse_drag(event.pos)

            elif event.type == pygame.VIDEORESIZE:
                pass

    def _on_mouse_down(self, pos: tuple[int, int], button: int) -> None:
        cell = self._cell_at(pos)
        if cell is None:
            return

        # Stamp placement
        if self._stamp_name and button == 1:
            r, c = cell
            self.grid = stamp(self.grid, self._stamp_pat, r, c, self._stamp_rot)
            self.generation = 0
            self._on_board_edit()
            return

        r, c = cell
        if button == 1:
            # Toggle cell state
            new_val = 1 - int(self.grid[r, c])
            self._painting   = new_val
            self._paint_cell = (r, c)
            self.grid[r, c]  = new_val
        else:
            # Right click erases
            self._painting   = 0
            self._paint_cell = (r, c)
            self.grid[r, c]  = 0

        self.generation = 0
        self._on_board_edit()

    def _on_mouse_drag(self, pos: tuple[int, int]) -> None:
        if self._stamp_name:
            return
        cell = self._cell_at(pos)
        if cell is None or cell == self._paint_cell:
            return
        r, c = cell
        if self.grid[r, c] != self._painting:
            self.grid[r, c]  = self._painting
            self._paint_cell = cell
            self._on_board_edit()

    def _on_board_edit(self) -> None:
        self._last_edit_t = time.perf_counter()
        self.fate_result  = None

    def _on_key(self, key: int, mod: int) -> None:
        if key == pygame.K_SPACE:
            self.running = not self.running

        elif key in (pygame.K_RIGHT, pygame.K_n):
            self._advance()

        elif key == pygame.K_UP:
            self.speed = min(self.speed * 2, 200.0)
            self._status(f"Speed: {self.speed:.1f} g/s")

        elif key == pygame.K_DOWN:
            self.speed = max(self.speed / 2, 0.5)
            self._status(f"Speed: {self.speed:.1f} g/s")

        elif key == pygame.K_c:
            self.grid = empty_grid(self.board_size)
            self.generation = 0
            self.fate_result = None
            self._on_board_edit()
            self._status("Grid Cleared")

        elif key == pygame.K_r:
            self.grid = random_grid(self.board_size)
            self.generation = 0
            self.fate_result = None
            self._on_board_edit()
            self._status("Randomized Board")

        elif key == pygame.K_g:
            self.show_grid = not self.show_grid
            self._status(f"Grid lines: {'ON' if self.show_grid else 'OFF'}")

        elif key == pygame.K_p:
            self.live_predict = not self.live_predict
            self._status(f"Prediction: {'ON' if self.live_predict else 'OFF'}")

        elif key == pygame.K_v:
            self._verify()

        elif key == pygame.K_f:
            self._fast_forward()

        elif key == pygame.K_s:
            self._save_pattern()

        elif key == pygame.K_l:
            self._load_pattern()

        elif key == pygame.K_LEFTBRACKET:
            self._zoom(-1)

        elif key == pygame.K_RIGHTBRACKET:
            self._zoom(+1)

        elif key == pygame.K_COMMA:
            self._resize_board(self.board_size - 4)

        elif key == pygame.K_PERIOD:
            self._resize_board(self.board_size + 4)

        elif key == pygame.K_q and self._stamp_name:
            self._stamp_rot = (self._stamp_rot - 1) % 4
            self._stamp_pat = rotate_pattern(get_pattern(self._stamp_name), self._stamp_rot)
            self._status(f"Rotated {self._stamp_name} ({self._stamp_rot * 90}°)")

        elif key == pygame.K_e and self._stamp_name:
            self._stamp_rot = (self._stamp_rot + 1) % 4
            self._stamp_pat = rotate_pattern(get_pattern(self._stamp_name), self._stamp_rot)
            self._status(f"Rotated {self._stamp_name} ({self._stamp_rot * 90}°)")

        elif key == pygame.K_ESCAPE:
            if self._stamp_name:
                self._stamp_name = None
                self._stamp_pat  = None
                self._stamp_rot  = 0
                self._status("Stamp Deselected")
            else:
                self._quit()

        elif pygame.K_1 <= key <= pygame.K_9:
            idx = key - pygame.K_1
            if idx < len(PATTERN_ORDER):
                name = PATTERN_ORDER[idx]
                self._stamp_name = name
                self._stamp_rot  = 0
                self._stamp_pat  = get_pattern(name)
                self._status(f"Stamp Selected: {name} (Click board to place)")

    # ── Simulation & Prediction ───────────────────────────────────────────────

    def _advance(self) -> None:
        self.grid = step(self.grid)
        self.generation += 1
        self.fate_result = None
        if self.live_predict:
            self._last_edit_t = time.perf_counter()

    def _maybe_step(self) -> None:
        if not self.running:
            return
        now = time.perf_counter()
        if now - self._last_step_t >= 1.0 / self.speed:
            self._advance()
            self._last_step_t = now

    def _maybe_predict(self) -> None:
        if not self.live_predict or self.predictor is None or self._last_edit_t == 0.0:
            return
        elapsed_ms = (time.perf_counter() - self._last_edit_t) * 1000
        if elapsed_ms >= PREDICT_DEBOUNCE_MS and not self._predict_scheduled:
            self._predict_scheduled = True
            snap = self.grid.copy()
            def _task():
                res = self.predictor.predict(snap)
                self.probs = res
                self._predict_scheduled = False
                self._last_edit_t = 0.0
            threading.Thread(target=_task, daemon=True).start()

    def _verify(self) -> None:
        if self._verify_running:
            return
        self._verify_running = True
        self._status("Evaluating ground truth oracle…")
        snap = self.grid.copy()
        def _task():
            res = run_until_settled(snap)
            self.fate_result = res
            if self.probs is not None:
                self.score_total += 1
                if int(np.argmax(self.probs)) == res.label:
                    self.score_correct += 1
            self._verify_running = False
        threading.Thread(target=_task, daemon=True).start()

    def _fast_forward(self) -> None:
        self._status("Fast-forwarding to settled state…")
        res = run_until_settled(self.grid.copy())
        self.fate_result = res
        g = self.grid.copy()
        target = res.settle_time if res.settle_time is not None else res.generations_run
        for _ in range(target):
            g = step(g)
        self.grid = g
        self.generation = target
        self._on_board_edit()
        self._status(f"Jumped to generation {target}")

    def _zoom(self, delta: int) -> None:
        self.cell_px = max(CELL_PX_MIN, min(CELL_PX_MAX, self.cell_px + delta))

    def _resize_board(self, new_size: int) -> None:
        new_size = max(BOARD_SIZE_MIN, min(BOARD_SIZE_MAX, new_size))
        if new_size == self.board_size:
            return
        old = self.grid
        new = empty_grid(new_size)
        cr = min(old.shape[0], new_size)
        cc = min(old.shape[1], new_size)
        new[:cr, :cc] = old[:cr, :cc]
        self.grid = new
        self.board_size = new_size
        self.generation = 0
        self.fate_result = None
        self._on_board_edit()
        self._status(f"Grid resized: {new_size}×{new_size}")

    def _save_pattern(self) -> None:
        PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
        idx = len(list(PATTERNS_DIR.glob("*.cells")))
        path = PATTERNS_DIR / f"pattern_{idx:03d}.cells"
        save_cells(path, self.grid)
        self._status(f"Pattern saved: {path.name}")

    def _load_pattern(self) -> None:
        files = sorted(PATTERNS_DIR.glob("*.cells"))
        if not files:
            self._status("No saved patterns found")
            return
        path = files[-1]
        pat = load_cells(path)
        ph, pw = pat.shape
        r0 = max(0, (self.board_size - ph) // 2)
        c0 = max(0, (self.board_size - pw) // 2)
        self.grid = empty_grid(self.board_size)
        self.grid = stamp(self.grid, pat, r0, c0)
        self.generation = 0
        self.fate_result = None
        self._on_board_edit()
        self._status(f"Pattern loaded: {path.name}")

    def _status(self, msg: str, duration: float = 2.5) -> None:
        self._status_msg = msg
        self._status_t   = time.perf_counter() + duration

    # ─────────────────────────────────────────────────────────────────────────
    #  Drawing Subsystem
    # ─────────────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.screen.fill(C_CANVAS)
        self._draw_board()
        self._draw_stamp_ghost()
        self._draw_panel()
        self._draw_toast()

    def _draw_board(self) -> None:
        surf = self.screen
        cp   = self.cell_px
        bs   = self.board_size
        bp   = self.board_px

        # Fill background dead space
        pygame.draw.rect(surf, C_DEAD, (0, 0, bp, bp))

        # Live cells as smooth crisp white tiles
        radius = 2 if cp >= 10 else 0
        for r in range(bs):
            for c in range(bs):
                if self.grid[r, c]:
                    rect = (c * cp + 1, r * cp + 1, cp - 1, cp - 1)
                    if radius > 0:
                        pygame.draw.rect(surf, C_ALIVE, rect, border_radius=radius)
                    else:
                        pygame.draw.rect(surf, C_ALIVE, rect)

        # Subtle clean grid lines
        if self.show_grid and cp >= 6:
            for i in range(bs + 1):
                pygame.draw.line(surf, C_GRID_LINE, (i * cp, 0), (i * cp, bp))
                pygame.draw.line(surf, C_GRID_LINE, (0, i * cp), (bp, i * cp))

    def _draw_stamp_ghost(self) -> None:
        if self._stamp_pat is None:
            return
        mx, my = pygame.mouse.get_pos()
        bp = self.board_px
        if mx >= bp or my >= bp:
            return

        cp = self.cell_px
        bs = self.board_size
        cr, cc = my // cp, mx // cp

        ghost_surf = pygame.Surface((bp, bp), pygame.SRCALPHA)
        ph, pw = self._stamp_pat.shape
        for r in range(ph):
            for c in range(pw):
                if self._stamp_pat[r, c]:
                    dr = (cr + r) % bs
                    dc = (cc + c) % bs
                    pygame.draw.rect(ghost_surf, C_STAMP_GHOST,
                                     (dc * cp + 1, dr * cp + 1, cp - 1, cp - 1),
                                     border_radius=1 if cp >= 8 else 0)
        self.screen.blit(ghost_surf, (0, 0))

    def _draw_panel(self) -> None:
        surf   = self.screen
        win_w  = surf.get_width()
        win_h  = surf.get_height()
        px     = self.panel_x
        pw     = PANEL_W

        # Panel canvas & divider line
        pygame.draw.rect(surf, C_PANEL_BG, (px, 0, pw, win_h))
        pygame.draw.line(surf, C_PANEL_EDGE, (px, 0), (px, win_h), 1)

        pad = 16
        card_w = pw - (pad * 2)
        y = 16

        # ── Header Bar ────────────────────────────────────────────────────────
        title = self.font_title.render("LIFE ORACLE", True, C_TEXT_WHITE)
        surf.blit(title, (px + pad, y))

        # Status Pill (RUNNING / PAUSED)
        if self.running:
            pill_txt = "RUNNING"
            pill_bg, pill_fg, pill_ed = C_RUN_BG, C_RUN_TEXT, C_RUN_EDGE
        else:
            pill_txt = "PAUSED"
            pill_bg, pill_fg, pill_ed = C_PAUSE_BG, C_PAUSE_TEXT, C_PAUSE_EDGE

        pill_surf = self.font_mono_sm.render(pill_txt, True, pill_fg)
        pill_w = pill_surf.get_width() + 14
        pill_h = 20
        pill_x = px + pw - pad - pill_w
        pygame.draw.rect(surf, pill_bg, (pill_x, y, pill_w, pill_h), border_radius=10)
        pygame.draw.rect(surf, pill_ed, (pill_x, y, pill_w, pill_h), 1, border_radius=10)
        surf.blit(pill_surf, (pill_x + 7, y + 3))

        y += 28

        # ── Card 1: Telemetry & State ─────────────────────────────────────────
        card1_h = 68
        self._draw_card_box(surf, px + pad, y, card_w, card1_h)

        cy = y + 10
        cx = px + pad + 12

        # Generation & Population counters
        pop = int(self.grid.sum())
        self._draw_stat_line(surf, cx, cy, card_w - 24, "Generation", f"{self.generation:,}")
        cy += 24
        self._draw_stat_line(surf, cx, cy, card_w - 24, "Population", f"{pop:,}")

        y += card1_h + 10

        # ── Card 2: Controls & Grid Settings ──────────────────────────────────
        card2_h = 88
        self._draw_card_box(surf, px + pad, y, card_w, card2_h)
        cy = y + 10

        self._draw_chip_line(surf, cx, cy, card_w - 24, "Speed", f"{self.speed:.1f} g/s", "↑/↓")
        cy += 24
        self._draw_chip_line(surf, cx, cy, card_w - 24, "Grid Size", f"{self.board_size}×{self.board_size}", "</>")
        cy += 24
        self._draw_chip_line(surf, cx, cy, card_w - 24, "Zoom", f"{self.cell_px}px", "[/]")

        y += card2_h + 10

        # ── Card 3: Neural Prediction ─────────────────────────────────────────
        card3_h = 138
        self._draw_card_box(surf, px + pad, y, card_w, card3_h)
        cy = y + 10

        # Title + toggle hint
        pred_title = self.font_card_h.render("PREDICTION", True, C_TEXT_WHITE)
        surf.blit(pred_title, (cx, cy))
        p_hint = self.font_key.render("P", True, C_TEXT_MUTED)
        self._draw_badge(surf, px + pad + card_w - 28, cy, 18, 16, p_hint)
        cy += 22

        if self.predictor is None:
            self._render_text(surf, "Model not trained yet", self.font_small, cx, cy + 8, C_TEXT_MUTED)
            self._render_text(surf, "Run: python -m life.train", self.font_mono_sm, cx, cy + 28, C_TEXT_FAINT)
        elif self.probs is None:
            self._render_text(surf, "Paint cells to generate forecast…", self.font_small, cx, cy + 12, C_TEXT_MUTED)
        else:
            top_idx = int(np.argmax(self.probs))
            bar_w = card_w - 24
            for i, name in enumerate(CLASS_NAMES):
                val = float(self.probs[i])
                is_top = (i == top_idx)
                cy = self._draw_progress_bar(
                    surf, cx, cy, bar_w, 17, val, C_FATE[i],
                    name.capitalize(), is_top
                )

        y += card3_h + 10

        # ── Card 4: Ground Truth Verify ───────────────────────────────────────
        card4_h = 92
        self._draw_card_box(surf, px + pad, y, card_w, card4_h)
        cy = y + 10

        v_title = self.font_card_h.render("GROUND TRUTH", True, C_TEXT_WHITE)
        surf.blit(v_title, (cx, cy))
        v_key = self.font_key.render("V", True, C_TEXT_MUTED)
        self._draw_badge(surf, px + pad + card_w - 28, cy, 18, 16, v_key)
        cy += 22

        if self._verify_running:
            self._render_text(surf, "Simulating cycle detection…", self.font_small, cx, cy + 8, C_TEXT_MUTED)
        elif self.fate_result is None:
            self._render_text(surf, "Press [V] to run cycle oracle", self.font_small, cx, cy + 8, C_TEXT_MUTED)
            if self.score_total > 0:
                acc = self.score_correct / self.score_total
                self._render_text(surf, f"Session Score: {self.score_correct}/{self.score_total} ({acc:.0%})",
                                  self.font_mono_sm, cx, cy + 28, C_TEXT_MUTED)
        else:
            fr = self.fate_result
            fc = C_FATE[fr.label]

            # Fate badge
            fate_lbl = self.font_body.render(fr.label_name, True, fc)
            surf.blit(fate_lbl, (cx, cy))

            # Match indicator
            if self.probs is not None:
                match = (int(np.argmax(self.probs)) == fr.label)
                m_txt = "MATCH" if match else "MISMATCH"
                m_fg  = C_MATCH_TEXT if match else C_MISMATCH_TEXT
                m_bg  = C_MATCH_BG if match else C_MISMATCH_BG
                m_surf = self.font_mono_sm.render(m_txt, True, m_fg)
                mw = m_surf.get_width() + 10
                pygame.draw.rect(surf, m_bg, (px + pad + card_w - 12 - mw, cy, mw, 18), border_radius=4)
                surf.blit(m_surf, (px + pad + card_w - 12 - mw + 5, cy + 2))

            cy += 22
            # Cycle details
            details = []
            if fr.period is not None:
                details.append(f"Period: {fr.period}")
            if fr.settle_time is not None:
                details.append(f"Settled: gen {fr.settle_time}")
            details.append(f"End Pop: {fr.final_population}")
            self._render_text(surf, " · ".join(details), self.font_mono_sm, cx, cy, C_TEXT_MUTED)

            if self.score_total > 0:
                acc = self.score_correct / self.score_total
                cy += 18
                self._render_text(surf, f"Accuracy: {self.score_correct}/{self.score_total} ({acc:.0%})",
                                  self.font_small, cx, cy, C_TEXT_MUTED)

        y += card4_h + 10

        # ── Card 5: Pattern Stamps ────────────────────────────────────────────
        rem_h = win_h - y - 16
        if rem_h >= 110:
            card5_h = min(rem_h, 160)
            self._draw_card_box(surf, px + pad, y, card_w, card5_h)
            cy = y + 10

            s_title = self.font_card_h.render("PATTERN STAMPS", True, C_TEXT_WHITE)
            surf.blit(s_title, (cx, cy))
            if self._stamp_name:
                rot_str = f"{self._stamp_rot * 90}° (Q/E)"
                r_surf = self.font_mono_sm.render(rot_str, True, C_RUN_TEXT)
                surf.blit(r_surf, (px + pad + card_w - 12 - r_surf.get_width(), cy))
            cy += 22

            # 2-column pattern stamp grid
            col_w = (card_w - 28) // 2
            for i, name in enumerate(PATTERN_ORDER[:8]):
                col = i % 2
                row = i // 2
                item_x = cx + (col * (col_w + 8))
                item_y = cy + (row * 24)
                if item_y + 20 > y + card5_h:
                    break

                is_active = (name == self._stamp_name)
                bg_col = C_CARD_ACTIVE if is_active else C_CHIP_BG
                fg_col = C_TEXT_WHITE if is_active else C_TEXT_MUTED
                border_col = C_RUN_EDGE if is_active else C_CHIP_EDGE

                pygame.draw.rect(surf, bg_col, (item_x, item_y, col_w, 20), border_radius=4)
                pygame.draw.rect(surf, border_col, (item_x, item_y, col_w, 20), 1, border_radius=4)

                k_surf = self.font_key.render(f"{i+1}", True, C_TEXT_WHITE if is_active else C_TEXT_FAINT)
                surf.blit(k_surf, (item_x + 5, item_y + 3))

                n_surf = self.font_small.render(name, True, fg_col)
                surf.blit(n_surf, (item_x + 18, item_y + 3))

    # ─────────────────────────────────────────────────────────────────────────
    #  Drawing Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_card_box(self, surf: pygame.Surface, x: int, y: int, w: int, h: int) -> None:
        pygame.draw.rect(surf, C_CARD_BG, (x, y, w, h), border_radius=6)
        pygame.draw.rect(surf, C_CARD_EDGE, (x, y, w, h), 1, border_radius=6)

    def _draw_stat_line(self, surf: pygame.Surface, x: int, y: int, w: int, label: str, value: str) -> None:
        lbl_surf = self.font_body.render(label, True, C_TEXT_MUTED)
        val_surf = self.font_mono.render(value, True, C_TEXT_WHITE)
        surf.blit(lbl_surf, (x, y))
        surf.blit(val_surf, (x + w - val_surf.get_width(), y))

    def _draw_chip_line(self, surf: pygame.Surface, x: int, y: int, w: int, label: str, value: str, key_hint: str) -> None:
        lbl_surf = self.font_body.render(label, True, C_TEXT_MUTED)
        val_surf = self.font_mono_sm.render(value, True, C_TEXT_WHITE)
        surf.blit(lbl_surf, (x, y))

        vx = x + w - val_surf.get_width()
        surf.blit(val_surf, (vx, y + 1))

        # Shortcut badge
        hint_surf = self.font_key.render(key_hint, True, C_TEXT_FAINT)
        hw = hint_surf.get_width() + 8
        pygame.draw.rect(surf, C_CHIP_BG, (vx - hw - 8, y, hw, 18), border_radius=3)
        pygame.draw.rect(surf, C_CHIP_EDGE, (vx - hw - 8, y, hw, 18), 1, border_radius=3)
        surf.blit(hint_surf, (vx - hw - 4, y + 2))

    def _draw_progress_bar(
        self, surf: pygame.Surface, x: int, y: int, w: int, h: int,
        frac: float, color: tuple, label: str, is_top: bool
    ) -> int:
        track_col = (36, 40, 50)
        pygame.draw.rect(surf, track_col, (x, y, w, h), border_radius=4)

        fill_w = max(2, int(w * max(0.0, min(1.0, frac))))
        pygame.draw.rect(surf, color, (x, y, fill_w, h), border_radius=4)

        # Labels
        name_col = C_TEXT_WHITE if is_top else C_TEXT_MAIN
        lbl = self.font_small.render(label, True, name_col)
        pct = self.font_mono_sm.render(f"{frac:4.0%}", True, C_TEXT_WHITE if is_top else C_TEXT_MUTED)

        surf.blit(lbl, (x + 6, y + 1))
        surf.blit(pct, (x + w - pct.get_width() - 6, y + 1))

        return y + h + 5

    def _draw_badge(self, surf: pygame.Surface, x: int, y: int, w: int, h: int, text_surf: pygame.Surface) -> None:
        pygame.draw.rect(surf, C_CHIP_BG, (x, y, w, h), border_radius=3)
        pygame.draw.rect(surf, C_CHIP_EDGE, (x, y, w, h), 1, border_radius=3)
        tx = x + (w - text_surf.get_width()) // 2
        ty = y + (h - text_surf.get_height()) // 2
        surf.blit(text_surf, (tx, ty))

    def _render_text(self, surf: pygame.Surface, text: str, font: pygame.font.Font, x: int, y: int, color: tuple) -> None:
        surf.blit(font.render(text, True, color), (x, y))

    def _draw_toast(self) -> None:
        if not self._status_msg or time.perf_counter() > self._status_t:
            return
        surf = self.screen
        txt_surf = self.font_body.render(self._status_msg, True, C_TEXT_WHITE)
        tw, th = txt_surf.get_width() + 24, txt_surf.get_height() + 14

        bx = 16
        by = self.board_px - th - 16
        if by < 16:
            by = 16

        toast_bg = (24, 27, 34)
        toast_edge = (50, 56, 70)
        pygame.draw.rect(surf, toast_bg, (bx, by, tw, th), border_radius=6)
        pygame.draw.rect(surf, toast_edge, (bx, by, tw, th), 1, border_radius=6)
        surf.blit(txt_surf, (bx + 12, by + 7))

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
