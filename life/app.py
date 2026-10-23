"""
Life Oracle – modern minimal desktop application.
Light monochrome minimal design with dynamic pan/zoom infinite-feeling canvas.
Entry point: python -m life.app

Layout (resizable)
------------------
+-----------------------------------+-------------------+
|                                   |                   |
|     INFINITE / SCALABLE CANVAS    |    SIDE PANEL     |
|      (dynamic pan & zoom)         | (cards & metrics) |
|                                   |                   |
+-----------------------------------+-------------------+
"""

from __future__ import annotations

import math
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
#  Modern Minimalist Light Palette (Scandinavian / Swiss Minimal)
# ─────────────────────────────────────────────────────────────────────────────
C_CANVAS_BG     = (246, 247, 249)   # clean warm paper canvas
C_GRID_LINE     = (226, 230, 236)   # subtle hairline grid lines
C_BOARD_BG      = (255, 255, 255)   # primary board card background
C_BOARD_BORDER  = (205, 212, 222)   # primary board outline border
C_GHOST_BOARD   = (250, 251, 253)   # repeating wrap tile background

C_ALIVE         = (20,  24,  33)    # deep obsidian black live cells (crisp & bold)
C_ALIVE_GHOST   = (165, 174, 188)   # muted charcoal for toroidal wrap ghost cells
C_DEAD          = (255, 255, 255)   # dead cells match board card

# Side Panel & Cards
C_PANEL_BG      = (255, 255, 255)   # crisp white sidebar
C_PANEL_BORDER  = (226, 232, 240)   # divider edge between canvas and panel
C_CARD_BG       = (248, 250, 252)   # clean slate/ice card background
C_CARD_BORDER   = (226, 232, 240)   # subtle card outline
C_CARD_ACTIVE   = (238, 242, 248)   # active stamp background

# High-Contrast Light Typography
C_TEXT_PRIMARY   = (15,  23,  42)   # deep ink black for titles/values (AAA contrast)
C_TEXT_SECONDARY = (100, 116, 139)  # clean slate gray for labels
C_TEXT_FAINT     = (148, 163, 184)  # subdued hints / brackets

# Shortcut Chips
C_CHIP_BG       = (241, 245, 249)   # light gray key badge
C_CHIP_BORDER   = (203, 213, 225)   # key badge outline
C_CHIP_TEXT     = (51,  65,  85)

# Translucent Stamp Ghost
C_STAMP_GHOST   = (30,  41,  59, 90)

# Status Pill Badges
C_RUN_BG        = (220, 252, 231)
C_RUN_TEXT      = (22,  101, 52)
C_RUN_BORDER    = (134, 239, 172)

C_PAUSE_BG      = (241, 245, 249)
C_PAUSE_TEXT    = (71,  85,  105)
C_PAUSE_BORDER  = (203, 213, 225)

C_MATCH_BG      = (220, 252, 231)
C_MATCH_TEXT    = (22,  101, 52)
C_MISMATCH_BG   = (254, 226, 226)
C_MISMATCH_TEXT = (185, 28,  28)

# Fate Colors in Light Theme (Refined Saturated Hues)
C_FATE = {
    0: (239, 68,  68),    # EXTINCT    – vibrant crimson
    1: (37,  99,  235),   # STABLE     – royal sapphire
    2: (217, 119, 6),     # OSCILLATING – warm amber gold
    3: (124, 58,  237),   # UNSETTLED  – electric amethyst
}

PANEL_W        = 320
MIN_WIN_W      = 720
MIN_WIN_H      = 540
INIT_WIN_W     = _CFG_BOARD_SIZE * _CFG_CELL_PX + PANEL_W + 60
INIT_WIN_H     = max(_CFG_BOARD_SIZE * _CFG_CELL_PX + 60, 680)

ZOOM_MIN, ZOOM_MAX             = 3.0, 80.0
BOARD_SIZE_MIN, BOARD_SIZE_MAX = 8, 128


# ─────────────────────────────────────────────────────────────────────────────
#  Font Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(family_names: str, size: int, bold: bool = False) -> pygame.font.Font:
    """Attempt to load clean typography from system fonts."""
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

        # Clean typography
        sans = "adwaitasans, notosans, dejavusans, freesans, helvetica, arial, sans-serif"
        mono = "adwaitamono, hack, freemono, dejavusansmono, monospace"

        self.font_title   = _load_font(sans, 16, bold=True)
        self.font_card_h  = _load_font(sans, 13, bold=True)
        self.font_body    = _load_font(sans, 13, bold=False)
        self.font_small   = _load_font(sans, 11, bold=False)
        self.font_mono    = _load_font(mono, 13, bold=False)
        self.font_mono_sm = _load_font(mono, 11, bold=False)
        self.font_key     = _load_font(mono, 11, bold=True)

        # ── Grid & Canvas State ──────────────────────────────────────────
        self.board_size: int = _CFG_BOARD_SIZE
        self.zoom: float     = float(_CFG_CELL_PX)   # pixel width of each cell
        self.cam_x: float    = 30.0                  # camera top-left x in screen pixels
        self.cam_y: float    = 30.0                  # camera top-left y in screen pixels

        # Center camera initially
        self._center_camera()

        # ── Pan & Drag State ─────────────────────────────────────────────
        self._panning: bool = False
        self._pan_start_mouse: tuple[int, int] = (0, 0)
        self._pan_start_cam:   tuple[float, float] = (0.0, 0.0)

        # ── Simulation State ─────────────────────────────────────────────
        self.grid: Grid = empty_grid(self.board_size)
        self.generation: int = 0
        self.running: bool   = False
        self.speed: float    = DEFAULT_SPEED
        self._last_step_t: float = time.perf_counter()

        # ── Drawing State ────────────────────────────────────────────────
        self.show_grid: bool = True
        self._painting: Optional[int] = None
        self._paint_cell: Optional[tuple[int, int]] = None

        # ── Stamp State ──────────────────────────────────────────────────
        self._stamp_name: Optional[str]  = None
        self._stamp_pat:  Optional[Grid] = None
        self._stamp_rot:  int = 0

        # ── Prediction State ─────────────────────────────────────────────
        self.live_predict: bool = True
        self.probs: Optional[np.ndarray] = None
        self._last_edit_t: float = 0.0
        self._predict_scheduled: bool = False

        # ── Oracle / Verify State ────────────────────────────────────────
        self.fate_result: Optional[FateResult] = None
        self._verify_running: bool = False

        # ── Scoreboard ───────────────────────────────────────────────────
        self.score_total:   int = 0
        self.score_correct: int = 0

        # ── Floating Toast Message ───────────────────────────────────────
        self._status_msg: str   = ""
        self._status_t:   float = 0.0

        # ── Predictor ────────────────────────────────────────────────────
        self.predictor = None
        if _PREDICTOR_AVAILABLE:
            try:
                from .predictor import Predictor
                self.predictor = Predictor()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Geometry & Coordinates
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def panel_x(self) -> int:
        return self.screen.get_width() - PANEL_W

    @property
    def canvas_w(self) -> int:
        return self.panel_x

    @property
    def canvas_h(self) -> int:
        return self.screen.get_height()

    def _center_camera(self) -> None:
        """Center the board cleanly within the canvas viewport."""
        board_px = self.board_size * self.zoom
        cw = max(100, self.canvas_w)
        ch = max(100, self.canvas_h)
        self.cam_x = (cw - board_px) / 2.0
        self.cam_y = (ch - board_px) / 2.0

    def _fit_board(self) -> None:
        """Adjust zoom and center camera to fit the entire board with margin."""
        margin = 48
        cw = max(100, self.canvas_w - margin)
        ch = max(100, self.canvas_h - margin)
        best_zoom = min(cw / self.board_size, ch / self.board_size)
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, round(best_zoom, 1)))
        self._center_camera()
        self._status(f"Fit Board ({int(self.zoom)}px/cell)")

    def _screen_to_cell(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        """Map screen pixel (x, y) to global board cell (row, col) with toroidal wrapping."""
        x, y = pos
        if x >= self.panel_x:
            return None
        col = int(math.floor((x - self.cam_x) / self.zoom))
        row = int(math.floor((y - self.cam_y) / self.zoom))
        # Toroidal wrap
        return (row % self.board_size, col % self.board_size)

    def _zoom_at_mouse(self, factor: float, mouse_pos: tuple[int, int]) -> None:
        """Zoom smoothly anchored to the mouse position."""
        mx, my = mouse_pos
        if mx >= self.panel_x:
            # If cursor is over panel, zoom relative to canvas center
            mx, my = self.canvas_w / 2, self.canvas_h / 2

        old_zoom = self.zoom
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        # World coordinate under cursor before zoom
        world_x = (mx - self.cam_x) / old_zoom
        world_y = (my - self.cam_y) / old_zoom

        self.zoom = new_zoom
        # Adjust camera so world coordinate remains under cursor
        self.cam_x = mx - world_x * new_zoom
        self.cam_y = my - world_y * new_zoom

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
        keys = pygame.key.get_pressed()
        space_down = keys[pygame.K_SPACE]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()

            elif event.type == pygame.KEYDOWN:
                self._on_key(event.key, event.mod)

            elif event.type == pygame.MOUSEWHEEL:
                # Modern smooth mousewheel zoom anchored at mouse cursor
                mpos = pygame.mouse.get_pos()
                if event.y > 0:
                    self._zoom_at_mouse(1.15, mpos)
                elif event.y < 0:
                    self._zoom_at_mouse(1.0 / 1.15, mpos)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2 or (event.button == 1 and space_down):
                    # Pan trigger (middle click OR space+left click)
                    self._panning = True
                    self._pan_start_mouse = event.pos
                    self._pan_start_cam   = (self.cam_x, self.cam_y)
                elif event.button in (1, 3):
                    self._on_mouse_down(event.pos, event.button)
                elif event.button == 4:   # Scroll up fallback
                    self._zoom_at_mouse(1.15, event.pos)
                elif event.button == 5:   # Scroll down fallback
                    self._zoom_at_mouse(1.0 / 1.15, event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 2):
                    self._panning = False
                if event.button in (1, 3):
                    self._painting   = None
                    self._paint_cell = None

            elif event.type == pygame.MOUSEMOTION:
                if self._panning:
                    dx = event.pos[0] - self._pan_start_mouse[0]
                    dy = event.pos[1] - self._pan_start_mouse[1]
                    self.cam_x = self._pan_start_cam[0] + dx
                    self.cam_y = self._pan_start_cam[1] + dy
                elif self._painting is not None:
                    self._on_mouse_drag(event.pos)

            elif event.type == pygame.VIDEORESIZE:
                pass

    def _on_mouse_down(self, pos: tuple[int, int], button: int) -> None:
        cell = self._screen_to_cell(pos)
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
            new_val = 1 - int(self.grid[r, c])
            self._painting   = new_val
            self._paint_cell = (r, c)
            self.grid[r, c]  = new_val
        else:
            self._painting   = 0
            self._paint_cell = (r, c)
            self.grid[r, c]  = 0

        self.generation = 0
        self._on_board_edit()

    def _on_mouse_drag(self, pos: tuple[int, int]) -> None:
        if self._stamp_name:
            return
        cell = self._screen_to_cell(pos)
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
        # Space pauses/plays if not dragging canvas
        if key == pygame.K_SPACE and not self._panning:
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
            self._status("Randomized Soup")

        elif key == pygame.K_g:
            self.show_grid = not self.show_grid
            self._status(f"Grid Lines: {'ON' if self.show_grid else 'OFF'}")

        elif key == pygame.K_p:
            self.live_predict = not self.live_predict
            self._status(f"Live Prediction: {'ON' if self.live_predict else 'OFF'}")

        elif key == pygame.K_v:
            self._verify()

        elif key == pygame.K_f:
            self._fast_forward()

        elif key in (pygame.K_0, pygame.K_h):
            self._fit_board()

        elif key == pygame.K_LEFTBRACKET:
            self._zoom_at_mouse(1.0 / 1.2, (self.canvas_w / 2, self.canvas_h / 2))

        elif key == pygame.K_RIGHTBRACKET:
            self._zoom_at_mouse(1.2, (self.canvas_w / 2, self.canvas_h / 2))

        elif key == pygame.K_COMMA:
            self._resize_board(self.board_size - 4)

        elif key == pygame.K_PERIOD:
            self._resize_board(self.board_size + 4)

        elif key == pygame.K_s:
            self._save_pattern()

        elif key == pygame.K_l:
            self._load_pattern()

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
                self._status("Stamp Cleared")
            else:
                self._quit()

        elif pygame.K_1 <= key <= pygame.K_9:
            idx = key - pygame.K_1
            if idx < len(PATTERN_ORDER):
                name = PATTERN_ORDER[idx]
                self._stamp_name = name
                self._stamp_rot  = 0
                self._stamp_pat  = get_pattern(name)
                self._status(f"Stamp Selected: {name}")

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
        self._status("Simulating ground truth oracle…")
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
        self._status("Fast-forwarding to settled cycle…")
        res = run_until_settled(self.grid.copy())
        self.fate_result = res
        g = self.grid.copy()
        target = res.settle_time if res.settle_time is not None else res.generations_run
        for _ in range(target):
            g = step(g)
        self.grid = g
        self.generation = target
        self._on_board_edit()
        self._status(f"Jumped to gen {target}")

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
        self._status(f"Saved: {path.name}")

    def _load_pattern(self) -> None:
        files = sorted(PATTERNS_DIR.glob("*.cells"))
        if not files:
            self._status("No saved patterns")
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
        self._status(f"Loaded: {path.name}")

    def _status(self, msg: str, duration: float = 2.5) -> None:
        self._status_msg = msg
        self._status_t   = time.perf_counter() + duration

    # ─────────────────────────────────────────────────────────────────────────
    #  Drawing Subsystem (Dynamic Infinite Canvas)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.screen.fill(C_CANVAS_BG)
        self._draw_infinite_canvas()
        self._draw_stamp_ghost()
        self._draw_panel()
        self._draw_canvas_overlay()
        self._draw_toast()

    def _draw_infinite_canvas(self) -> None:
        surf = self.screen
        cw   = self.canvas_w
        ch   = self.canvas_h
        bs   = self.board_size
        z    = self.zoom

        board_px = bs * z

        # Clip rendering strictly to canvas area to avoid drawing under panel
        surf.set_clip(pygame.Rect(0, 0, cw, ch))

        # ── 1. Draw Toroidal Wrap Backgrounds (Ghost Tiles around Main) ────────
        # Render a 3x3 tiled universe (-1, 0, 1) so toroidal wrap looks seamless!
        for dr_wrap in (-1, 0, 1):
            for dc_wrap in (-1, 0, 1):
                tx = self.cam_x + (dc_wrap * board_px)
                ty = self.cam_y + (dr_wrap * board_px)

                # Culling: skip if tile is offscreen
                if tx + board_px < 0 or tx > cw or ty + board_px < 0 or ty > ch:
                    continue

                rect = pygame.Rect(int(tx), int(ty), int(board_px), int(board_px))
                if dr_wrap == 0 and dc_wrap == 0:
                    # Primary main board: crisp bright white card with subtle outline
                    pygame.draw.rect(surf, C_BOARD_BG, rect)
                    pygame.draw.rect(surf, C_BOARD_BORDER, rect, 1)
                else:
                    # Ghost wrap tile: faint canvas background with dashed / hairline border
                    pygame.draw.rect(surf, C_GHOST_BOARD, rect)
                    pygame.draw.rect(surf, C_GRID_LINE, rect, 1)

        # ── 2. Draw Subtle Grid Lines ─────────────────────────────────────────
        if self.show_grid and z >= 5.0:
            # We draw grid lines for visible areas
            start_c = int(math.floor((0 - self.cam_x) / z)) - 1
            end_c   = int(math.ceil((cw - self.cam_x) / z)) + 1
            start_r = int(math.floor((0 - self.cam_y) / z)) - 1
            end_r   = int(math.ceil((ch - self.cam_y) / z)) + 1

            for c in range(start_c, end_c + 1):
                gx = int(self.cam_x + c * z)
                pygame.draw.line(surf, C_GRID_LINE, (gx, 0), (gx, ch))

            for r in range(start_r, end_r + 1):
                gy = int(self.cam_y + r * z)
                pygame.draw.line(surf, C_GRID_LINE, (0, gy), (cw, gy))

        # ── 3. Draw Living Cells (Main + Toroidal Ghost Wrap) ──────────────────
        radius = 2 if z >= 12 else 0

        for dr_wrap in (-1, 0, 1):
            for dc_wrap in (-1, 0, 1):
                tx = self.cam_x + (dc_wrap * board_px)
                ty = self.cam_y + (dr_wrap * board_px)

                # Frustum cull
                if tx + board_px < 0 or tx > cw or ty + board_px < 0 or ty > ch:
                    continue

                is_main = (dr_wrap == 0 and dc_wrap == 0)
                cell_col = C_ALIVE if is_main else C_ALIVE_GHOST

                for r in range(bs):
                    cy = ty + r * z
                    if cy + z < 0 or cy > ch:
                        continue
                    for c in range(bs):
                        if self.grid[r, c]:
                            cx = tx + c * z
                            if cx + z < 0 or cx > cw:
                                continue
                            rect = pygame.Rect(int(cx + 1), int(cy + 1), max(1, int(z - 1)), max(1, int(z - 1)))
                            if radius > 0 and is_main:
                                pygame.draw.rect(surf, cell_col, rect, border_radius=radius)
                            else:
                                pygame.draw.rect(surf, cell_col, rect)

        # Unset clipping
        surf.set_clip(None)

    def _draw_stamp_ghost(self) -> None:
        if self._stamp_pat is None:
            return
        mx, my = pygame.mouse.get_pos()
        if mx >= self.panel_x:
            return

        cell = self._screen_to_cell((mx, my))
        if cell is None:
            return

        surf = self.screen
        cr, cc = cell
        ph, pw = self._stamp_pat.shape
        z = self.zoom

        ghost_surf = pygame.Surface((self.canvas_w, self.canvas_h), pygame.SRCALPHA)
        for r in range(ph):
            for c in range(pw):
                if self._stamp_pat[r, c]:
                    # Target board cell
                    tr = (cr + r) % self.board_size
                    tc = (cc + c) % self.board_size
                    sx = int(self.cam_x + tc * z)
                    sy = int(self.cam_y + tr * z)
                    if 0 <= sx < self.canvas_w and 0 <= sy < self.canvas_h:
                        pygame.draw.rect(ghost_surf, C_STAMP_GHOST,
                                         (sx + 1, sy + 1, max(1, int(z - 1)), max(1, int(z - 1))),
                                         border_radius=2 if z >= 12 else 0)
        surf.blit(ghost_surf, (0, 0))

    def _draw_canvas_overlay(self) -> None:
        """Draw small HUD badges on the canvas (zoom level, center button hint)."""
        surf = self.screen
        hud_txt = f"{int(self.zoom)}px/cell · Space+Drag to Pan · [0] to Center"
        h_surf = self.font_mono_sm.render(hud_txt, True, C_TEXT_SECONDARY)
        hw, hh = h_surf.get_width() + 16, h_surf.get_height() + 8
        hx, hy = 14, 14
        pygame.draw.rect(surf, (255, 255, 255, 220), (hx, hy, hw, hh), border_radius=6)
        pygame.draw.rect(surf, C_CARD_BORDER, (hx, hy, hw, hh), 1, border_radius=6)
        surf.blit(h_surf, (hx + 8, hy + 4))

    def _draw_panel(self) -> None:
        surf   = self.screen
        win_w  = surf.get_width()
        win_h  = surf.get_height()
        px     = self.panel_x
        pw     = PANEL_W

        # Side panel background & left border
        pygame.draw.rect(surf, C_PANEL_BG, (px, 0, pw, win_h))
        pygame.draw.line(surf, C_PANEL_BORDER, (px, 0), (px, win_h), 1)

        pad = 16
        card_w = pw - (pad * 2)
        y = 16

        # ── Header Bar ────────────────────────────────────────────────────────
        title = self.font_title.render("LIFE ORACLE", True, C_TEXT_PRIMARY)
        surf.blit(title, (px + pad, y))

        # Status Pill Badge
        if self.running:
            pill_txt = "RUNNING"
            p_bg, p_fg, p_bd = C_RUN_BG, C_RUN_TEXT, C_RUN_BORDER
        else:
            pill_txt = "PAUSED"
            p_bg, p_fg, p_bd = C_PAUSE_BG, C_PAUSE_TEXT, C_PAUSE_BORDER

        pill_surf = self.font_mono_sm.render(pill_txt, True, p_fg)
        pw_pill = pill_surf.get_width() + 14
        ph_pill = 20
        px_pill = px + pw - pad - pw_pill
        pygame.draw.rect(surf, p_bg, (px_pill, y, pw_pill, ph_pill), border_radius=10)
        pygame.draw.rect(surf, p_bd, (px_pill, y, pw_pill, ph_pill), 1, border_radius=10)
        surf.blit(pill_surf, (px_pill + 7, y + 3))

        y += 28

        # ── Card 1: Telemetry & State ─────────────────────────────────────────
        card1_h = 68
        self._draw_card_box(surf, px + pad, y, card_w, card1_h)
        cy = y + 10
        cx = px + pad + 12

        pop = int(self.grid.sum())
        self._draw_stat_line(surf, cx, cy, card_w - 24, "Generation", f"{self.generation:,}")
        cy += 24
        self._draw_stat_line(surf, cx, cy, card_w - 24, "Population", f"{pop:,}")

        y += card1_h + 10

        # ── Card 2: Interactive Controls ──────────────────────────────────────
        card2_h = 88
        self._draw_card_box(surf, px + pad, y, card_w, card2_h)
        cy = y + 10

        self._draw_chip_line(surf, cx, cy, card_w - 24, "Speed", f"{self.speed:.1f} g/s", "↑/↓")
        cy += 24
        self._draw_chip_line(surf, cx, cy, card_w - 24, "Grid Size", f"{self.board_size}×{self.board_size}", "</>")
        cy += 24
        self._draw_chip_line(surf, cx, cy, card_w - 24, "Zoom", f"{int(self.zoom)}px", "Scroll")

        y += card2_h + 10

        # ── Card 3: Neural Prediction ─────────────────────────────────────────
        card3_h = 138
        self._draw_card_box(surf, px + pad, y, card_w, card3_h)
        cy = y + 10

        p_title = self.font_card_h.render("PREDICTION", True, C_TEXT_PRIMARY)
        surf.blit(p_title, (cx, cy))
        p_badge = self.font_key.render("P", True, C_CHIP_TEXT)
        self._draw_badge(surf, px + pad + card_w - 28, cy, 18, 16, p_badge)
        cy += 22

        if self.predictor is None:
            self._render_text(surf, "Model not trained yet", self.font_small, cx, cy + 8, C_TEXT_SECONDARY)
            self._render_text(surf, "Run: python -m life.train", self.font_mono_sm, cx, cy + 28, C_TEXT_FAINT)
        elif self.probs is None:
            self._render_text(surf, "Paint cells to generate forecast…", self.font_small, cx, cy + 12, C_TEXT_SECONDARY)
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

        v_title = self.font_card_h.render("GROUND TRUTH", True, C_TEXT_PRIMARY)
        surf.blit(v_title, (cx, cy))
        v_badge = self.font_key.render("V", True, C_CHIP_TEXT)
        self._draw_badge(surf, px + pad + card_w - 28, cy, 18, 16, v_badge)
        cy += 22

        if self._verify_running:
            self._render_text(surf, "Simulating cycle detection…", self.font_small, cx, cy + 8, C_TEXT_SECONDARY)
        elif self.fate_result is None:
            self._render_text(surf, "Press [V] to run cycle oracle", self.font_small, cx, cy + 8, C_TEXT_SECONDARY)
            if self.score_total > 0:
                acc = self.score_correct / self.score_total
                self._render_text(surf, f"Session Score: {self.score_correct}/{self.score_total} ({acc:.0%})",
                                  self.font_mono_sm, cx, cy + 28, C_TEXT_SECONDARY)
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
            details = []
            if fr.period is not None:
                details.append(f"Period: {fr.period}")
            if fr.settle_time is not None:
                details.append(f"Settled: gen {fr.settle_time}")
            details.append(f"End Pop: {fr.final_population}")
            self._render_text(surf, " · ".join(details), self.font_mono_sm, cx, cy, C_TEXT_SECONDARY)

            if self.score_total > 0:
                acc = self.score_correct / self.score_total
                cy += 18
                self._render_text(surf, f"Accuracy: {self.score_correct}/{self.score_total} ({acc:.0%})",
                                  self.font_small, cx, cy, C_TEXT_SECONDARY)

        y += card4_h + 10

        # ── Card 5: Pattern Stamps ────────────────────────────────────────────
        rem_h = win_h - y - 16
        if rem_h >= 110:
            card5_h = min(rem_h, 160)
            self._draw_card_box(surf, px + pad, y, card_w, card5_h)
            cy = y + 10

            s_title = self.font_card_h.render("PATTERN STAMPS", True, C_TEXT_PRIMARY)
            surf.blit(s_title, (cx, cy))
            if self._stamp_name:
                rot_str = f"{self._stamp_rot * 90}° (Q/E)"
                r_surf = self.font_mono_sm.render(rot_str, True, C_RUN_TEXT)
                surf.blit(r_surf, (px + pad + card_w - 12 - r_surf.get_width(), cy))
            cy += 22

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
                fg_col = C_TEXT_PRIMARY if is_active else C_TEXT_SECONDARY
                border_col = C_CHIP_BORDER if not is_active else C_RUN_BORDER

                pygame.draw.rect(surf, bg_col, (item_x, item_y, col_w, 20), border_radius=4)
                pygame.draw.rect(surf, border_col, (item_x, item_y, col_w, 20), 1, border_radius=4)

                k_surf = self.font_key.render(f"{i+1}", True, C_TEXT_PRIMARY if is_active else C_TEXT_FAINT)
                surf.blit(k_surf, (item_x + 5, item_y + 3))

                n_surf = self.font_small.render(name, True, fg_col)
                surf.blit(n_surf, (item_x + 18, item_y + 3))

    # ─────────────────────────────────────────────────────────────────────────
    #  Drawing Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_card_box(self, surf: pygame.Surface, x: int, y: int, w: int, h: int) -> None:
        pygame.draw.rect(surf, C_CARD_BG, (x, y, w, h), border_radius=6)
        pygame.draw.rect(surf, C_CARD_BORDER, (x, y, w, h), 1, border_radius=6)

    def _draw_stat_line(self, surf: pygame.Surface, x: int, y: int, w: int, label: str, value: str) -> None:
        lbl_surf = self.font_body.render(label, True, C_TEXT_SECONDARY)
        val_surf = self.font_mono.render(value, True, C_TEXT_PRIMARY)
        surf.blit(lbl_surf, (x, y))
        surf.blit(val_surf, (x + w - val_surf.get_width(), y))

    def _draw_chip_line(self, surf: pygame.Surface, x: int, y: int, w: int, label: str, value: str, key_hint: str) -> None:
        lbl_surf = self.font_body.render(label, True, C_TEXT_SECONDARY)
        val_surf = self.font_mono_sm.render(value, True, C_TEXT_PRIMARY)
        surf.blit(lbl_surf, (x, y))

        vx = x + w - val_surf.get_width()
        surf.blit(val_surf, (vx, y + 1))

        hint_surf = self.font_key.render(key_hint, True, C_CHIP_TEXT)
        hw = hint_surf.get_width() + 8
        pygame.draw.rect(surf, C_CHIP_BG, (vx - hw - 8, y, hw, 18), border_radius=3)
        pygame.draw.rect(surf, C_CHIP_BORDER, (vx - hw - 8, y, hw, 18), 1, border_radius=3)
        surf.blit(hint_surf, (vx - hw - 4, y + 2))

    def _draw_progress_bar(
        self, surf: pygame.Surface, x: int, y: int, w: int, h: int,
        frac: float, color: tuple, label: str, is_top: bool
    ) -> int:
        track_col = (235, 238, 244)
        pygame.draw.rect(surf, track_col, (x, y, w, h), border_radius=4)

        fill_w = max(2, int(w * max(0.0, min(1.0, frac))))
        pygame.draw.rect(surf, color, (x, y, fill_w, h), border_radius=4)

        name_col = C_TEXT_PRIMARY if is_top else C_TEXT_SECONDARY
        lbl = self.font_small.render(label, True, name_col)
        pct = self.font_mono_sm.render(f"{frac:4.0%}", True, C_TEXT_PRIMARY if is_top else C_TEXT_SECONDARY)

        surf.blit(lbl, (x + 6, y + 1))
        surf.blit(pct, (x + w - pct.get_width() - 6, y + 1))

        return y + h + 5

    def _draw_badge(self, surf: pygame.Surface, x: int, y: int, w: int, h: int, text_surf: pygame.Surface) -> None:
        pygame.draw.rect(surf, C_CHIP_BG, (x, y, w, h), border_radius=3)
        pygame.draw.rect(surf, C_CHIP_BORDER, (x, y, w, h), 1, border_radius=3)
        tx = x + (w - text_surf.get_width()) // 2
        ty = y + (h - text_surf.get_height()) // 2
        surf.blit(text_surf, (tx, ty))

    def _render_text(self, surf: pygame.Surface, text: str, font: pygame.font.Font, x: int, y: int, color: tuple) -> None:
        surf.blit(font.render(text, True, color), (x, y))

    def _draw_toast(self) -> None:
        if not self._status_msg or time.perf_counter() > self._status_t:
            return
        surf = self.screen
        txt_surf = self.font_body.render(self._status_msg, True, C_TEXT_PRIMARY)
        tw, th = txt_surf.get_width() + 24, txt_surf.get_height() + 14

        bx = 16
        by = self.canvas_h - th - 16
        if by < 16:
            by = 16

        toast_bg = (255, 255, 255)
        toast_edge = (203, 213, 225)
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
