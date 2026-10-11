"""pygame renderer for the client: world view, server 'ghost' overlay, HUD."""
from __future__ import annotations

import numpy as np
import pygame

from . import config as C
from . import state as S

BG = (12, 12, 24)
SHIP_COL = [(80, 200, 255), (255, 150, 80)]
GHOST_COL = [(40, 100, 128), (128, 75, 40)]
INV_COL = [(180, 255, 120), (255, 230, 100), (255, 120, 160)]
BULLET_COL = (255, 255, 255)
HUD_COL = (200, 200, 220)


class Renderer:
    def __init__(self, size: int, title: str):
        pygame.init()
        self.size = size
        self.screen = pygame.display.set_mode((size, size + 110))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("monospace", 15)

    def px(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.size), int(y * self.size)

    def ship_poly(self, x: float, y: float, r: float = C.SHIP_RADIUS):
        cx, cy = float(x) * self.size, float(y) * self.size
        R = float(r) * self.size
        return [(cx, cy - R), (cx - R, cy + R * 0.9), (cx, cy + R * 0.4), (cx + R, cy + R * 0.9)]

    def draw(self, state: np.ndarray, render_pos: np.ndarray, ghost: np.ndarray | None, own: int, hud_lines: list[str], show_ghost: bool,
             frame: np.ndarray | None = None):
        """`frame` (H, W, 3) uint8 from the neural renderer replaces the vector drawing of the world."""
        scr = self.screen
        scr.fill(BG)
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            scr.blit(pygame.transform.scale(surf, (self.size, self.size)), (0, 0))
            if show_ghost and ghost is not None:
                gsh = S.ships(ghost[None, :])[0]
                for p in range(C.N_PLAYERS):
                    pygame.draw.polygon(scr, GHOST_COL[p], self.ship_poly(gsh[p, 0], gsh[p, 1]), width=1)
            pygame.draw.polygon(scr, (255, 255, 255), self.ship_poly(render_pos[own, 0], render_pos[own, 1]), width=2)
            self.hud(hud_lines)
            return
        pygame.draw.line(scr, (40, 40, 70), (0, int(C.SHIP_Y_MIN * self.size)), (self.size, int(C.SHIP_Y_MIN * self.size)))
        s = state[None, :]
        # invaders
        ipos = S.invader_positions(S.formation(s))[0]
        alive = S.invaders(s)[0]
        w, h = C.INV_HALF_W * self.size, C.INV_HALF_H * self.size
        for i in range(C.N_INVADERS):
            if alive[i] > 0.5:
                x, y = float(ipos[i, 0]) * self.size, float(ipos[i, 1]) * self.size
                pygame.draw.rect(scr, INV_COL[i // C.INV_COLS], pygame.Rect(x - w, y - h, 2 * w, 2 * h), border_radius=3)
        # bullets
        for p in range(C.N_PLAYERS):
            for b in range(C.N_BULLETS):
                bx, by, act = S.bullets(s)[0, p, b]
                if act > 0.5:
                    x, y = float(bx) * self.size, float(by) * self.size
                    pygame.draw.rect(scr, BULLET_COL, pygame.Rect(x - 2, y - 6, 4, 12))
        # ghost = last authoritative server state
        if show_ghost and ghost is not None:
            gsh = S.ships(ghost[None, :])[0]
            for p in range(C.N_PLAYERS):
                pygame.draw.polygon(scr, GHOST_COL[p], self.ship_poly(gsh[p, 0], gsh[p, 1]), width=1)
        # ships (predicted, with visual smoothing offset applied)
        for p in range(C.N_PLAYERS):
            poly = self.ship_poly(render_pos[p, 0], render_pos[p, 1])
            pygame.draw.polygon(scr, SHIP_COL[p], poly)
            if p == own:
                pygame.draw.polygon(scr, (255, 255, 255), poly, width=2)
        self.hud(hud_lines)

    def hud(self, hud_lines: list[str]) -> None:
        y = self.size + 6
        for line in hud_lines:
            self.screen.blit(self.font.render(line, True, HUD_COL), (8, y))
            y += 17
        pygame.display.flip()

    def screenshot(self, path: str) -> None:
        pygame.image.save(self.screen, path)
