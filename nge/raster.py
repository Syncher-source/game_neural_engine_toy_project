"""Batched GPU rasterizer (torch) of the classic look: ground truth for the neural renderer.

rasterize(state, H, W) -> (B, H, W, 3) float32 in [0, 1]
"""
from __future__ import annotations

import torch

from . import config as C
from . import state as S

BG = (12 / 255, 12 / 255, 24 / 255)
LINE = (40 / 255, 40 / 255, 70 / 255)
SHIP_COL = ((80 / 255, 200 / 255, 1.0), (1.0, 150 / 255, 80 / 255))
INV_COL = ((180 / 255, 1.0, 120 / 255), (1.0, 230 / 255, 100 / 255), (1.0, 120 / 255, 160 / 255))
BULLET_HALF_W, BULLET_HALF_H = 0.006, 0.015   # ~3x7 px at 240x240; sub-pixel bullets were unlearnable


def _grid(H: int, W: int, device):
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return X, Y


def _tri(X, Y, p0, p1, p2):
    """Point-in-triangle mask. X, Y: (H, W); p*: (B, 2) -> (B, H, W) bool."""
    def edge(a, b):
        return (X[None] - a[:, 0, None, None]) * (b[:, 1] - a[:, 1])[:, None, None] - (Y[None] - a[:, 1, None, None]) * (b[:, 0] - a[:, 0])[:, None, None]
    e0, e1, e2 = edge(p0, p1), edge(p1, p2), edge(p2, p0)
    return ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))


def _paint(img, mask, color):
    col = torch.tensor(color, device=img.device, dtype=img.dtype)
    img[mask] = col


@torch.no_grad()
def rasterize(s: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, dev = s.shape[0], s.device
    X, Y = _grid(H, W, dev)
    img = torch.empty(B, H, W, 3, device=dev, dtype=torch.float32)
    img[:] = torch.tensor(BG, device=dev)
    # band line
    line = (Y - C.SHIP_Y_MIN).abs() < 0.5 / H
    _paint(img, line[None].expand(B, H, W), LINE)
    # invaders
    form = s[:, S.FORM_OFF:S.FORM_OFF + 2]
    for i in range(C.N_INVADERS):
        row, col = divmod(i, C.INV_COLS)
        ix = C.INV_X0 + form[:, 0] + col * C.INV_DX
        iy = C.INV_Y0 + form[:, 1] + row * C.INV_DY
        alive = s[:, S.INV_OFF + i] > 0.5
        m = alive[:, None, None] & ((X[None] - ix[:, None, None]).abs() < C.INV_HALF_W) & ((Y[None] - iy[:, None, None]).abs() < C.INV_HALF_H)
        _paint(img, m, INV_COL[row])
    # bullets
    for p in range(C.N_PLAYERS):
        for b in range(C.N_BULLETS):
            o = S.bullet_idx(p, b, 0)
            act = s[:, o + 2] > 0.5
            m = act[:, None, None] & ((X[None] - s[:, o, None, None]).abs() < BULLET_HALF_W) & ((Y[None] - s[:, o + 1, None, None]).abs() < BULLET_HALF_H)
            _paint(img, m, (1.0, 1.0, 1.0))
    # ships: same polygon as the SDL/pygame renderers
    R = C.SHIP_RADIUS
    for p in range(C.N_PLAYERS):
        cx, cy = s[:, S.ship_idx(p, 0)], s[:, S.ship_idx(p, 1)]
        top = torch.stack([cx, cy - R], 1)
        left = torch.stack([cx - R, cy + 0.9 * R], 1)
        notch = torch.stack([cx, cy + 0.4 * R], 1)
        right = torch.stack([cx + R, cy + 0.9 * R], 1)
        m = _tri(X, Y, top, left, notch) | _tri(X, Y, top, notch, right)
        _paint(img, m, SHIP_COL[p])
    return img
