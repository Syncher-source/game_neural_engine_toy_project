"""Batched GPU sprite compositor (torch): the 1950s look, ground truth for the neural renderer.

    rasterize(state, H, W, window=None) -> rgb (B, h, w, 3), obj_alpha (B, h, w), laser_alpha (B, h, w)

Sprites come from `nge.sprites` (generated once per process, premultiplied RGBA textures) and
are placed with bilinear sampling, so the output is correct at any resolution and for any
sub-window (training uses random crops of the 720x720 frame). Training-side code only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import state as S

BG = (0.02, 0.02, 0.06)
SHIP_HALF = (0.042, 0.042)        # world half extents of the sprite quads
INV_HALF = (0.040, 0.040)
LASER_HALF = (0.014, 0.028)       # bolt ~12 px wide, 40 px tall at 720
SPRITE_PX = 128

_cache: dict = {}


def _tex(img) -> torch.Tensor:
    """PIL RGBA -> premultiplied float (4, H, W) with a transparent 1 px border."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    rgb, al = a[..., :3] * a[..., 3:4], a[..., 3:4]
    t = torch.from_numpy(np.concatenate([rgb, al], -1)).permute(2, 0, 1)
    return F.pad(t, (1, 1, 1, 1))


def sprites(device) -> dict:
    key = str(device)
    if key not in _cache:
        from . import sprites as SP
        _cache[key] = {
            "ship": torch.stack([_tex(SP.rocket(SPRITE_PX, p)) for p in range(C.N_PLAYERS)]).to(device),
            "inv": torch.stack([_tex(SP.saucer(SPRITE_PX)), _tex(SP.war_machine(SPRITE_PX)), _tex(SP.orb(SPRITE_PX))]).to(device),
            "laser": torch.stack([_tex(SP.laser(SPRITE_PX // 2, SPRITE_PX, p)) for p in range(C.N_PLAYERS)]).to(device),
        }
    return _cache[key]


def grid(H: int, W: int, device, window=None):
    """Pixel-centre world coordinates X, Y (h, w) of the full frame or of window (y0, x0, h, w)."""
    y0, x0, h, w = window if window is not None else (0, 0, H, W)
    ys = (torch.arange(y0, y0 + h, device=device, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(x0, x0 + w, device=device, dtype=torch.float32) + 0.5) / W
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return X, Y


def _layer(tex: torch.Tensor, X, Y, cx, cy, half, vis):
    """Sample one texture for K objects per batch item. tex (4, Hs, Ws); cx, cy, vis (B, K).
    Returns premultiplied RGBA (B, K, 4, h, w)."""
    B, K = cx.shape
    u = (X[None, None] - cx[:, :, None, None]) / half[0]        # (B, K, h, w) in [-1, 1] on the quad
    v = (Y[None, None] - cy[:, :, None, None]) / half[1]
    g = torch.stack([u, v], -1).reshape(B * K, *X.shape, 2)
    out = F.grid_sample(tex[None].expand(B * K, -1, -1, -1), g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.reshape(B, K, 4, *X.shape) * vis[:, :, None, None, None]


def _over(dst, src):
    """Premultiplied 'over': dst, src (B, 4, h, w)."""
    return src + dst * (1 - src[:, 3:4])


@torch.no_grad()
def rasterize(s: torch.Tensor, H: int, W: int, window=None):
    B, dev = s.shape[0], s.device
    sp = sprites(dev)
    X, Y = grid(H, W, dev, window)
    h, w = X.shape
    out = torch.zeros(B, 4, h, w, device=dev)
    out[:, :3] = torch.tensor(BG, device=dev)[None, :, None, None]
    out[:, 3] = 1.0
    # invaders: one texture per row, 6 per row; they never overlap so the row layer is a plain sum
    form = s[:, S.FORM_OFF:S.FORM_OFF + 2]
    cols = torch.arange(C.INV_COLS, device=dev, dtype=torch.float32)
    for row in range(C.INV_ROWS):
        cx = C.INV_X0 + form[:, 0:1] + cols[None] * C.INV_DX
        cy = (C.INV_Y0 + form[:, 1:2] + row * C.INV_DY).expand(-1, C.INV_COLS)
        alive = (s[:, S.INV_OFF + row * C.INV_COLS: S.INV_OFF + (row + 1) * C.INV_COLS] > 0.5).float()
        out = _over(out, _layer(sp["inv"][row], X, Y, cx, cy, INV_HALF, alive).sum(1))
    # lasers, then ships (drawn last: a ship passes in front of everything)
    laser_a = torch.zeros(B, h, w, device=dev)
    for p in range(C.N_PLAYERS):
        o = [S.bullet_idx(p, b, 0) for b in range(C.N_BULLETS)]
        cx = s[:, o]; cy = s[:, [i + 1 for i in o]]; act = (s[:, [i + 2 for i in o]] > 0.5).float()
        lay = _layer(sp["laser"][p], X, Y, cx, cy, LASER_HALF, act)
        for b in range(C.N_BULLETS):
            out = _over(out, lay[:, b]); laser_a = torch.maximum(laser_a, lay[:, b, 3])
    for p in range(C.N_PLAYERS):
        cx = s[:, S.ship_idx(p, 0):S.ship_idx(p, 0) + 1]; cy = s[:, S.ship_idx(p, 1):S.ship_idx(p, 1) + 1]
        out = _over(out, _layer(sp["ship"][p], X, Y, cx, cy, SHIP_HALF, torch.ones_like(cx))[:, 0])
    rgb = out[:, :3].permute(0, 2, 3, 1).clamp(0, 1)
    bg = torch.tensor(BG, device=dev)
    obj_a = ((rgb - bg).abs().sum(-1) > 2e-3).float()
    return rgb, obj_a, laser_a


def object_centres(s: torch.Tensor):
    """(B, N, 2) world positions of every object slot and (B, N) sampling weights for crop
    selection: invisible slots (dead invaders, inactive lasers) get 0, active lasers are
    up-weighted because they are the smallest and rarest thing on screen."""
    form = s[:, S.FORM_OFF:S.FORM_OFF + 2]
    pts, wts = [], []
    for i in range(C.N_INVADERS):
        row, col = divmod(i, C.INV_COLS)
        pts.append(torch.stack([C.INV_X0 + form[:, 0] + col * C.INV_DX, C.INV_Y0 + form[:, 1] + row * C.INV_DY], -1))
        wts.append((s[:, S.INV_OFF + i] > 0.5).float())
    for p in range(C.N_PLAYERS):
        for b in range(C.N_BULLETS):
            o = S.bullet_idx(p, b, 0); pts.append(s[:, o:o + 2]); wts.append((s[:, o + 2] > 0.5).float() * 6.0)
        pts.append(s[:, S.ship_idx(p, 0):S.ship_idx(p, 0) + 2]); wts.append(torch.full_like(s[:, 0], 3.0))
    return torch.stack(pts, 1), torch.stack(wts, 1)
