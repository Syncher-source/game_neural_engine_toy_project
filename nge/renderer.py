"""Neural renderer: state -> RGB frame, a spatial-broadcast per-pixel network.

Every pixel receives the (normalised) world state plus Fourier features of its own
(x, y) coordinate and a small conv stack decides its colour.  The network is
resolution-agnostic in principle (the coordinate grid is a buffer), so the same
weights can be exported at any H x W.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import state as S


class NeuralRenderer(nn.Module):
    # (x, y) index pairs of the movable objects: ships and bullet slots. With `rel_coords` every
    # pixel also receives its offset from each of these, so small sprites are easy to place; the
    # network still learns every shape, colour, visibility rule and the compositing.
    OBJ_XY = [(S.ship_idx(p, 0), S.ship_idx(p, 1)) for p in range(C.N_PLAYERS)] + \
             [(S.bullet_idx(p, b, 0), S.bullet_idx(p, b, 1)) for p in range(C.N_PLAYERS) for b in range(C.N_BULLETS)]
    REL_SCALE = 0.05
    # per-object scale of the relative coordinates: bullets are thin, so their neighbourhood is
    # magnified (clamp +-2 => +-0.05 world) and, with `rel_fourier`, expanded into sin/cos features
    # at 3 frequencies, which makes a few-pixel-wide bright bolt linearly accessible
    OBJ_SCALE = [0.05] * C.N_PLAYERS + [0.025] * (C.N_PLAYERS * C.N_BULLETS)
    REL_FREQS = (0.5, 1.0, 2.0)      # periods 4, 2, 1 in clamped units: no repeats inside the +-2 window at the lowest one

    def __init__(self, H: int = 180, W: int = 180, hidden: int = 256, n_freq: int = 6, s_mean=None, s_std=None, rel_coords: bool = True,
                 rel_fourier: bool = True):
        super().__init__()
        self.H, self.W, self.hidden, self.n_freq, self.rel_coords, self.rel_fourier = H, W, hidden, n_freq, rel_coords, rel_fourier
        self.register_buffer("obj_scale", torch.tensor(self.OBJ_SCALE if rel_fourier else [self.REL_SCALE] * len(self.OBJ_XY))[None, :, None, None])
        self.register_buffer("s_mean", torch.zeros(S.STATE_DIM) if s_mean is None else torch.as_tensor(s_mean, dtype=torch.float32))
        self.register_buffer("s_std", torch.ones(S.STATE_DIM) if s_std is None else torch.as_tensor(s_std, dtype=torch.float32))
        self.register_buffer("coords", self.make_coords(H, W, n_freq), persistent=False)   # 34 x H x W, rebuilt from H, W
        self.register_buffer("ones_hw", torch.ones(1, 1, H, W), persistent=False)   # broadcasting helper (exports as a plain Mul, no Expand)
        self.register_buffer("obj_x_idx", torch.tensor([i for i, _ in self.OBJ_XY], dtype=torch.long))
        self.register_buffer("obj_y_idx", torch.tensor([j for _, j in self.OBJ_XY], dtype=torch.long))
        # formation-relative coordinates (+ Fourier features): the invader grid is a fixed pattern in that frame
        in_ch = S.STATE_DIM + self.coords.shape[0] + ((2 * len(self.OBJ_XY) + 2 + 4 * n_freq) if rel_coords else 0) \
            + ((2 * len(self.OBJ_XY) * 2 * len(self.REL_FREQS) + len(self.OBJ_XY)) if (rel_coords and rel_fourier) else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden // 2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 1),
        )

    @staticmethod
    def make_coords(H: int, W: int, n_freq: int) -> torch.Tensor:
        ys = (torch.arange(H, dtype=torch.float32) + 0.5) / H
        xs = (torch.arange(W, dtype=torch.float32) + 0.5) / W
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        feats = [X, Y]
        for k in range(n_freq):
            f = (2.0 ** k) * math.pi
            feats += [torch.sin(f * X), torch.cos(f * X), torch.sin(f * Y), torch.cos(f * Y)]
        return torch.stack(feats, 0)                        # (2 + 4 n_freq, H, W)

    def forward(self, state: torch.Tensor, window=None) -> torch.Tensor:
        """(B, STATE_DIM) -> (B, H, W, 3) in [0, 1].

        window = (y0, x0, h, w) renders only that pixel window of the H x W frame (training on
        crops). The window is evaluated with a 2 px margin so the 3x3 convolutions see the same
        neighbourhood as in the full frame (except at the frame border).
        """
        B = state.shape[0]
        if window is None:
            coords = self.coords
            x = ((state - self.s_mean) / self.s_std)[:, :, None, None] * self.ones_hw    # (B, STATE_DIM, H, W)
            c = coords[None] * torch.ones(B, 1, 1, 1, device=state.device, dtype=state.dtype)
        else:
            y0, x0, h, w = window
            m = 2
            ya, yb, xa, xb = max(y0 - m, 0), min(y0 + h + m, self.H), max(x0 - m, 0), min(x0 + w + m, self.W)
            coords = self.coords[:, ya:yb, xa:xb]
            x = ((state - self.s_mean) / self.s_std)[:, :, None, None].expand(-1, -1, yb - ya, xb - xa)
            c = coords[None].expand(B, -1, -1, -1)
        feats = [x, c]
        if self.rel_coords:
            X, Y = coords[0], coords[1]                                             # (h, w)
            ox = torch.index_select(state, 1, self.obj_x_idx)[:, :, None, None]     # (B, K, 1, 1)
            oy = torch.index_select(state, 1, self.obj_y_idx)[:, :, None, None]
            rx = torch.clamp((X[None, None] - ox) / self.obj_scale, -2.0, 2.0)
            ry = torch.clamp((Y[None, None] - oy) / self.obj_scale, -2.0, 2.0)
            feats += [rx, ry]
            fx = X[None, None] - state[:, S.FORM_OFF + 0][:, None, None, None]        # (B, 1, H, W)
            fy = Y[None, None] - state[:, S.FORM_OFF + 1][:, None, None, None]
            ff = [fx, fy]
            for k in range(self.n_freq):
                f = (2.0 ** k) * math.pi
                ff += [torch.sin(f * fx), torch.cos(f * fx), torch.sin(f * fy), torch.cos(f * fy)]
            feats += ff
            if self.rel_fourier:
                for f in self.REL_FREQS:
                    feats += [torch.sin(f * math.pi * rx), torch.cos(f * math.pi * rx), torch.sin(f * math.pi * ry), torch.cos(f * math.pi * ry)]
                # radial distance to each object: not separable in x / y, so two objects sharing a
                # row or a column cannot add up to a ghost between them
                feats += [torch.clamp(torch.sqrt(rx * rx + ry * ry + 1e-6), max=2.0)]
        y = self.net(torch.cat(feats, 1))
        if window is not None:
            y = y[:, :, y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w]
        return torch.sigmoid(y).permute(0, 2, 3, 1)

    def at_resolution(self, H: int, W: int) -> "NeuralRenderer":
        """Same weights, different output size."""
        m = NeuralRenderer(H, W, self.hidden, self.n_freq, self.s_mean, self.s_std, self.rel_coords, self.rel_fourier)
        m.net.load_state_dict(self.net.state_dict())
        return m.to(self.s_mean.device).eval()


def save_renderer(model: NeuralRenderer, path: str | Path, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"H": model.H, "W": model.W, "hidden": model.hidden, "n_freq": model.n_freq, "rel_coords": model.rel_coords,
                "rel_fourier": model.rel_fourier, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_renderer(path: str | Path, device: str = "cpu") -> NeuralRenderer:
    ck = torch.load(path, map_location="cpu")
    m = NeuralRenderer(ck["H"], ck["W"], ck["hidden"], ck["n_freq"], rel_coords=ck.get("rel_coords", False), rel_fourier=ck.get("rel_fourier", False))
    sd = dict(ck["state_dict"])
    w0 = "net.0.weight"
    if sd[w0].shape != m.net[0].weight.shape:
        # the per-pixel feature set grew since this checkpoint was saved: old channels keep their
        # weights, new ones start at zero (the model behaves exactly as before until fine-tuned)
        new = torch.zeros_like(m.net[0].weight); n = min(sd[w0].shape[1], new.shape[1]); new[:, :n] = sd[w0][:, :n]; sd[w0] = new
    m.load_state_dict(sd, strict=False)      # older checkpoints carry the coords buffer; it is rebuilt
    return m.to(device).eval()


class RendererRuntime:
    """Client-side helper: numpy state -> uint8 RGB frame (H, W, 3)."""

    def __init__(self, ckpt: str | Path, device: str = "cpu"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = load_renderer(ckpt, device)
        self.name = f"neural({device})"
        self.last_ms = 0.0

    @torch.no_grad()
    def render(self, state: np.ndarray) -> np.ndarray:
        import time
        t = time.perf_counter()
        st = torch.as_tensor(np.ascontiguousarray(state, dtype=np.float32), device=self.device)[None]
        img = (self.model(st)[0] * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        self.last_ms = (time.perf_counter() - t) * 1e3
        return img
