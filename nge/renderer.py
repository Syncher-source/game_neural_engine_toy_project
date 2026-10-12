"""Neural renderer: state -> RGB frame, a spatial-broadcast per-pixel network.

Every pixel receives the (normalised) world state plus Fourier features of its own
(x, y) coordinate and a small conv stack decides its colour.  The network is
resolution-agnostic in principle (the coordinate grid is a buffer), so the same
weights can be exported at any H x W.

Local effects state
-------------------
Explosions are not game state: nothing about them is simulated by the world model or sent over
the wire. The renderer keeps a small *local* state vector `h` of its own, FX_DIM floats, one
record per invader slot: [seconds since the slot was last alive, x, y of the slot when it was
last alive]. It is advanced by `step_fx(state, h, dt)` with frame time, a fixed rule of the
kind the exported graphs already use for clamps (nothing is learned there), and the network
draws from (state, h): the explosion itself, how sparks fly and fade, is learned from the
teacher's particle system (`effects.py`). On a client, `h` lives only in the renderer; rewind
and replay never touch it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import state as S

# ---- local effects state layout (renderer-only, never on the wire) ----
FX_STRIDE = 3                            # per invader slot: [age, cx, cy]
FX_DIM = C.N_INVADERS * FX_STRIDE
FX_DURATION = 0.7                        # seconds; ages saturate here and nothing is drawn (same as effects.DURATION)
FX_SEED = 0.5                            # the teacher's fixed particle seed when training the renderer


def fx_slot_positions(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, N_INVADERS) x and y of every invader slot from the formation offsets."""
    rows, cols = torch.div(torch.arange(C.N_INVADERS, device=state.device), C.INV_COLS, rounding_mode="floor"), torch.arange(C.N_INVADERS, device=state.device) % C.INV_COLS
    x = C.INV_X0 + state[:, S.FORM_OFF:S.FORM_OFF + 1] + cols[None].to(state.dtype) * C.INV_DX
    y = C.INV_Y0 + state[:, S.FORM_OFF + 1:S.FORM_OFF + 2] + rows[None].to(state.dtype) * C.INV_DY
    return x, y


def fx_idle(state: torch.Tensor) -> torch.Tensor:
    """Local state with every explosion long over: what a renderer starts from."""
    x, y = fx_slot_positions(state)
    return torch.stack([torch.full_like(x, FX_DURATION), x, y], -1).reshape(state.shape[0], FX_DIM)


def step_fx(state: torch.Tensor, h: torch.Tensor, dt: torch.Tensor | float) -> torch.Tensor:
    """Advance the local state by one rendered frame of `dt` seconds.
    Alive slot: age 0 and the centre follows the slot. Dead slot: age grows (saturating at
    FX_DURATION) and the centre stays where the slot was when it was last alive."""
    B = state.shape[0]
    alive = (state[:, S.INV_OFF:S.STATE_DIM] > 0.5).to(state.dtype)             # (B, N)
    hh = h.reshape(B, C.N_INVADERS, FX_STRIDE)
    x, y = fx_slot_positions(state)
    dt = torch.as_tensor(dt, dtype=state.dtype, device=state.device).reshape(-1, 1)
    age = (1.0 - alive) * torch.minimum(hh[:, :, 0] + dt, torch.full_like(hh[:, :, 0], FX_DURATION))
    cx = alive * x + (1.0 - alive) * hh[:, :, 1]
    cy = alive * y + (1.0 - alive) * hh[:, :, 2]
    return torch.stack([age, cx, cy], -1).reshape(B, FX_DIM)


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
    # explosions: every pixel picks the one exploding slot within reach (clamp +-2 => +-0.12 world,
    # as far as the sparks fly) and receives its offset from that slot's remembered centre, with the
    # same Fourier expansion the bullets use, plus the slot's age features. One shared feature set
    # for all 18 slots, so the explosion is learned once, not once per slot.
    FX_SCALE = 0.06
    FX_REACH = 2.2                   # circular reach in scaled units; the features fade to 0 over the last FX_EDGE units
    FX_EDGE = 0.4                    # (a hard square edge showed through as a faint box around every explosion)
    FX_AGE_FEATS = 7                 # life, sin/cos(pi life), sin/cos(2 pi life), sin/cos(4 pi life)
    # sparks are radial streaks and the ring is a circle: separable in polar coordinates, so the
    # pixel also gets angle harmonics (cos/sin k theta, by the angle-addition recurrence: no atan2
    # in the graph) and radial Fourier features fine enough to resolve a 6 px spark
    FX_HARMONICS = 16
    FX_RADIAL_FREQS = (0.5, 1.0, 2.0, 4.0, 8.0)

    def __init__(self, H: int = 180, W: int = 180, hidden: int = 256, n_freq: int = 6, s_mean=None, s_std=None, rel_coords: bool = True,
                 rel_fourier: bool = True, fx: bool = False):
        super().__init__()
        self.H, self.W, self.hidden, self.n_freq, self.rel_coords, self.rel_fourier, self.fx = H, W, hidden, n_freq, rel_coords, rel_fourier, fx
        self.register_buffer("obj_scale", torch.tensor(self.OBJ_SCALE if rel_fourier else [self.REL_SCALE] * len(self.OBJ_XY))[None, :, None, None])
        self.register_buffer("s_mean", torch.zeros(S.STATE_DIM) if s_mean is None else torch.as_tensor(s_mean, dtype=torch.float32))
        self.register_buffer("s_std", torch.ones(S.STATE_DIM) if s_std is None else torch.as_tensor(s_std, dtype=torch.float32))
        self.register_buffer("coords", self.make_coords(H, W, n_freq), persistent=False)   # 34 x H x W, rebuilt from H, W
        self.register_buffer("ones_hw", torch.ones(1, 1, H, W), persistent=False)   # broadcasting helper (exports as a plain Mul, no Expand)
        self.register_buffer("obj_x_idx", torch.tensor([i for i, _ in self.OBJ_XY], dtype=torch.long))
        self.register_buffer("obj_y_idx", torch.tensor([j for _, j in self.OBJ_XY], dtype=torch.long))
        self.register_buffer("fx_age_idx", torch.arange(0, FX_DIM, FX_STRIDE))
        self.register_buffer("fx_x_idx", torch.arange(1, FX_DIM, FX_STRIDE))
        self.register_buffer("fx_y_idx", torch.arange(2, FX_DIM, FX_STRIDE))
        # formation-relative coordinates (+ Fourier features): the invader grid is a fixed pattern in that frame
        in_ch = S.STATE_DIM + self.coords.shape[0] + ((2 * len(self.OBJ_XY) + 2 + 4 * n_freq) if rel_coords else 0) \
            + ((2 * len(self.OBJ_XY) * 2 * len(self.REL_FREQS) + len(self.OBJ_XY)) if (rel_coords and rel_fourier) else 0) \
            + ((3 + 2 * 2 * len(self.REL_FREQS) + 2 * self.FX_HARMONICS + 2 * len(self.FX_RADIAL_FREQS) + self.FX_AGE_FEATS) if fx else 0)
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

    def forward(self, state: torch.Tensor, window=None, h: torch.Tensor | None = None) -> torch.Tensor:
        """(B, STATE_DIM) [+ local state h (B, FX_DIM)] -> (B, H, W, 3) in [0, 1].

        window = (y0, x0, h, w) renders only that pixel window of the H x W frame (training on
        crops). The window is evaluated with a 2 px margin so the 3x3 convolutions see the same
        neighbourhood as in the full frame (except at the frame border).
        Without `h` the local state is idle (no explosion in flight).
        """
        B = state.shape[0]
        if window is None:
            coords = self.coords
            x = ((state - self.s_mean) / self.s_std)[:, :, None, None] * self.ones_hw    # (B, STATE_DIM, H, W)
            c = coords[None] * torch.ones(B, 1, 1, 1, device=state.device, dtype=state.dtype)
        else:
            y0, x0, hh, ww = window
            m = 2
            ya, yb, xa, xb = max(y0 - m, 0), min(y0 + hh + m, self.H), max(x0 - m, 0), min(x0 + ww + m, self.W)
            coords = self.coords[:, ya:yb, xa:xb]
            x = ((state - self.s_mean) / self.s_std)[:, :, None, None].expand(-1, -1, yb - ya, xb - xa)
            c = coords[None].expand(B, -1, -1, -1)
        feats = [x, c]
        X, Y = coords[0], coords[1]                                                 # (h, w)
        if self.rel_coords:
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
                feats += self._fourier(rx, ry)
        if self.fx:
            if h is None:
                h = fx_idle(state)
            age = torch.index_select(h, 1, self.fx_age_idx)                           # (B, N)
            cx = torch.index_select(h, 1, self.fx_x_idx)[:, :, None, None]
            cy = torch.index_select(h, 1, self.fx_y_idx)[:, :, None, None]
            alive = (state[:, S.INV_OFF:S.STATE_DIM] > 0.5).to(state.dtype)
            life = torch.clamp(1.0 - age / FX_DURATION, 0.0, 1.0) * (1.0 - alive)   # 0 for alive slots and finished explosions
            rx_all = (X[None, None] - cx) / self.FX_SCALE                           # (B, N, h, w)
            ry_all = (Y[None, None] - cy) / self.FX_SCALE
            r_all = torch.sqrt(rx_all * rx_all + ry_all * ry_all + 1e-8)
            edge = torch.clamp((self.FX_REACH - r_all) / self.FX_EDGE, 0.0, 1.0)      # 1 inside, soft 0 at the reach
            score = life[:, :, None, None] * edge                                     # (B, N, h, w)
            best = torch.argmax(score, dim=1, keepdim=True)                           # the one live explosion this pixel belongs to
            m = torch.gather(edge, 1, best) * (torch.gather(score, 1, best) > 0.0).to(state.dtype)   # soft in-reach mask of that slot
            lf = torch.gather(life[:, :, None, None] * torch.ones_like(edge), 1, best) * m              # its life, 0 where none is in reach
            rx = torch.clamp(torch.gather(rx_all, 1, best), -self.FX_REACH, self.FX_REACH) * m
            ry = torch.clamp(torch.gather(ry_all, 1, best), -self.FX_REACH, self.FX_REACH) * m
            feats += [rx, ry] + [f * m for f in self._fourier(rx, ry)]
            r = torch.gather(r_all, 1, best)
            ct, st = torch.gather(rx_all, 1, best) / (r + 1e-3), torch.gather(ry_all, 1, best) / (r + 1e-3)   # cos/sin of the angle to the centre
            ck, sk = ct, st
            for _ in range(self.FX_HARMONICS):
                feats += [ck * m, sk * m]
                ck, sk = ck * ct - sk * st, sk * ct + ck * st
            rr = torch.clamp(r, max=self.FX_REACH)
            feats += [g(f * math.pi * rr) * m for f in self.FX_RADIAL_FREQS for g in (torch.sin, torch.cos)]
            feats += [lf] + [g(k * math.pi * lf) * m for k in (1.0, 2.0, 4.0) for g in (torch.sin, torch.cos)]
        y = self.net(torch.cat(feats, 1))
        if window is not None:
            y = y[:, :, y0 - ya:y0 - ya + hh, x0 - xa:x0 - xa + ww]
        return torch.sigmoid(y).permute(0, 2, 3, 1)

    def _fourier(self, rx, ry):
        out = []
        for f in self.REL_FREQS:
            out += [torch.sin(f * math.pi * rx), torch.cos(f * math.pi * rx), torch.sin(f * math.pi * ry), torch.cos(f * math.pi * ry)]
        # radial distance to each object: not separable in x / y, so two objects sharing a
        # row or a column cannot add up to a ghost between them
        out += [torch.clamp(torch.sqrt(rx * rx + ry * ry + 1e-6), max=2.0)]
        return out

    def at_resolution(self, H: int, W: int) -> "NeuralRenderer":
        """Same weights, different output size."""
        m = NeuralRenderer(H, W, self.hidden, self.n_freq, self.s_mean, self.s_std, self.rel_coords, self.rel_fourier, self.fx)
        m.net.load_state_dict(self.net.state_dict())
        return m.to(self.s_mean.device).eval()


def save_renderer(model: NeuralRenderer, path: str | Path, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"H": model.H, "W": model.W, "hidden": model.hidden, "n_freq": model.n_freq, "rel_coords": model.rel_coords,
                "rel_fourier": model.rel_fourier, "fx": model.fx, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def grow_first_layer(sd: dict, model: NeuralRenderer) -> dict:
    """The per-pixel feature set grew since this state dict was saved: old channels keep their
    weights, new ones start at zero (the model behaves exactly as before until fine-tuned)."""
    sd = dict(sd)
    w0 = "net.0.weight"
    if sd[w0].shape != model.net[0].weight.shape:
        new = torch.zeros_like(model.net[0].weight); n = min(sd[w0].shape[1], new.shape[1]); new[:, :n] = sd[w0][:, :n]; sd[w0] = new
    return sd


def load_renderer(path: str | Path, device: str = "cpu", fx: bool | None = None) -> NeuralRenderer:
    """fx=True loads an older checkpoint into a model with the effects features (zero-initialised)."""
    ck = torch.load(path, map_location="cpu")
    m = NeuralRenderer(ck["H"], ck["W"], ck["hidden"], ck["n_freq"], rel_coords=ck.get("rel_coords", False), rel_fourier=ck.get("rel_fourier", False),
                       fx=ck.get("fx", False) if fx is None else fx)
    m.load_state_dict(grow_first_layer(ck["state_dict"], m), strict=False)      # older checkpoints carry the coords buffer; it is rebuilt
    return m.to(device).eval()


class RendererRuntime:
    """Client-side helper: numpy state -> uint8 RGB frame (H, W, 3). Keeps the renderer's local
    effects state between frames and advances it with wall-clock frame time."""

    def __init__(self, ckpt: str | Path, device: str = "cpu"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = load_renderer(ckpt, device)
        self.name = f"neural({device})" + ("+fx" if self.model.fx else "")
        self.last_ms = 0.0
        self.h = None
        self.t_last = None

    def reset_fx(self) -> None:
        self.h = None

    @torch.no_grad()
    def render(self, state: np.ndarray) -> np.ndarray:
        import time
        t = time.perf_counter()
        st = torch.as_tensor(np.ascontiguousarray(state, dtype=np.float32), device=self.device)[None]
        if self.model.fx:
            dt = 0.0 if self.t_last is None else min(max(t - self.t_last, 0.0), 0.1)
            self.t_last = t
            self.h = step_fx(st, fx_idle(st) if self.h is None else self.h, dt)
        img = (self.model(st, h=self.h)[0] * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        self.last_ms = (time.perf_counter() - t) * 1e3
        return img
