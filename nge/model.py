"""Neural world model: a residual MLP that replaces the classic physics step.

    next_state = f_theta(state, inputs)

Continuous entries are predicted as normalised deltas (next = cur + delta*std),
binary flags as logits.  `NeuralSim` wraps the network behind the same
`step(s, u)` interface as `ClassicSim`, keeps the weights on the chosen device
(the server uses the GPU) and post-processes the output so the state stays on
the game manifold: flags are thresholded, inactive bullets are zeroed and
continuous values are clamped to their legal ranges.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import state as S

CONT_IDX_T = torch.as_tensor(S.CONT_IDX, dtype=torch.long)
BIN_IDX_T = torch.as_tensor(S.BIN_IDX, dtype=torch.long)
BUL_ACT_T = torch.as_tensor(S.BUL_ACT_IDX, dtype=torch.long)
BUL_POS_T = torch.as_tensor(S.BUL_POS_IDX.reshape(-1, 2), dtype=torch.long)   # (6, 2) -> x,y per bullet


class WorldModel(nn.Module):
    def __init__(self, hidden: int = 512, layers: int = 4, s_mean=None, s_std=None, d_std=None):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        in_dim = S.STATE_DIM + S.INPUT_DIM
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        ])
        self.out = nn.Linear(hidden, S.N_CONT + S.N_BIN)
        self.register_buffer("s_mean", torch.zeros(S.STATE_DIM) if s_mean is None else torch.as_tensor(s_mean, dtype=torch.float32))
        self.register_buffer("s_std", torch.ones(S.STATE_DIM) if s_std is None else torch.as_tensor(s_std, dtype=torch.float32))
        self.register_buffer("d_std", torch.ones(S.N_CONT) if d_std is None else torch.as_tensor(d_std, dtype=torch.float32))
        self.register_buffer("cont_lo", torch.as_tensor(S.CONT_LO[S.CONT_IDX]))
        self.register_buffer("cont_hi", torch.as_tensor(S.CONT_HI[S.CONT_IDX]))

    @staticmethod
    def split(x: torch.Tensor):
        """x (B, MODEL_DIM) -> state (B, STATE_DIM), inputs (B, INPUT_DIM)."""
        return x[:, : S.STATE_DIM], x[:, S.STATE_DIM:]

    def forward(self, x: torch.Tensor):
        """x: the single model vector (B, MODEL_DIM) = [state | inputs].
        Returns (normalised continuous deltas (B, N_CONT), binary logits (B, N_BIN))."""
        s, u = self.split(x)
        x = torch.cat([(s - self.s_mean) / self.s_std, u], dim=-1)
        h = torch.nn.functional.silu(self.inp(x))
        for blk in self.blocks:
            h = h + blk(h)
        y = self.out(h)
        return y[:, : S.N_CONT], y[:, S.N_CONT:]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """World step: x (B, MODEL_DIM) -> x_next (B, MODEL_DIM) with the input slots consumed (zero)."""
        nxt = self.predict_state(x)
        return torch.cat([nxt, torch.zeros(x.shape[0], S.INPUT_DIM, device=x.device, dtype=x.dtype)], dim=1)

    @torch.no_grad()
    def predict_state(self, x: torch.Tensor) -> torch.Tensor:
        """Next world state (B, STATE_DIM) with manifold post-processing."""
        s, _ = self.split(x)
        d, logit = self.forward(x)
        nxt = s.clone()
        cont = s[:, CONT_IDX_T.to(s.device)] + d * self.d_std
        nxt[:, CONT_IDX_T.to(s.device)] = torch.minimum(torch.maximum(cont, self.cont_lo), self.cont_hi)
        nxt[:, BIN_IDX_T.to(s.device)] = (logit > 0).to(s.dtype)
        act = nxt[:, BUL_ACT_T.to(s.device)]                        # (B, 6)
        pos_idx = BUL_POS_T.to(s.device)
        nxt[:, pos_idx[:, 0]] = nxt[:, pos_idx[:, 0]] * act
        nxt[:, pos_idx[:, 1]] = nxt[:, pos_idx[:, 1]] * act
        return nxt


def save_checkpoint(model: WorldModel, path: str | Path, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": model.hidden, "layers": model.layers, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_model(path: str | Path, device: str = "cpu") -> WorldModel:
    ck = torch.load(path, map_location="cpu")
    m = WorldModel(ck["hidden"], ck["layers"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


class NeuralSim:
    """Classic-sim-compatible `step` backed by the world model on `device`."""

    def __init__(self, ckpt: str | Path, device: str = "cpu", threads: int | None = 1):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device == "cpu" and threads:
            torch.set_num_threads(threads)
        self.device = device
        self.model = load_model(ckpt, device)
        self.name = f"neural({device})"
        self.last_step_ms = 0.0

    def step(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(S.pack(s, u), device=self.device)       # the single model vector
        if self.device.startswith("cuda"):
            t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            out = self.model.predict(x)
            t1.record(); torch.cuda.synchronize()
            self.last_step_ms = t0.elapsed_time(t1)
        else:
            import time
            t = time.perf_counter()
            out = self.model.predict(x)
            self.last_step_ms = (time.perf_counter() - t) * 1e3
        return out[:, : S.STATE_DIM].cpu().numpy()                 # numpy convenience API keeps state in / state out

    def step_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Device-resident variant for batched evaluation / multi-room servers. x: (B, MODEL_DIM)."""
        return self.model.predict(x)
