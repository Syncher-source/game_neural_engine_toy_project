"""Reference ("classic") simulation, batched over worlds with numpy.

This is the ground-truth physics the neural world model is trained to imitate.
`step(s, u)` maps a batch of states and inputs to the batch of next states.
It is deliberately written as pure array code so a few thousand worlds can be
simulated in parallel when generating training data.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import state as S


def step(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Advance one tick.

    s: (B, STATE_DIM) float32
    u: (B, INPUT_DIM) float32, per player [ax, ay, fire]
    returns a new (B, STATE_DIM) array (the input is not modified).
    """
    s = np.array(s, dtype=np.float32, copy=True)
    B = s.shape[0]
    sh = S.ships(s)
    bu = S.bullets(s)
    fo = S.formation(s)
    inv = S.invaders(s)
    inp = np.asarray(u, dtype=np.float32).reshape(B, C.N_PLAYERS, S.INPUT_STRIDE)
    ax = np.clip(inp[..., 0], -1, 1)
    ay = np.clip(inp[..., 1], -1, 1)
    fire = inp[..., 2] > 0.5

    # ---- ship dynamics --------------------------------------------------
    pos = sh[..., 0:2]
    vel = sh[..., 2:4]
    vel[..., 0] = (vel[..., 0] + ax * C.SHIP_ACCEL * C.DT) * C.SHIP_DAMPING
    vel[..., 1] = (vel[..., 1] + ay * C.SHIP_ACCEL * C.DT) * C.SHIP_DAMPING
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    vel *= np.minimum(1.0, C.SHIP_MAX_SPEED / np.maximum(speed, 1e-9))
    pos += vel * C.DT

    # walls stop the ship on that axis
    lo_x, hi_x = pos[..., 0] < C.SHIP_X_MIN, pos[..., 0] > C.SHIP_X_MAX
    lo_y, hi_y = pos[..., 1] < C.SHIP_Y_MIN, pos[..., 1] > C.SHIP_Y_MAX
    pos[..., 0] = np.clip(pos[..., 0], C.SHIP_X_MIN, C.SHIP_X_MAX)
    pos[..., 1] = np.clip(pos[..., 1], C.SHIP_Y_MIN, C.SHIP_Y_MAX)
    vel[..., 0] = np.where(lo_x | hi_x, 0.0, vel[..., 0])
    vel[..., 1] = np.where(lo_y | hi_y, 0.0, vel[..., 1])

    # ship-ship collision: both speeds go to zero, ships are separated
    d = pos[:, 1] - pos[:, 0]
    dist = np.linalg.norm(d, axis=-1)
    coll = dist < 2 * C.SHIP_RADIUS
    if coll.any():
        n = np.where(dist[:, None] > 1e-6, d / np.maximum(dist[:, None], 1e-6), np.array([1.0, 0.0], dtype=np.float32))
        mid = 0.5 * (pos[:, 0] + pos[:, 1])
        p0 = mid - n * C.SHIP_RADIUS
        p1 = mid + n * C.SHIP_RADIUS
        pos[coll, 0] = p0[coll]
        pos[coll, 1] = p1[coll]
        vel[coll] = 0.0
        pos[..., 0] = np.clip(pos[..., 0], C.SHIP_X_MIN, C.SHIP_X_MAX)
        pos[..., 1] = np.clip(pos[..., 1], C.SHIP_Y_MIN, C.SHIP_Y_MAX)

    # cooldown
    sh[..., 4] = np.maximum(0.0, sh[..., 4] - C.FIRE_COOLDOWN_DECAY)

    # ---- bullets --------------------------------------------------------
    active = bu[..., 2] > 0.5
    bu[..., 1] = np.where(active, bu[..., 1] - C.BULLET_SPEED * C.DT, 0.0)
    bu[..., 0] = np.where(active, bu[..., 0], 0.0)
    off = active & (bu[..., 1] < 0.0)
    active &= ~off

    # bullet / invader hits
    ipos = S.invader_positions(fo)                                  # (B, NI, 2)
    bx = bu[..., 0][..., None]                                     # (B, P, NB, 1)
    by = bu[..., 1][..., None]
    ix = ipos[:, None, None, :, 0]                                 # (B, 1, 1, NI)
    iy = ipos[:, None, None, :, 1]
    alive = inv[:, None, None, :] > 0.5
    hits = active[..., None] & alive & (np.abs(bx - ix) < C.INV_HALF_W) & (np.abs(by - iy) < C.INV_HALF_H)
    killed = hits.any(axis=(1, 2))                                 # (B, NI)
    bullet_hit = hits.any(axis=3)                                  # (B, P, NB)
    inv[killed] = 0.0
    active &= ~bullet_hit

    bu[..., 2] = active.astype(np.float32)
    bu[..., 0] = np.where(active, bu[..., 0], 0.0)
    bu[..., 1] = np.where(active, bu[..., 1], 0.0)

    # firing: needs cooldown expired and a free slot
    can_fire = fire & (sh[..., 4] <= 0.0) & (~active).any(axis=-1)   # (B, P)
    slot = np.argmax(~active, axis=-1)                                # first free slot
    bi, pi = np.nonzero(can_fire)
    si = slot[bi, pi]
    bu[bi, pi, si, 0] = sh[bi, pi, 0]
    bu[bi, pi, si, 1] = sh[bi, pi, 1] - C.SHIP_RADIUS
    bu[bi, pi, si, 2] = 1.0
    sh[bi, pi, 4] = 1.0

    # ---- formation ------------------------------------------------------
    right = fo[:, 2] > 0.5
    fo[:, 0] += np.where(right, 1.0, -1.0) * C.FORM_SPEED * C.DT
    bounce_r = fo[:, 0] > C.FORM_OX_MAX
    bounce_l = fo[:, 0] < 0.0
    fo[:, 0] = np.clip(fo[:, 0], 0.0, C.FORM_OX_MAX)
    fo[:, 2] = np.where(bounce_r, 0.0, np.where(bounce_l, 1.0, fo[:, 2]))
    fo[:, 1] += np.where(bounce_r | bounce_l, C.FORM_STEP_DOWN, 0.0)
    fo[:, 1] = np.where(fo[:, 1] > C.FORM_OY_MAX + 1e-4, 0.0, fo[:, 1])

    # wave cleared -> new wave
    cleared = inv.sum(axis=1) == 0
    inv[cleared] = 1.0
    fo[cleared, 1] = 0.0
    return s


class ClassicSim:
    """Step interface shared with NeuralSim (see model.py)."""

    name = "classic"

    def step(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        return step(s, u)
