"""Flat world-state layout.

The whole game state is one float32 vector of STATE_DIM entries so that the
same array can be fed to the classic simulation, to the neural world model and
over the wire.  Batched arrays have shape (B, STATE_DIM).

Layout
------
ships      : N_PLAYERS x [x, y, vx, vy, cooldown]
bullets    : N_PLAYERS x N_BULLETS x [x, y, active]
formation  : [ox, oy, moving_right]
invaders   : N_INVADERS x alive

`active`, `moving_right` and `alive` are binary flags (0/1). Everything else is
continuous.  Inactive bullets have x = y = 0 by convention.

Inputs are a vector of INPUT_DIM entries: N_PLAYERS x [ax, ay, fire] with
ax, ay in [-1, 1] and fire in {0, 1}.
"""
from __future__ import annotations

import numpy as np

from . import config as C

SHIP_OFF = 0
SHIP_STRIDE = 5
BUL_OFF = SHIP_OFF + C.N_PLAYERS * SHIP_STRIDE
BUL_STRIDE = 3
FORM_OFF = BUL_OFF + C.N_PLAYERS * C.N_BULLETS * BUL_STRIDE
INV_OFF = FORM_OFF + 3
STATE_DIM = INV_OFF + C.N_INVADERS

INPUT_STRIDE = 3
INPUT_DIM = C.N_PLAYERS * INPUT_STRIDE
MODEL_DIM = STATE_DIM + INPUT_DIM   # the one vector a model receives: [world state | all player inputs]
# World model contract: x_in (MODEL_DIM) -> x_out (MODEL_DIM), same type, with the input slots of
# x_out zeroed ("consumed"). Renderer contract: x (MODEL_DIM) -> image; the input slots are
# ignored structurally (sliced off inside the graph).


def pack(state: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Build the model vector (B, MODEL_DIM) = [state | inputs].

    The server packs [authoritative world state | remote players' inputs]; a client packs
    [its local world state (network state + local prediction) | local input, last known
    remote input]. The models never see anything else.
    """
    return np.concatenate([np.asarray(state, dtype=np.float32), np.asarray(inputs, dtype=np.float32)], axis=1)


def ship_idx(p: int, k: int) -> int:
    return SHIP_OFF + p * SHIP_STRIDE + k


def bullet_idx(p: int, b: int, k: int) -> int:
    return BUL_OFF + (p * C.N_BULLETS + b) * BUL_STRIDE + k


BUL_ACT_IDX = np.array([bullet_idx(p, b, 2) for p in range(C.N_PLAYERS) for b in range(C.N_BULLETS)])
BUL_POS_IDX = np.array([bullet_idx(p, b, k) for p in range(C.N_PLAYERS) for b in range(C.N_BULLETS) for k in (0, 1)])
SHIP_POS_IDX = np.array([ship_idx(p, k) for p in range(C.N_PLAYERS) for k in (0, 1)])
BIN_IDX = np.concatenate([BUL_ACT_IDX, [FORM_OFF + 2], np.arange(INV_OFF, STATE_DIM)])
CONT_IDX = np.array(sorted(set(range(STATE_DIM)) - set(BIN_IDX.tolist())))
N_CONT = len(CONT_IDX)
N_BIN = len(BIN_IDX)

# Lower / upper bounds for every continuous entry (used to clamp neural output).
CONT_LO = np.zeros(STATE_DIM, dtype=np.float32)
CONT_HI = np.ones(STATE_DIM, dtype=np.float32)
for _p in range(C.N_PLAYERS):
    CONT_LO[ship_idx(_p, 0)], CONT_HI[ship_idx(_p, 0)] = C.SHIP_X_MIN, C.SHIP_X_MAX
    CONT_LO[ship_idx(_p, 1)], CONT_HI[ship_idx(_p, 1)] = C.SHIP_Y_MIN, C.SHIP_Y_MAX
    CONT_LO[ship_idx(_p, 2)], CONT_HI[ship_idx(_p, 2)] = -C.SHIP_MAX_SPEED, C.SHIP_MAX_SPEED
    CONT_LO[ship_idx(_p, 3)], CONT_HI[ship_idx(_p, 3)] = -C.SHIP_MAX_SPEED, C.SHIP_MAX_SPEED
CONT_HI[FORM_OFF + 0] = C.FORM_OX_MAX
CONT_HI[FORM_OFF + 1] = C.FORM_OY_MAX


# ---- views -------------------------------------------------------------
def ships(s: np.ndarray) -> np.ndarray:
    """(B, N_PLAYERS, 5) writable view."""
    return s[:, SHIP_OFF:BUL_OFF].reshape(s.shape[0], C.N_PLAYERS, SHIP_STRIDE)


def bullets(s: np.ndarray) -> np.ndarray:
    """(B, N_PLAYERS, N_BULLETS, 3) writable view."""
    return s[:, BUL_OFF:FORM_OFF].reshape(s.shape[0], C.N_PLAYERS, C.N_BULLETS, BUL_STRIDE)


def formation(s: np.ndarray) -> np.ndarray:
    """(B, 3) writable view: ox, oy, moving_right."""
    return s[:, FORM_OFF:INV_OFF]


def invaders(s: np.ndarray) -> np.ndarray:
    """(B, N_INVADERS) writable view of alive flags."""
    return s[:, INV_OFF:STATE_DIM]


def invader_positions(form: np.ndarray) -> np.ndarray:
    """(B, N_INVADERS, 2) centre positions derived from formation offsets."""
    rows, cols = np.divmod(np.arange(C.N_INVADERS), C.INV_COLS)
    x = C.INV_X0 + form[:, 0:1] + cols[None, :] * C.INV_DX
    y = C.INV_Y0 + form[:, 1:2] + rows[None, :] * C.INV_DY
    return np.stack([x, y], axis=-1)


# ---- constructors ------------------------------------------------------
def initial_state(B: int = 1) -> np.ndarray:
    """Canonical match start: ships side by side, fresh invader wave."""
    s = np.zeros((B, STATE_DIM), dtype=np.float32)
    sh = ships(s)
    sh[:, 0, 0], sh[:, 0, 1] = 0.3, 0.9
    sh[:, 1, 0], sh[:, 1, 1] = 0.7, 0.9
    formation(s)[:, 2] = 1.0
    invaders(s)[:] = 1.0
    return s


def random_state(B: int, rng: np.random.Generator) -> np.ndarray:
    """Broad random states used to cover the state space during training."""
    s = np.zeros((B, STATE_DIM), dtype=np.float32)
    sh = ships(s)
    sh[..., 0] = rng.uniform(C.SHIP_X_MIN, C.SHIP_X_MAX, (B, C.N_PLAYERS))
    sh[..., 1] = rng.uniform(C.SHIP_Y_MIN, C.SHIP_Y_MAX, (B, C.N_PLAYERS))
    sh[..., 2:4] = rng.uniform(-0.6, 0.6, (B, C.N_PLAYERS, 2))
    sh[..., 4] = np.where(rng.random((B, C.N_PLAYERS)) < 0.5, 0.0, rng.uniform(0, 1, (B, C.N_PLAYERS)))
    sh[..., 4] = np.round(sh[..., 4] / C.FIRE_COOLDOWN_DECAY) * C.FIRE_COOLDOWN_DECAY
    # Occasionally start ships touching to exercise the collision rule.
    touch = rng.random(B) < 0.1
    sh[touch, 1, 0] = np.clip(sh[touch, 0, 0] + rng.uniform(-0.08, 0.08, touch.sum()), C.SHIP_X_MIN, C.SHIP_X_MAX)
    sh[touch, 1, 1] = np.clip(sh[touch, 0, 1] + rng.uniform(-0.08, 0.08, touch.sum()), C.SHIP_Y_MIN, C.SHIP_Y_MAX)

    bu = bullets(s)
    act = rng.random((B, C.N_PLAYERS, C.N_BULLETS)) < 0.4
    bu[..., 0] = np.where(act, rng.uniform(C.SHIP_X_MIN, C.SHIP_X_MAX, act.shape), 0.0)
    bu[..., 1] = np.where(act, rng.uniform(0.0, C.SHIP_Y_MAX, act.shape), 0.0)
    bu[..., 2] = act.astype(np.float32)

    fo = formation(s)
    fo[:, 0] = rng.uniform(0, C.FORM_OX_MAX, B)
    fo[:, 1] = rng.integers(0, int(round(C.FORM_OY_MAX / C.FORM_STEP_DOWN)) + 1, B) * C.FORM_STEP_DOWN
    fo[:, 2] = (rng.random(B) < 0.5).astype(np.float32)

    inv = invaders(s)
    p_alive = rng.uniform(0.3, 1.0, (B, 1))
    inv[:] = (rng.random((B, C.N_INVADERS)) < p_alive).astype(np.float32)
    fresh = rng.random(B) < 0.15
    inv[fresh] = 1.0
    return s
