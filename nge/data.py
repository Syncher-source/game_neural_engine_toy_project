"""Training-data generation: roll the classic sim over many worlds with a mix
of input policies and record (state, input, next_state) transitions.

Policies are chosen per world and per player so the data covers idle ships,
twitchy random pilots, continuous stick values and deliberate collision seekers.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import sim
from . import state as S

POLICY_RANDOM, POLICY_CHASER, POLICY_IDLE, POLICY_ANALOG = 0, 1, 2, 3


class PolicyBank:
    """Vectorised input policies for B worlds x N_PLAYERS players."""

    def __init__(self, B: int, rng: np.random.Generator):
        self.rng = rng
        self.B = B
        self.kind = rng.choice([POLICY_RANDOM, POLICY_CHASER, POLICY_IDLE, POLICY_ANALOG],
                               size=(B, C.N_PLAYERS), p=[0.45, 0.25, 0.1, 0.2])
        self.cur = np.zeros((B, C.N_PLAYERS, 3), dtype=np.float32)
        self.change_p = rng.uniform(0.03, 0.3, (B, C.N_PLAYERS))
        self.fire_p = rng.uniform(0.0, 0.6, (B, C.N_PLAYERS))

    def act(self, s: np.ndarray) -> np.ndarray:
        rng, B = self.rng, self.B
        sh = S.ships(s)
        resample = rng.random((B, C.N_PLAYERS)) < self.change_p
        disc = rng.integers(-1, 2, (B, C.N_PLAYERS, 2)).astype(np.float32)
        ana = rng.uniform(-1, 1, (B, C.N_PLAYERS, 2)).astype(np.float32)
        new_dir = np.where((self.kind == POLICY_ANALOG)[..., None], ana, disc)
        self.cur[..., 0:2] = np.where(resample[..., None], new_dir, self.cur[..., 0:2])
        # chasers steer toward the other ship (with some noise) -> collisions
        other = sh[:, ::-1, 0:2]
        delta = other - sh[..., 0:2]
        chase = np.sign(delta) + rng.normal(0, 0.3, delta.shape)
        is_chaser = (self.kind == POLICY_CHASER)[..., None]
        self.cur[..., 0:2] = np.where(is_chaser, np.clip(chase, -1, 1), self.cur[..., 0:2])
        self.cur[..., 2] = (rng.random((B, C.N_PLAYERS)) < self.fire_p).astype(np.float32)
        idle = (self.kind == POLICY_IDLE)[..., None]
        self.cur[:] = np.where(idle, 0.0, self.cur)
        return self.cur.reshape(B, S.INPUT_DIM).copy()


def generate(n_samples: int, worlds: int = 4096, episode: int = 200, seed: int = 0, verbose: bool = True, events: bool = False):
    """Returns (states, inputs, next_states) float32 arrays with n_samples rows; with events=True
    also (hit (n, N_PLAYERS*N_BULLETS) int64, reset (n,) bool), see sim.step_events."""
    rng = np.random.default_rng(seed)
    out_s, out_u, out_n, out_h, out_r = [], [], [], [], []
    have = 0
    while have < n_samples:
        from_start = rng.random(worlds) < 0.3
        s = S.random_state(worlds, rng)
        s[from_start] = S.initial_state(int(from_start.sum()))
        pol = PolicyBank(worlds, rng)
        for _ in range(episode):
            u = pol.act(s)
            n, hit, reset = sim.step_events(s, u)
            out_s.append(s); out_u.append(u); out_n.append(n); out_h.append(hit); out_r.append(reset)
            s = n
            have += worlds
            if have >= n_samples:
                break
        if verbose:
            print(f"  generated {min(have, n_samples):,}/{n_samples:,} transitions", flush=True)
    st = np.concatenate(out_s)[:n_samples]
    ut = np.concatenate(out_u)[:n_samples]
    nt = np.concatenate(out_n)[:n_samples]
    if events:
        return st, ut, nt, np.concatenate(out_h)[:n_samples], np.concatenate(out_r)[:n_samples]
    return st, ut, nt
