"""Rule-violation meter: how often does a world model break the game's rules?

    python -m nge.rules [--ckpt checkpoints/world_model.pt] [--worlds 512] [--ticks 600] [--classic]

The world model is rolled out closed loop (its own outputs fed back) with the training input
policies, and every transition (s -> s') is checked against rules that the teacher obeys by
construction. Each rule is counted as violations per 1000 ticks (and, where it makes sense, as
a fraction of the events it concerns). The classic sim must score 0 everywhere.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import config as C
from . import data, sim
from . import state as S

RULES = [
    ("multi_kill", "ticks where more invaders died than bullets were consumed by hits", "per 1000 ticks"),
    ("ghost_kill", "invaders that died with no active bullet inside their hit box", "per 1000 kills"),
    ("vanish", "bullets that disappeared on screen without a kill (not off the top)", "per 1000 bullet-ticks"),
    ("resurrection", "invaders that came back to life outside a wave reset", "per 1000 ticks"),
    ("spawn_no_fire", "bullets that appeared without fire pressed and cooldown expired", "per 1000 spawns"),
    ("spawn_misplaced", "bullets that spawned further than 0.02 from their ship", "per 1000 spawns"),
    ("overlap", "ticks with the ships interpenetrating by more than 0.005", "per 1000 ticks"),
    ("no_stop", "ticks with the ships touching but not both stopped (|v| > 0.02, 2.5% of max speed)", "per 1000 ticks"),
    ("cooldown_jump", "cooldown increased by > 0.05 (half a decay step) without a spawn", "per 1000 ticks"),
    ("bullet_drift", "active bullets that moved sideways > 0.004 or not up by the bullet speed (+-0.01)", "per 1000 bullet-ticks"),
]


def check(s: np.ndarray, u: np.ndarray, n: np.ndarray) -> dict:
    """Violation counts and event counts for a batch of transitions. s, n: (B, STATE_DIM); u: (B, INPUT_DIM)."""
    B = s.shape[0]
    sh, shn = S.ships(s), S.ships(n)
    bu, bun = S.bullets(s), S.bullets(n)
    inv, invn = S.invaders(s), S.invaders(n)
    ipos = S.invader_positions(S.formation(s))                        # (B, NI, 2)
    inp = u.reshape(B, C.N_PLAYERS, 3)
    out = {}
    act, actn = bu[..., 2] > 0.5, bun[..., 2] > 0.5
    alive, aliven = inv > 0.5, invn > 0.5
    killed = alive & ~aliven                                           # (B, NI)
    reset = (~aliven).sum(1) == 0                                      # all alive after: possible wave reset
    reset &= (killed.sum(1) + (~alive).sum(1)) >= C.N_INVADERS - 1    # ... and (almost) everything was dead or died
    # a bullet slot can be consumed (hit or off screen) and re-fired in the same tick: detect that as
    # a jump back down to the ship; such a slot counts as both consumed and spawned
    moved_y = bu[..., 1] - C.BULLET_SPEED * C.DT
    respawned = act & actn & (bun[..., 1] - moved_y > 0.05)
    consumed = act & (~actn | respawned) & (moved_y >= 0.0)
    out["multi_kill"] = (np.maximum(killed.sum(1) - consumed.sum((1, 2)), 0) > 0).sum(), B
    # ghost kill: killed invader with no active bullet in its hit box (margin for one tick of motion)
    bx = bu[..., 0].reshape(B, -1)[:, :, None]; by = moved_y.reshape(B, -1)[:, :, None]
    near = act.reshape(B, -1)[:, :, None] & (np.abs(bx - ipos[:, None, :, 0]) < C.INV_HALF_W + 0.01) & (np.abs(by - ipos[:, None, :, 1]) < C.INV_HALF_H + 0.02)
    out["ghost_kill"] = (killed & ~near.any(1)).sum(), max(killed.sum(), 1)
    # vanish: consumed but no kill anywhere near it
    hit_near = act.reshape(B, -1)[:, :, None] & killed[:, None, :] & (np.abs(bx - ipos[:, None, :, 0]) < C.INV_HALF_W + 0.01) & (np.abs(by - ipos[:, None, :, 1]) < C.INV_HALF_H + 0.02)
    out["vanish"] = (consumed.reshape(B, -1) & ~hit_near.any(2) & ~reset[:, None]).sum(), max(act.sum(), 1)
    out["resurrection"] = ((~alive & aliven) & ~reset[:, None]).sum(), B
    # spawns: fire pressed and the cooldown (which decays first) expired; the new bullet sits at the
    # ship's position after this tick's movement
    spawned = (~act & actn) | respawned
    fire_ok = (inp[..., 2] > 0.5) & (sh[..., 4] - C.FIRE_COOLDOWN_DECAY <= 1e-6)      # (B, P)
    out["spawn_no_fire"] = (spawned & ~fire_ok[:, :, None]).sum(), max(spawned.sum(), 1)
    d = np.hypot(bun[..., 0] - shn[..., 0:1], bun[..., 1] - (shn[..., 1:2] - C.SHIP_RADIUS))
    out["spawn_misplaced"] = (spawned & (d > 0.02)).sum(), max(spawned.sum(), 1)
    # ships (the wall clamp after a collision may leave them touching or overlapping: excluded)
    dist = np.hypot(shn[:, 0, 0] - shn[:, 1, 0], shn[:, 0, 1] - shn[:, 1, 1])
    at_wall = ((shn[..., 0] <= C.SHIP_X_MIN + 1e-5) | (shn[..., 0] >= C.SHIP_X_MAX - 1e-5) |
               (shn[..., 1] <= C.SHIP_Y_MIN + 1e-5) | (shn[..., 1] >= C.SHIP_Y_MAX - 1e-5)).any(1)
    touching = (dist < 2 * C.SHIP_RADIUS - 1e-4) & ~at_wall
    stopped = (np.abs(shn[..., 2:4]) < 0.02).all((1, 2))              # regression noise is not a violation
    out["overlap"] = ((dist < 2 * C.SHIP_RADIUS - 0.005) & ~at_wall).sum(), B
    out["no_stop"] = (touching & ~stopped).sum(), B
    cd_up = (shn[..., 4] > sh[..., 4] + 0.05) & ~spawned.any(2)
    out["cooldown_jump"] = cd_up.sum(), B
    both = act & actn & ~respawned
    drift = both & ((np.abs(bun[..., 0] - bu[..., 0]) > 0.004) | (np.abs(bun[..., 1] - moved_y) > 0.01))
    out["bullet_drift"] = drift.sum(), max(both.sum(), 1)
    return out


def rollout_violations(step_fn, worlds: int, ticks: int, seed: int, from_start_frac: float = 0.5):
    rng = np.random.default_rng(seed)
    s = S.random_state(worlds, rng)
    k = int(worlds * from_start_frac)
    s[:k] = S.initial_state(k)
    pol = data.PolicyBank(worlds, rng)
    tot = {r[0]: [0, 0] for r in RULES}
    for _ in range(ticks):
        u = pol.act(s)
        n = step_fn(s, u)
        for key, (v, e) in check(s, u, n).items():
            tot[key][0] += int(v); tot[key][1] += int(e)
        s = n
    return tot


def report(tot: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'rule':16s} {'violations':>10s} {'events':>10s}   rate")
    for key, desc, unit in RULES:
        v, e = tot[key]
        print(f"  {key:16s} {v:10d} {e:10d}   {1000.0 * v / max(e, 1):8.2f} {unit}   ({desc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/world_model.pt")
    ap.add_argument("--worlds", type=int, default=512)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--classic", action="store_true", help="measure the reference sim (must be all zeros)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.classic:
        report(rollout_violations(sim.step, args.worlds, args.ticks, args.seed), f"classic sim, {args.worlds} worlds x {args.ticks} ticks")
        return
    from .model import load_model
    m = load_model(args.ckpt, args.device)

    @torch.no_grad()
    def step_fn(s, u):
        x = torch.as_tensor(S.pack(s, u), device=args.device)
        return m.predict_state(x).cpu().numpy()

    report(rollout_violations(step_fn, args.worlds, args.ticks, args.seed), f"{args.ckpt} closed loop, {args.worlds} worlds x {args.ticks} ticks")


if __name__ == "__main__":
    main()
