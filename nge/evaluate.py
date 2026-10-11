"""Compare neural vs classic rollouts from identical starts with identical inputs.

    python -m nge.evaluate --ckpt checkpoints/world_model.pt --worlds 512 --ticks 300

Reports, per horizon, the mean absolute ship-position error, the fraction of
invader/bullet flags that agree and the ship speed error.  This is the
"how far does the neural world drift from ideal physics" number.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import config as C
from . import data, sim
from . import state as S
from .model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/world_model.pt")
    ap.add_argument("--worlds", type=int, default=512)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    model = load_model(args.ckpt, args.device)

    s_c = S.initial_state(args.worlds)
    s_c[args.worlds // 2:] = S.random_state(args.worlds - args.worlds // 2, rng)
    s_n = torch.as_tensor(s_c, device=args.device)
    pol = data.PolicyBank(args.worlds, rng)
    horizons = {1, 2, 5, 10, 30, 60, 90, 150, 300}
    inv_idx = torch.arange(S.INV_OFF, S.STATE_DIM, device=args.device)
    act_idx = torch.as_tensor(S.BUL_ACT_IDX, device=args.device)
    pos_idx = torch.as_tensor(S.SHIP_POS_IDX, device=args.device)
    vel_idx = torch.as_tensor([S.ship_idx(p, k) for p in range(C.N_PLAYERS) for k in (2, 3)], device=args.device)

    # one-step accuracy along the classic trajectory (teacher forced)
    one_step_pos = []; one_step_flag = []
    print(f"{'tick':>5} | {'ship pos MAE':>12} | {'ship vel MAE':>12} | {'invader flag agree':>18} | {'bullet flag agree':>17}")
    for t in range(1, args.ticks + 1):
        u = pol.act(s_c)
        ut = torch.as_tensor(u, device=args.device)
        s_c_next = sim.step(s_c, u)
        with torch.no_grad():
            tf = model.predict_state(torch.cat([torch.as_tensor(s_c, device=args.device), ut], 1))
            s_n = model.predict_state(torch.cat([s_n, ut], 1))
        truth = torch.as_tensor(s_c_next, device=args.device)
        one_step_pos.append((tf[:, pos_idx] - truth[:, pos_idx]).abs().mean().item())
        one_step_flag.append((tf[:, inv_idx] == truth[:, inv_idx]).float().mean().item())
        s_c = s_c_next
        if t in horizons:
            pe = (s_n[:, pos_idx] - truth[:, pos_idx]).abs().mean().item()
            ve = (s_n[:, vel_idx] - truth[:, vel_idx]).abs().mean().item()
            fa = (s_n[:, inv_idx] == truth[:, inv_idx]).float().mean().item()
            ba = (s_n[:, act_idx] == truth[:, act_idx]).float().mean().item()
            print(f"{t:5d} | {pe:12.5f} | {ve:12.5f} | {fa:18.4f} | {ba:17.4f}")
    print(f"\nteacher-forced one-step: ship pos MAE {np.mean(one_step_pos):.5f} (ship radius {C.SHIP_RADIUS}), "
          f"invader flag agreement {np.mean(one_step_flag):.5f}")


if __name__ == "__main__":
    main()
