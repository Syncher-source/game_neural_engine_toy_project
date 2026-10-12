"""Closed-loop fine-tuning of the world model on its own rollouts.

    python -m nge.finetune --ckpt checkpoints/world_model.pt --rounds 4

Each round rolls the model out closed loop (its outputs fed back, training input policies),
collects the states it actually visits, asks the teacher (classic sim) for the one-step target
and events at those states, and trains on them mixed 1:1 with ordinary teacher data. The model
thereby learns to come back from its own mistakes instead of only from teacher states. Rule
violations (nge.rules) are reported before and after.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import data, rules, sim
from . import state as S
from .model import (BUL_ACT_T, CONT_IDX_T, FORM_DIR_IDX, HEAD_ACT, HEAD_DIR, HEAD_HIT, HEAD_RESET, N_BUL, N_HIT, load_model, save_checkpoint)
from .train import build_masks


def make_loss(model, dev, hit_weight):
    cont_idx = CONT_IDX_T.to(dev); act_idx = BUL_ACT_T.to(dev)
    hit_w = torch.ones(N_HIT, device=dev); hit_w[: C.N_INVADERS] = hit_weight

    def losses(s, u, n, h, r):
        d_pred, logits = model(torch.cat([s, u], 1))
        d_true = (n[:, cont_idx] - s[:, cont_idx]) / model.d_std
        mask = build_masks(n)
        l_cont = ((d_pred - d_true) ** 2 * mask).sum() / mask.sum()
        l_act = F.binary_cross_entropy_with_logits(logits[:, HEAD_ACT:HEAD_ACT + N_BUL], n[:, act_idx])
        l_dir = F.binary_cross_entropy_with_logits(logits[:, HEAD_DIR], n[:, FORM_DIR_IDX])
        l_reset = F.binary_cross_entropy_with_logits(logits[:, HEAD_RESET], r)
        l_hit = F.cross_entropy(logits[:, HEAD_HIT:].reshape(-1, N_HIT), h.reshape(-1), weight=hit_w)
        return l_cont + l_act + 0.2 * l_dir + 0.2 * l_reset + l_hit
    return losses


@torch.no_grad()
def on_policy(model, dev, worlds, ticks, seed):
    """Roll the model closed loop; return (s, u) visited and the teacher's (n, hit, reset) at them."""
    rng = np.random.default_rng(seed)
    s = S.random_state(worlds, rng); k = worlds // 2; s[:k] = S.initial_state(k)
    pol = data.PolicyBank(worlds, rng)
    S_, U_, N_, H_, R_ = [], [], [], [], []
    for _ in range(ticks):
        u = pol.act(s)
        n_t, h, r = sim.step_events(s, u)              # teacher target at the model's own state
        S_.append(s); U_.append(u); N_.append(n_t); H_.append(h); R_.append(r)
        s = model.predict_state(torch.as_tensor(S.pack(s, u), device=dev)).cpu().numpy()
    return [np.concatenate(a) for a in (S_, U_, N_, H_, R_)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/world_model.pt")
    ap.add_argument("--out", default="checkpoints/world_model.pt")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-5, help="peak learning rate; cosine-decayed to zero within each round")
    ap.add_argument("--teacher-ratio", type=float, default=3.0, help="teacher samples per on-policy sample in each round")
    ap.add_argument("--hit-weight", type=float, default=8.0)
    ap.add_argument("--pool", type=int, default=2_000_000, help="teacher transitions mixed in with the on-policy data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    model = load_model(args.ckpt, dev)
    losses = make_loss(model, dev, args.hit_weight)

    def step_fn(s, u):
        with torch.no_grad():
            return model.predict_state(torch.as_tensor(S.pack(s, u), device=dev)).cpu().numpy()

    baseline = {}

    def violations(tag):
        tot = rules.rollout_violations(step_fn, 256, 300, seed=99)
        rates = {k: 1000.0 * v / max(e, 1) for k, (v, e) in tot.items()}
        if not baseline:
            baseline.update(rates)
        # violation index: mean ratio to the starting rates (1.0 = no change), the model-selection score.
        # The floor of 5 per 1000 keeps rules with tiny baselines (3 -> 8) from dominating real changes (142 -> 74).
        index = float(np.mean([(rates[k] + 5.0) / (baseline[k] + 5.0) for k in rates]))
        print(f"  [{tag}] index {index:.3f} | " + "  ".join(f"{k} {r:.2f}" for k, r in rates.items()), flush=True)
        return index

    print("teacher pool ..."); ps, pu, pn, ph, pr = data.generate(args.pool, seed=args.seed + 1, verbose=False, events=True)
    P = [torch.as_tensor(a, device=dev) for a in (ps, pu, pn)] + [torch.as_tensor(ph, device=dev), torch.as_tensor(pr, device=dev, dtype=torch.float32)]
    import copy
    best_index, best_state, best_round = violations("before"), copy.deepcopy(model.state_dict()), 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        model.eval()
        os_, ou, on, oh, orr = on_policy(model, dev, args.worlds, args.ticks, seed=args.seed + 100 + rnd)
        O = [torch.as_tensor(a, device=dev) for a in (os_, ou, on)] + [torch.as_tensor(oh, device=dev), torch.as_tensor(orr, device=dev, dtype=torch.float32)]
        n_on = len(os_); n_te = int(n_on * args.teacher_ratio)
        steps = args.epochs * ((n_on + n_te) // args.batch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
        for g in opt.param_groups: g["lr"] = args.lr
        model.train()
        tot = 0.0; nb = 0
        for ep in range(args.epochs):
            sel = torch.randperm(len(P[0]), device=dev)[:n_te]
            D = [torch.cat([o, p[sel]]) for o, p in zip(O, P)]
            perm = torch.randperm(len(D[0]), device=dev)
            for i in range(0, len(perm) - args.batch + 1, args.batch):
                idx = perm[i:i + args.batch]
                loss = losses(*[d[idx] for d in D])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                tot += loss.item(); nb += 1
        model.eval()
        print(f"round {rnd}/{args.rounds}: {n_on:,} on-policy + {n_te:,} teacher samples, {nb} steps, loss {tot / max(nb, 1):.4f}, {time.time() - t0:.0f}s")
        index = violations(f"after round {rnd}")
        if index < best_index:
            best_index, best_state, best_round = index, copy.deepcopy(model.state_dict()), rnd
    model.load_state_dict(best_state)
    save_checkpoint(model, args.out, extra={"finetune_rounds": best_round, "violation_index": best_index})
    print(f"saved {args.out} (round {best_round}, violation index {best_index:.3f} vs 1.000 before)")


if __name__ == "__main__":
    main()
