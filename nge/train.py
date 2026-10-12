"""Train the neural world model on classic-sim transitions.

    python -m nge.train --samples 3000000 --epochs 30

The whole dataset lives on the GPU; each epoch shuffles indices and runs
mini-batches.  Loss = MSE on normalised continuous deltas (bullet positions are
masked out when the bullet is inactive next tick, since they are zeroed anyway)
+ BCE on the binary flags.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from . import config as C
from . import state as S
from .model import (WorldModel, save_checkpoint, CONT_IDX_T, BIN_IDX_T, BUL_ACT_T,
                    HEAD_ACT, HEAD_DIR, HEAD_RESET, HEAD_HIT, N_BUL, N_HIT, FORM_DIR_IDX)


def build_masks(nxt: torch.Tensor) -> torch.Tensor:
    """(B, N_CONT) weight mask: 0 for x,y of bullets inactive at t+1."""
    mask = torch.ones(nxt.shape[0], S.N_CONT, device=nxt.device)
    cont_pos = {int(i): k for k, i in enumerate(S.CONT_IDX)}
    act = nxt[:, BUL_ACT_T.to(nxt.device)]
    for b, (ix, iy) in enumerate(S.BUL_POS_IDX.reshape(-1, 2)):
        mask[:, cont_pos[int(ix)]] = act[:, b]
        mask[:, cont_pos[int(iy)]] = act[:, b]
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3_000_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bin-weight", type=float, default=1.0)
    ap.add_argument("--hit-weight", type=float, default=8.0, help="class weight of the 'hit invader i' outcomes vs 'none' in the hit head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/world_model.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    print(f"device={dev} ({torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'cpu'})")

    t0 = time.time()
    print("generating data with the classic sim ...")
    st, ut, nt, ht, rt = data.generate(args.samples, seed=args.seed, events=True)
    print(f"  done in {time.time() - t0:.1f}s   (bullet-ticks with a hit: {(ht < C.N_INVADERS).sum():,}, wave resets: {int(rt.sum()):,})")

    # normalisation statistics
    s_mean = st.mean(0); s_std = st.std(0) + 1e-3
    d = (nt - st)[:, S.CONT_IDX]
    d_std = d.std(0) + 1e-4

    n_val = min(200_000, len(st) // 10)
    S_tr = torch.as_tensor(st[:-n_val], device=dev); U_tr = torch.as_tensor(ut[:-n_val], device=dev); N_tr = torch.as_tensor(nt[:-n_val], device=dev)
    S_va = torch.as_tensor(st[-n_val:], device=dev); U_va = torch.as_tensor(ut[-n_val:], device=dev); N_va = torch.as_tensor(nt[-n_val:], device=dev)
    H_tr = torch.as_tensor(ht[:-n_val], device=dev); R_tr = torch.as_tensor(rt[:-n_val], device=dev, dtype=torch.float32)
    H_va = torch.as_tensor(ht[-n_val:], device=dev); R_va = torch.as_tensor(rt[-n_val:], device=dev, dtype=torch.float32)
    del st, ut, nt, ht, rt

    model = WorldModel(args.hidden, args.layers, s_mean, s_std, d_std).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: hidden={args.hidden} layers={args.layers} params={n_par:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * (len(S_tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    cont_idx = CONT_IDX_T.to(dev); bin_idx = BIN_IDX_T.to(dev)

    act_idx = BUL_ACT_T.to(dev)
    hit_w = torch.ones(N_HIT, device=dev); hit_w[: C.N_INVADERS] = args.hit_weight     # hits are ~1% of bullet-ticks

    def losses(s, u, n, h, r):
        d_pred, logits = model(torch.cat([s, u], 1))
        d_true = (n[:, cont_idx] - s[:, cont_idx]) / model.d_std
        mask = build_masks(n)
        l_cont = ((d_pred - d_true) ** 2 * mask).sum() / mask.sum()
        l_act = F.binary_cross_entropy_with_logits(logits[:, HEAD_ACT:HEAD_ACT + N_BUL], n[:, act_idx])
        l_dir = F.binary_cross_entropy_with_logits(logits[:, HEAD_DIR], n[:, FORM_DIR_IDX])
        l_reset = F.binary_cross_entropy_with_logits(logits[:, HEAD_RESET], r)
        hit_l = logits[:, HEAD_HIT:].reshape(-1, N_HIT)
        l_hit = F.cross_entropy(hit_l, h.reshape(-1), weight=hit_w)
        l_bin = l_act + 0.2 * l_dir + 0.2 * l_reset + l_hit
        return l_cont, l_bin

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(S_tr), device=dev)
        tot_c = tot_b = 0.0; nb = 0
        te = time.time()
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            l_c, l_b = losses(S_tr[idx], U_tr[idx], N_tr[idx], H_tr[idx], R_tr[idx])
            loss = l_c + args.bin_weight * l_b
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot_c += l_c.item(); tot_b += l_b.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vc, vb = losses(S_va, U_va, N_va, H_va, R_va)
            xva = torch.cat([S_va, U_va], 1)
            pred = model.predict_state(xva)
            flag_acc = (pred[:, bin_idx] == N_va[:, bin_idx]).float().mean().item()
            changed = N_va[:, bin_idx] != S_va[:, bin_idx]
            trans_acc = (pred[:, bin_idx] == N_va[:, bin_idx])[changed].float().mean().item() if changed.any() else float('nan')
            _, logits = model(xva)
            choice = model.events(S_va, logits)[3]
            is_hit_t = H_va < C.N_INVADERS
            hit_rec = (choice[is_hit_t] == H_va[is_hit_t]).float().mean().item()
            hit_prec = (choice[choice < C.N_INVADERS] == H_va[choice < C.N_INVADERS]).float().mean().item() if (choice < C.N_INVADERS).any() else float('nan')
            ship_err = (pred[:, S.SHIP_POS_IDX] - N_va[:, S.SHIP_POS_IDX]).abs().mean().item()
        print(f"ep {ep + 1:3d}/{args.epochs} | train cont {tot_c / nb:.5f} ev {tot_b / nb:.5f} | "
              f"val cont {vc.item():.5f} ev {vb.item():.5f} | flag acc {flag_acc:.5f} transition acc {trans_acc:.4f} | "
              f"hit recall {hit_rec:.3f} precision {hit_prec:.3f} | ship pos MAE {ship_err:.5f} | {time.time() - te:.1f}s", flush=True)

    save_checkpoint(model, args.out, extra={"samples": args.samples, "epochs": args.epochs})
    print(f"saved {args.out}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
