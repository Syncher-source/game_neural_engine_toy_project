"""Train the neural renderer against the sprite compositor (or the old flat look).

    python -m nge.train_renderer --steps 12000 --batch 32 --res 720 --crop 160

States are sampled from a pool of simulated states (same distribution as the world-model
training data) and rasterized on the fly, so the data is unlimited. At high resolution the
network is trained on random crops centred on objects most of the time; being (almost) per-pixel
it renders the full frame at inference. Object and laser pixels are up-weighted in the loss.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import data
from . import state as S
from .renderer import NeuralRenderer, save_renderer


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def save_preview(gt: torch.Tensor, pred: torch.Tensor, path: str) -> None:
    from PIL import Image
    g = (gt.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()      # (N, H, W, 3)
    p = (pred.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    row_gt = np.concatenate(list(g), axis=1)
    row_pr = np.concatenate(list(p), axis=1)
    sheet = np.concatenate([row_gt, np.full((4, row_gt.shape[1], 3), 255, np.uint8), row_pr], axis=0)
    Image.fromarray(sheet).save(path)


def sample_windows(s: torch.Tensor, H: int, W: int, crop: int, p_obj: float, raster):
    """One crop per batch item. With probability p_obj it is centred on a random visible object
    (positives); otherwise, half of the time, on the same column or row as an object but at a
    random distance (negatives: the network must learn that a ship's x alone does not make a
    ship), and the rest uniformly over the frame (background cleanliness)."""
    B = s.shape[0]
    centres, wts = raster.object_centres(s)                              # (B, N, 2), (B, N)
    pick = torch.multinomial(wts + 1e-6, 1)[:, 0]
    c = centres[torch.arange(B, device=s.device), pick]                  # (B, 2) world
    jitter = (torch.rand(B, 2, device=s.device) - 0.5) * (crop / H) * 0.8
    c = c + jitter
    uni = torch.rand(B, 2, device=s.device)
    col_or_row = torch.rand(B, device=s.device) < 0.5
    neg = torch.where(col_or_row[:, None], torch.stack([c[:, 0], uni[:, 1]], 1), torch.stack([uni[:, 0], c[:, 1]], 1))
    r = torch.rand(B, device=s.device)
    use_obj = r < p_obj
    use_neg = (r >= p_obj) & (r < p_obj + (1 - p_obj) / 2)
    c = torch.where(use_obj[:, None], c, torch.where(use_neg[:, None], neg, uni))
    x0 = (c[:, 0] * W - crop / 2).round().long().clamp(0, W - crop)
    y0 = (c[:, 1] * H - crop / 2).round().long().clamp(0, H - crop)
    return [(int(y), int(x), crop, crop) for y, x in zip(y0.tolist(), x0.tolist())]


def band_mask(s: torch.Tensor, H: int, W: int, window, raster) -> torch.Tensor:
    """Hard negatives for one crop: pixels that share a row or a column with a visible ship or
    laser within the relative-coordinate window (|rx| < 2 or |ry| < 2), i.e. where the network
    is tempted to draw ghosts if it keys on one coordinate alone. (1, h, w) float."""
    from .renderer import NeuralRenderer as NR
    X, Y = raster.grid(H, W, s.device, window)
    band = torch.zeros_like(X, dtype=torch.bool)
    for k, (ix, iy) in enumerate(NR.OBJ_XY):
        vis = True if k < 2 else bool(s[0, ix + 2] > 0.5)           # ships always, bullets when active
        if not vis: continue
        r = 2.0 * NR.OBJ_SCALE[k]
        band |= ((X - s[0, ix]).abs() < r) | ((Y - s[0, iy]).abs() < r)
    return band.float()[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--res", type=int, default=720)
    ap.add_argument("--crop", type=int, default=160, help="training crop size in px (0 = full frames)")
    ap.add_argument("--p-obj", type=float, default=0.7, help="probability that a crop is centred on an object (rest: negatives / uniform)")
    ap.add_argument("--look", choices=["sprites", "classic"], default="sprites")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-freq", type=int, default=8, help="coordinate Fourier features (higher = sharper small shapes)")
    ap.add_argument("--no-rel-coords", action="store_true", help="fully implicit renderer (no per-object relative coordinates)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pool", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="start from this renderer checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/renderer.pt")
    ap.add_argument("--preview", default="checkpoints/renderer_preview.png")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    H = W = args.res
    if args.look == "sprites":
        from . import raster_sprites as raster
    else:
        from . import raster as raster_classic

        class raster:  # adapter: same interface as raster_sprites
            BG = raster_classic.BG

            @staticmethod
            def rasterize(s, H, W, window=None):
                img = raster_classic.rasterize(s, H, W)
                if window is not None:
                    y0, x0, h, w = window; img = img[:, y0:y0 + h, x0:x0 + w]
                bg = torch.tensor(raster_classic.BG, device=s.device)
                return img, ((img - bg).abs().sum(-1) > 1e-3).float(), (img.min(-1).values > 0.99).float()

            @staticmethod
            def object_centres(s):
                from . import raster_sprites
                return raster_sprites.object_centres(s)

    print("building state pool with the classic sim ...")
    st, _, _ = data.generate(args.pool, worlds=2048, episode=100, seed=args.seed, verbose=False)
    rng = np.random.default_rng(args.seed)
    extra = S.random_state(args.pool // 4, rng)
    pool = torch.as_tensor(np.concatenate([st, extra]), device=dev)
    s_mean, s_std = pool.mean(0), pool.std(0) + 1e-3
    val = pool[-64:]; pool = pool[:-64]
    val_img, val_obj, val_las = [torch.cat(t) for t in zip(*[raster.rasterize(val[i:i + 8], H, W) for i in range(0, len(val), 8)])]

    model = NeuralRenderer(H, W, args.hidden, args.n_freq, s_mean=s_mean, s_std=s_std, rel_coords=not args.no_rel_coords).to(dev)
    if args.init:
        from .renderer import load_renderer
        src = load_renderer(args.init, dev).net.state_dict(); dst = model.net.state_dict()
        for k, v in src.items():
            if dst[k].shape == v.shape: dst[k] = v
            elif k == "0.weight":                                   # feature set grew: old channels keep their weights, new ones start at 0
                dst[k].zero_(); dst[k][:, : v.shape[1]] = v
        model.net.load_state_dict(dst); print(f"initialised from {args.init}")
    print(f"renderer: {H}x{W}, crop {args.crop or 'full'}, look {args.look}, hidden {args.hidden}, params {sum(p.numel() for p in model.parameters()):,}, device {dev}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(pool), (args.batch,), device=dev)
        s = pool[idx]
        if args.crop:
            wins = sample_windows(s, H, W, args.crop, args.p_obj, raster)
            gts = [raster.rasterize(s[i:i + 1], H, W, wins[i]) for i in range(args.batch)]
            gt = torch.cat([g[0] for g in gts]); obj = torch.cat([g[1] for g in gts]); las = torch.cat([g[2] for g in gts])
            band = torch.cat([band_mask(s[i:i + 1], H, W, wins[i], raster) for i in range(args.batch)])
            pred = torch.cat([model(s[i:i + 1], wins[i]) for i in range(args.batch)])
        else:
            gt, obj, las = raster.rasterize(s, H, W)
            band = torch.cat([band_mask(s[i:i + 1], H, W, None, raster) for i in range(args.batch)])
            pred = model(s)
        # object / laser pixels up-weighted; the row/column bands around ships and lasers are the
        # hard negatives (ghost bolts and streaks appear there if they are cheap to get wrong)
        w = (1.0 + 8.0 * band + 15.0 * obj + 40.0 * (las > 0.1).float())[..., None]
        loss = (w * (pred - gt) ** 2).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                pv = torch.cat([model(val[i:i + 4]) for i in range(0, len(val), 4)])
                ps = psnr(pv, val_img)
                ok = (pv - val_img).abs().max(-1).values < 0.1
                acc = ok[val_obj > 0.5].float().mean().item()
                bacc = ok[val_las > 0.5].float().mean().item() if (val_las > 0.5).any() else float("nan")
            model.train()
            print(f"step {step:5d} | loss {loss.item():.5f} | val PSNR {ps:.2f} dB | object px within 0.1: {acc:.3f} | laser px within 0.1: {bacc:.3f} | {time.time() - t0:.0f}s", flush=True)

    model.eval()
    save_renderer(model, args.out, extra={"steps": args.steps, "look": args.look})
    with torch.no_grad():
        save_preview(val_img[:3], torch.cat([model(val[i:i + 1]) for i in range(3)]), args.preview)
    print(f"saved {args.out} and {args.preview}")


if __name__ == "__main__":
    main()
