"""Local visual effects for the teacher rasterizer: invader explosions.

Nothing here is game state. An effect is triggered by an *observed* transition between two
consecutive rendered states (an invader's alive flag going 1 -> 0), lives only on the machine
that draws it, and advances with frame time, independent of the 30 Hz tick. Nothing of it is
simulated by the world model or sent over the wire. The neural renderer is meant to learn these
effects from its own local recurrent state; this module is the ground truth it is trained
against and, like `sprites.py`, never ships in the game binaries.

An explosion is fully described by (centre, age, invader row, seed). Every particle position is
an analytic function of the age, so the look is identical at any frame rate and no integration
state is needed.

    python -m nge.effects                                   # assets/sprites/explosion_sheet.png, explosion_frame.png, explosion.gif
    python -m nge.effects --student checkpoints/renderer.pt  # teacher vs renderer network: explosion_student_sheet.png, explosion_student.gif
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import state as S

MAX_EXPL = 8              # concurrent explosions per world; the oldest is recycled when full
DURATION = 0.7            # seconds until a slot is free again
N_SPARK = 18
N_CHUNK = 5
_BUDGET = 40_000_000      # max elements of one (K, N, h, w) temporary; larger batches are processed in slices

HOT = (1.0, 0.97, 0.85)
# per invader row (saucer, war machine, orb): spark colour, debris material, shock-ring colour
_PALETTE = [
    ((0.55, 1.00, 0.65), (0.80, 0.82, 0.88), (0.70, 1.00, 0.80)),
    ((1.00, 0.62, 0.30), (0.78, 0.45, 0.25), (1.00, 0.70, 0.40)),
    ((1.00, 0.82, 0.35), (0.42, 0.45, 0.55), (1.00, 0.45, 0.30)),
]


def _hash(seed: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """Deterministic pseudo-random numbers in [0, 1): seed (K, 1) x i (1, N) -> (K, N)."""
    return torch.frac(torch.sin(seed * 12.9898 + i * 78.233 + 1.0) * 43758.5453)


def _lerp(a, b, t):
    return a + (b - a) * t


class Effects:
    """Batched explosion state for B worlds. Tensors (B, MAX_EXPL, ...), all on one device."""

    def __init__(self, B: int, device, generator: torch.Generator | None = None, slots: int = MAX_EXPL):
        self.pos = torch.zeros(B, slots, 2, device=device)
        self.age = torch.full((B, slots), DURATION, device=device)
        self.row = torch.zeros(B, slots, dtype=torch.long, device=device)
        self.seed = torch.zeros(B, slots, device=device)
        self.gen = generator

    @classmethod
    def from_local(cls, state: torch.Tensor, h: torch.Tensor, seed: float | None = None) -> "Effects":
        """The explosions a renderer's local state `h` (B, FX_DIM, see renderer.py) describes: one
        slot per invader, at the remembered centre, with the remembered age; alive slots show
        nothing. With a fixed `seed` the teacher is a deterministic function of (state, h), which
        is what the renderer is trained to reproduce."""
        from . import renderer as R
        B = state.shape[0]
        hh = h.reshape(B, C.N_INVADERS, R.FX_STRIDE)
        fx = cls(B, state.device, slots=C.N_INVADERS)
        alive = state[:, S.INV_OFF:S.STATE_DIM] > 0.5
        fx.age = torch.where(alive, torch.full_like(hh[:, :, 0], DURATION), hh[:, :, 0].clamp(max=DURATION))
        fx.pos = hh[:, :, 1:3].clone()
        fx.row = torch.div(torch.arange(C.N_INVADERS, device=state.device), C.INV_COLS, rounding_mode="floor")[None].expand(B, -1).clone()
        fx.seed = torch.full_like(fx.age, R.FX_SEED if seed is None else seed)
        return fx

    @property
    def B(self) -> int:
        return self.pos.shape[0]

    def active(self) -> torch.Tensor:
        return self.age < DURATION

    def n_active(self) -> torch.Tensor:
        """(B,) number of live explosions per world."""
        return self.active().sum(1)

    def reset(self, b=None) -> None:
        """Clear every effect (of world b, or of all worlds): used when a sequence restarts."""
        if b is None:
            self.age.fill_(DURATION)
        else:
            self.age[b] = DURATION

    def spawn(self, b: int, x: float, y: float, row: int, seed: float) -> None:
        free = (self.age[b] >= DURATION).nonzero()
        k = int(free[0]) if len(free) else int(self.age[b].argmax())
        self.pos[b, k, 0], self.pos[b, k, 1] = x, y
        self.age[b, k] = 0.0
        self.row[b, k] = row
        self.seed[b, k] = seed

    def observe(self, prev: torch.Tensor, cur: torch.Tensor) -> int:
        """Spawn an explosion for every invader alive in `prev` and dead in `cur` (B, STATE_DIM each),
        at the position it had in `prev`. Returns the number spawned. The last invader of a wave is
        never seen dying (the sim respawns the wave in the same tick), which is exactly what a
        renderer looking at consecutive states would see as well."""
        killed = (prev[:, S.INV_OFF:S.STATE_DIM] > 0.5) & (cur[:, S.INV_OFF:S.STATE_DIM] <= 0.5)
        idx = killed.nonzero()
        if idx.numel() == 0:
            return 0
        form = prev[:, S.FORM_OFF:S.FORM_OFF + 2]
        seeds = torch.rand(len(idx), generator=self.gen).tolist()
        for (b, i), sd in zip(idx.tolist(), seeds):
            row, col = divmod(i, C.INV_COLS)
            x = C.INV_X0 + float(form[b, 0]) + col * C.INV_DX
            y = C.INV_Y0 + float(form[b, 1]) + row * C.INV_DY
            self.spawn(b, x, y, row, sd)
        return len(idx)

    def advance(self, dt: float) -> None:
        self.age.add_(dt).clamp_(max=DURATION)

    # ------------------------------------------------------------------ drawing
    @torch.no_grad()
    def render(self, X: torch.Tensor, Y: torch.Tensor, px: float) -> torch.Tensor:
        """Premultiplied RGBA layer (B, 4, h, w) of all live explosions on the pixel grid X, Y (h, w)
        of world coordinates; px is the world size of one pixel (anti-aliasing width)."""
        h, w = X.shape
        out = torch.zeros(self.B, 4, h, w, device=X.device)
        live = self.active().nonzero()
        if live.numel() == 0:
            return out
        step = max(1, _BUDGET // (N_SPARK * h * w))
        for a in range(0, len(live), step):
            bi, mi = live[a:a + step, 0], live[a:a + step, 1]
            layer = self._draw(X, Y, px, self.pos[bi, mi], self.age[bi, mi], self.row[bi, mi], self.seed[bi, mi])
            out.index_add_(0, bi, layer)
        return out.clamp_(0, 1)

    def _draw(self, X, Y, px, c, t, row, seed) -> torch.Tensor:
        """K explosions -> premultiplied RGBA (K, 4, h, w)."""
        K, dev = c.shape[0], X.device
        pal = torch.tensor(_PALETTE, device=dev)[row]                       # (K, 3 kinds, 3)
        hot = torch.tensor(HOT, device=dev).expand(K, 3)
        spark_col, chunk_col, ring_col = pal[:, 0], pal[:, 1], pal[:, 2]
        tk = t[:, None]                                                      # (K, 1)
        tt = t[:, None, None]                                                # (K, 1, 1)
        dx = X[None] - c[:, 0, None, None]                                   # (K, h, w)
        dy = Y[None] - c[:, 1, None, None]
        r = torch.sqrt(dx * dx + dy * dy)
        rgb = torch.zeros(K, 3, X.shape[0], X.shape[1], device=dev)
        alpha = torch.zeros(K, X.shape[0], X.shape[1], device=dev)

        def add(prem_rgb, a):                                                # premultiplied 'over' accumulation
            nonlocal rgb, alpha
            rgb = rgb + prem_rgb
            alpha = alpha + a * (1 - alpha)

        # 1. flash: a hot disc that swells and dies within 150 ms
        rf = 0.015 + 0.10 * tt
        a_flash = torch.exp(-(r / rf) ** 2) * (1 - tt / 0.15).clamp(min=0) ** 2 * 1.3
        a_flash = a_flash.clamp(max=1)
        add(hot[:, :, None, None] * a_flash[:, None], a_flash)
        # 2. shock ring: expands fast to about one invader's size, thins out, gone by 0.35 s
        R = 0.015 + 0.045 * (1 - torch.exp(-tt / 0.10))
        th = 0.003 + 0.006 * tt / 0.35
        a_ring = torch.exp(-((r - R) / th) ** 2) * (1 - tt / 0.35).clamp(min=0) ** 1.5 * 0.9
        add(ring_col[:, :, None, None] * a_ring[:, None], a_ring)
        # 3. sparks: hot streaks flying out with drag, each with its own lifetime and flicker
        i = torch.arange(N_SPARK, device=dev, dtype=torch.float32)[None]
        sd = seed[:, None]
        h1, h2, h3, h4 = (_hash(sd, i + 100.0 * k) for k in range(4))
        ang = 2 * math.pi * (i + 0.9 * (h1 - 0.5)) / N_SPARK
        spd = 0.10 + 0.28 * h2                                               # farthest spark: ~0.12 world, one invader pitch
        life = DURATION * (0.5 + 0.5 * h3)
        tau = 0.32
        disp = spd * tau * (1 - torch.exp(-tk / tau))
        disp0 = spd * tau * (1 - torch.exp(-(tk - 0.045).clamp(min=0) / tau))
        ux, uy = torch.cos(ang), torch.sin(ang)
        coverage = self._streaks(X, Y, px, c, ux, uy, disp0, disp)
        rs = 0.0042 * (1 - 0.5 * tk / DURATION)                              # (K, 1)
        fade = (1 - tk / life).clamp(min=0) ** 1.3 * (0.8 + 0.2 * torch.sin(55 * tk + 2 * math.pi * h4))
        a_sp = (coverage(rs) * fade[:, :, None, None]).clamp(max=1)          # (K, N, h, w)
        col = _lerp(hot, spark_col, (tk / 0.22).clamp(max=1))                # (K, 3)
        a_tot = 1 - torch.prod(1 - a_sp, dim=1)
        add(col[:, :, None, None] * a_sp.sum(1)[:, None].clamp(max=1), a_tot)
        # 4. debris: a few slow metal chunks of the invader's material, hot at first, shrinking late
        j = torch.arange(N_CHUNK, device=dev, dtype=torch.float32)[None]
        g1, g2, g3 = (_hash(sd, j + 500.0 * k) for k in range(3))
        ang = 2 * math.pi * (j + 0.8 * (g1 - 0.5)) / N_CHUNK
        spd = 0.06 + 0.13 * g2
        tau = 0.5
        disp = spd * tau * (1 - torch.exp(-tk / tau))
        ux, uy = torch.cos(ang), torch.sin(ang)
        cx = c[:, 0:1] + ux * disp                                           # (K, Nc)
        cy = c[:, 1:2] + uy * disp
        ddx = X[None, None] - cx[..., None, None]                            # (K, Nc, h, w)
        ddy = Y[None, None] - cy[..., None, None]
        shrink = (1 - (tk - 0.35).clamp(min=0) / (DURATION - 0.35)).clamp(min=0)
        rc = 0.0075 * (0.7 + 0.6 * g3) * shrink                              # (K, Nc)
        d = torch.sqrt(ddx * ddx + ddy * ddy)
        cov = (0.5 - (d - rc[..., None, None]) / px).clamp(0, 1)
        nx = ddx / rc[..., None, None].clamp(min=1e-6)
        ny = ddy / rc[..., None, None].clamp(min=1e-6)
        shade = (0.45 + 0.55 * (0.5 - 0.5 * (0.5 * nx + 0.85 * ny)).clamp(0, 1))   # lit from the upper left, like the sprites
        heat = (tk / 0.2).clamp(max=1)                                       # (K, 1)
        base = _lerp(hot, chunk_col, heat)                                   # (K, 3)
        col_px = base[:, :, None, None, None] * _lerp(torch.ones_like(shade), shade, heat[:, :, None, None])[:, None]   # (K, 3, Nc, h, w)
        prem = (col_px * cov[:, None]).sum(2)
        add(prem.clamp(max=1), 1 - torch.prod(1 - cov, dim=1))
        return torch.cat([rgb.clamp(0, 1), alpha[:, None]], 1)

    @staticmethod
    def _streaks(X, Y, px, c, ux, uy, d0, d1):
        """Distance from every pixel to the K x N segments from c + u d0 to c + u d1, returned as a
        function radius (K, N) -> coverage (K, N, h, w) (anti-aliased core + soft halo)."""
        qx = c[:, 0:1] + ux * d0
        qy = c[:, 1:2] + uy * d0
        ex = ux * (d1 - d0)
        ey = uy * (d1 - d0)
        wx = X[None, None] - qx[..., None, None]
        wy = Y[None, None] - qy[..., None, None]
        e2 = (ex * ex + ey * ey)[..., None, None] + 1e-12
        s = ((wx * ex[..., None, None] + wy * ey[..., None, None]) / e2).clamp(0, 1)
        d = torch.sqrt((wx - ex[..., None, None] * s) ** 2 + (wy - ey[..., None, None] * s) ** 2)

        def coverage(rs):
            r4 = rs[..., None, None]
            core = (0.5 - (d - r4) / px).clamp(0, 1)
            halo = torch.exp(-(d / (3.5 * r4)) ** 2) * 0.45
            return core + halo

        return coverage


# ---------------------------------------------------------------------------------- preview
def _scripted_rollout(ticks: int, seed: int = 0):
    """Classic-sim rollout where both ships sit under live invaders and fire: a dense kill stream.
    Ship x is placed directly (lead-compensated) each tick: this is a teacher demo, not gameplay."""
    from . import sim
    rng = np.random.default_rng(seed)
    s = S.initial_state(1)
    states = [s.copy()]
    for _ in range(ticks):
        sh = S.ships(s); fo = S.formation(s); inv = S.invaders(s)
        ipos = S.invader_positions(fo)[0]
        u = np.zeros((1, S.INPUT_DIM), dtype=np.float32)
        for p in range(C.N_PLAYERS):
            cols = range(0, C.INV_COLS // 2) if p == 0 else range(C.INV_COLS // 2, C.INV_COLS)
            cand = [i for i in range(C.N_INVADERS) if inv[0, i] > 0.5 and (i % C.INV_COLS) in cols]
            if not cand:
                cand = [i for i in range(C.N_INVADERS) if inv[0, i] > 0.5]
            if cand:
                low = max(i // C.INV_COLS for i in cand)
                cand = [i for i in cand if i // C.INV_COLS == low]
                tgt = min(cand, key=lambda i: abs(ipos[i, 0] - sh[0, p, 0]))
                travel = (sh[0, p, 1] - C.SHIP_RADIUS - ipos[tgt, 1]) / C.BULLET_SPEED
                lead = (1.0 if fo[0, 2] > 0.5 else -1.0) * C.FORM_SPEED * travel
                sh[0, p, 0] = np.clip(ipos[tgt, 0] + lead + rng.normal(0, 0.004), C.SHIP_X_MIN, C.SHIP_X_MAX)
                sh[0, p, 2:4] = 0.0
            u[0, p * S.INPUT_STRIDE + 2] = 1.0
        s = sim.step(s, u)
        states.append(s.copy())
    return np.concatenate(states)


def main():
    import argparse
    from PIL import Image
    from . import raster_sprites as R
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="", help="renderer checkpoint with local effects state: also write the teacher-vs-renderer comparison")
    ap.add_argument("--out", default="assets/sprites")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    H = 720
    torch.manual_seed(0)
    if args.student:
        student_preview(args.student, out, dev, H); return

    # 1. sheet: one explosion per invader row, 10 frames 50 ms apart, 160 px crops
    s = torch.as_tensor(S.initial_state(3), device=dev)
    fx = Effects(3, dev)
    crop, frames, step_s = 160, 10, 0.05
    tiles = []
    for b in range(3):
        i = b * C.INV_COLS + 2
        s[b, S.INV_OFF + i] = 0.0                                          # the dead invader is gone from the state
        x = C.INV_X0 + 2 * C.INV_DX; y = C.INV_Y0 + b * C.INV_DY
        fx.spawn(b, x, y, b, 0.37 + 0.21 * b)
        wins = [(int(round(y * H - crop / 2)), int(round(x * H - crop / 2)), crop, crop)]
        tiles.append((b, wins[0]))
    rows = [[] for _ in range(3)]
    for f in range(frames):
        for b, win in tiles:
            rgb, _, _, _ = R.rasterize(s[b:b + 1], H, H, win, fx=_sub(fx, b))
            rows[b].append((rgb[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
        fx.advance(step_s)
    sheet = np.concatenate([np.concatenate(r, 1) for r in rows], 0)
    Image.fromarray(sheet).save(out / "explosion_sheet.png")

    # 2. rollout: 30 Hz sim, 60 fps frames; effects spawn from observed alive transitions
    ticks, fps = 150, 60
    st = torch.as_tensor(_scripted_rollout(ticks), device=dev)
    fx = Effects(1, dev, generator=torch.Generator().manual_seed(1))
    gif, best, best_n = [], None, -1
    sub = fps // C.TICK_RATE
    for k in range(ticks):
        n = fx.observe(st[k:k + 1], st[k + 1:k + 2])
        for _ in range(sub):
            rgb, _, _, _ = R.rasterize(st[k + 1:k + 2], 360, 360, fx=fx)
            gif.append(Image.fromarray((rgb[0] * 255).to(torch.uint8).cpu().numpy()))
            if int(fx.n_active()[0]) > best_n and 0.1 < float(fx.age.min()) < 0.2:
                full, _, _, _ = R.rasterize(st[k + 1:k + 2], H, H, fx=fx)
                best, best_n = full[0], int(fx.n_active()[0])
            fx.advance(1.0 / fps)
    gif[0].save(out / "explosion.gif", save_all=True, append_images=gif[1:], duration=1000 // fps, loop=0)
    if best is not None:
        Image.fromarray((best.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()).save(out / "explosion_frame.png")
    print(f"saved explosion_sheet.png ({sheet.shape[1]}x{sheet.shape[0]}), explosion.gif ({len(gif)} frames), "
          f"explosion_frame.png ({best_n} explosions in flight) in {out}")


def student_preview(ckpt: str, out: Path, dev: str, H: int = 720):
    """Teacher vs renderer network side by side: a frame sheet per invader row (10 frames, 50 ms
    apart) and the kill rollout as a GIF, the renderer carrying its own local state."""
    from PIL import Image
    from . import raster_sprites as R
    from . import renderer as NR
    model = NR.load_renderer(ckpt, dev)
    if not model.fx:
        print(f"{ckpt} has no local effects state (train with --fx)"); return
    if (model.H, model.W) != (H, H):
        model = model.at_resolution(H, H)
    crop, frames, step_s = 160, 10, 0.05
    rows = []
    with torch.no_grad():
        for b in range(3):
            s = torch.as_tensor(S.initial_state(1), device=dev)
            i = b * C.INV_COLS + 2
            s[0, S.INV_OFF + i] = 0.0
            x, y = C.INV_X0 + 2 * C.INV_DX, C.INV_Y0 + b * C.INV_DY
            win = (max(0, int(round(y * H - crop / 2))), max(0, int(round(x * H - crop / 2))), crop, crop)   # inside the frame (the network crops cannot go outside)
            h = NR.fx_idle(s); h[0, i * NR.FX_STRIDE] = 0.0
            tr, st = [], []
            for f in range(frames):
                rgb, _, _, _ = R.rasterize(s, H, H, win, fx=Effects.from_local(s, h))
                tr.append((rgb[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
                st.append((model(s, win, h)[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
                h[0, i * NR.FX_STRIDE] += step_s
            rows += [np.concatenate(tr, 1), np.concatenate(st, 1)]
        Image.fromarray(np.concatenate(rows, 0)).save(out / "explosion_student_sheet.png")
        ticks, fps = 150, 60
        states = torch.as_tensor(_scripted_rollout(ticks), device=dev)
        fx = Effects(1, dev, generator=torch.Generator().manual_seed(1))
        h = NR.fx_idle(states[:1])
        gif = []
        sub = fps // C.TICK_RATE
        for k in range(ticks):
            fx.observe(states[k:k + 1], states[k + 1:k + 2])
            for _ in range(sub):
                h = NR.step_fx(states[k + 1:k + 2], h, 1.0 / fps)
                t_rgb, _, _, _ = R.rasterize(states[k + 1:k + 2], H, H, fx=fx)
                s_rgb = model(states[k + 1:k + 2], h=h)
                pair = torch.cat([t_rgb[0], s_rgb[0]], 1).clamp(0, 1)                  # (H, 2H, 3)
                im = Image.fromarray((pair * 255).to(torch.uint8).cpu().numpy()).resize((2 * 360, 360), Image.LANCZOS)
                gif.append(im)
                fx.advance(1.0 / fps)
        gif[0].save(out / "explosion_student.gif", save_all=True, append_images=gif[1:], duration=1000 // fps, loop=0)
    print(f"saved explosion_student_sheet.png (teacher / renderer rows per invader type) and explosion_student.gif (teacher | renderer) in {out}")


def _sub(fx: Effects, b: int) -> Effects:
    """View of world b as a 1-world Effects (shares storage)."""
    v = Effects.__new__(Effects)
    v.pos, v.age, v.row, v.seed, v.gen = fx.pos[b:b + 1], fx.age[b:b + 1], fx.row[b:b + 1], fx.seed[b:b + 1], fx.gen
    return v


if __name__ == "__main__":
    main()
