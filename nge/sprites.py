"""Procedural sprite generator: 1950s space-opera look (chrome rockets, flying saucers).

Everything is drawn from signed-distance fields with a height field per part, lit by one
key light, supersampled and box-filtered, so the sprites are resolution independent and
re-colourable. Used only by the training-side renderer (hand-written drawing never ships in
the game binaries); the renderer network learns to reproduce these.

    python -m nge.sprites            # writes assets/sprites/*.png and the preview sheets
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SS = 4                                         # supersampling factor
LIGHT = np.array([-0.45, -0.75, 0.6])          # key light: upper left, in front
SILVER = (0.80, 0.82, 0.88)
DARK_STEEL = (0.42, 0.45, 0.55)
COPPER = (0.78, 0.45, 0.25)
GOLD = (0.95, 0.78, 0.35)
GLASS_GREEN = (0.40, 0.95, 0.55)
PLAYER_COLORS = {0: (0.95, 0.25, 0.18), 1: (0.20, 0.75, 1.00)}        # red rocket, cyan rocket


def _norm(v):
    return v / np.linalg.norm(v)


class Canvas:
    """RGBA float canvas in unit coordinates x, y in [-1, 1] (y down), supersampled."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.W, self.H = w * SS, h * SS
        ys, xs = np.mgrid[0:self.H, 0:self.W]
        asp = self.W / self.H
        self.x = ((xs + 0.5) / self.W * 2 - 1) * max(asp, 1.0)
        self.y = ((ys + 0.5) / self.H * 2 - 1) * max(1.0 / asp, 1.0)
        self.px = 2.0 / min(self.W, self.H)
        self.rgb = np.zeros((self.H, self.W, 3))
        self.a = np.zeros((self.H, self.W))

    # ---- signed distance primitives (negative inside) and matching height fields
    def ellipse(self, cx, cy, rx, ry):
        q = np.sqrt(((self.x - cx) / rx) ** 2 + ((self.y - cy) / ry) ** 2)
        sdf = (q - 1.0) * min(rx, ry)
        height = np.sqrt(np.clip(1.0 - q * q, 0, None)) * min(rx, ry)
        return sdf, height

    def capsule(self, ax, ay, bx, by, r):
        px, py = self.x - ax, self.y - ay
        bx, by = bx - ax, by - ay
        t = np.clip((px * bx + py * by) / (bx * bx + by * by), 0, 1)
        d = np.sqrt((px - bx * t) ** 2 + (py - by * t) ** 2)
        return d - r, np.sqrt(np.clip(r * r - d * d, 0, None))

    def polygon(self, pts, bevel=0.06):
        pts = np.asarray(pts, dtype=float)
        n = len(pts)
        d = np.full_like(self.x, 1e9)
        sign = np.ones_like(self.x)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            e = b - a
            w = np.stack([self.x - a[0], self.y - a[1]], -1)
            t = np.clip((w[..., 0] * e[0] + w[..., 1] * e[1]) / (e @ e), 0, 1)
            dd = np.hypot(w[..., 0] - e[0] * t, w[..., 1] - e[1] * t)
            d = np.minimum(d, dd)
            c1 = self.y >= a[1]; c2 = self.y < b[1]; c3 = e[0] * w[..., 1] > e[1] * w[..., 0]
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            sign = np.where(flip, -sign, sign)
        sdf = sign * d
        height = np.clip(-sdf, 0, bevel)
        return sdf, height

    @staticmethod
    def clip(sdf, keep):
        """Intersect with a half-plane / other sdf (keep <= 0)."""
        return np.maximum(sdf, keep)

    # ---- painting
    def _over(self, a, color):
        out_a = a + self.a * (1 - a)
        num = color * a[..., None] + self.rgb * (self.a * (1 - a))[..., None]
        self.rgb = np.where(out_a[..., None] > 1e-6, num / np.maximum(out_a, 1e-6)[..., None], self.rgb)
        self.a = out_a

    def shade(self, sdf, height, albedo, spec=0.45, shin=28, alpha=1.0, ambient=0.22, rim=0.0):
        """Lit material: Lambert + Blinn-Phong on the height field, anti-aliased by the sdf."""
        cov = np.clip(0.5 - sdf / self.px, 0, 1) * alpha
        if height is None:
            nx = ny = np.zeros_like(sdf); nz = np.ones_like(sdf)
        else:
            gy, gx = np.gradient(height, self.px)
            nx, ny, nz = -gx, -gy, np.ones_like(gx)
            inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx * inv, ny * inv, nz * inv
        L = _norm(LIGHT)
        diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
        Hv = _norm(L + np.array([0, 0, 1.0]))
        s = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1) ** shin
        col = np.asarray(albedo)[None, None, :] * (ambient + (1 - ambient) * diff)[..., None] + (spec * s)[..., None]
        if rim:                                                   # darken grazing edges (sphere/cylinder silhouettes)
            col = col * (1 - rim + rim * nz)[..., None]
        self._over(cov, np.clip(col, 0, 1))

    def flat(self, sdf, color, alpha=1.0):
        self._over(np.clip(0.5 - sdf / self.px, 0, 1) * alpha, np.asarray(color)[None, None, :] * np.ones_like(self.rgb))

    def glow(self, cx, cy, rx, ry, color, strength=1.0, power=2.0):
        """Additive emissive halo."""
        d2 = ((self.x - cx) / rx) ** 2 + ((self.y - cy) / ry) ** 2
        a = np.exp(-(d2 ** (power / 2)) * 2.5) * strength
        self.rgb = np.clip(self.rgb + a[..., None] * np.asarray(color)[None, None, :], 0, 1)
        self.a = np.clip(self.a + a * (1 - self.a), 0, 1)

    # ---- output
    def image(self) -> Image.Image:
        prem = self.rgb * self.a[..., None]
        prem = prem.reshape(self.h, SS, self.w, SS, 3).mean((1, 3))
        a = self.a.reshape(self.h, SS, self.w, SS).mean((1, 3))
        rgb = np.where(a[..., None] > 1e-6, prem / np.maximum(a, 1e-6)[..., None], 0)
        out = np.concatenate([rgb, a[..., None]], -1)
        return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


# ------------------------------------------------------------------------------- sprites
def rocket(size: int, player: int) -> Image.Image:
    """Defender: chrome Flash-Gordon rocket, nose/fins in the player colour, exhaust glow."""
    c = Canvas(size, size)
    acc = PLAYER_COLORS[player]
    # exhaust
    c.glow(0, 0.88, 0.22, 0.30, (1.0, 0.75, 0.25), 0.9)
    c.glow(0, 0.80, 0.10, 0.16, (1.0, 1.0, 0.85), 0.9)
    # fins (behind the body)
    for sx in (-1, 1):
        sdf, h = c.polygon([(sx * 0.22, 0.25), (sx * 0.80, 0.86), (sx * 0.80, 0.95), (sx * 0.22, 0.86)], bevel=0.05)
        c.shade(sdf, h, acc, spec=0.35, shin=18)
    # tail fin (centre, thin)
    sdf, h = c.polygon([(-0.06, 0.4), (0.06, 0.4), (0.06, 0.92), (-0.06, 0.92)], bevel=0.03)
    c.shade(sdf, h, tuple(v * 0.8 for v in acc), spec=0.3)
    # body: chrome cylinder
    sdf, h = c.capsule(0, -0.50, 0, 0.62, 0.27)
    c.shade(sdf, h, SILVER, spec=0.8, shin=40, rim=0.45)
    # colour band
    sdf, h = c.capsule(0, 0.28, 0, 0.40, 0.272)
    c.shade(c.clip(sdf, np.maximum(0.28 - c.y, c.y - 0.40)), h, acc, spec=0.5, shin=30, rim=0.4)
    # nose cone
    sdf, h = c.ellipse(0, -0.48, 0.265, 0.52)
    c.shade(c.clip(sdf, c.y + 0.49), h * 0.6, acc, spec=0.6, shin=30, rim=0.4)
    # porthole: ring + glass
    sdf, h = c.ellipse(0, -0.08, 0.13, 0.13)
    c.shade(sdf, h, DARK_STEEL, spec=0.6, shin=30)
    sdf, h = c.ellipse(0, -0.08, 0.095, 0.095)
    c.shade(sdf, h, (0.10, 0.30, 0.55), spec=1.0, shin=14, ambient=0.35)
    c.glow(-0.03, -0.11, 0.035, 0.035, (0.8, 0.95, 1.0), 0.8)
    return c.image()


def saucer(size: int) -> Image.Image:
    """Enemy row 0: classic flying saucer, chrome disc, green glass dome, running lights."""
    c = Canvas(size, size)
    c.glow(0, -0.2, 0.5, 0.5, GLASS_GREEN, 0.25)
    # underside
    sdf, h = c.ellipse(0, 0.33, 0.55, 0.20)
    c.shade(sdf, h * 0.5, DARK_STEEL, spec=0.4, rim=0.3)
    # disc
    sdf, h = c.ellipse(0, 0.12, 0.98, 0.30)
    c.shade(sdf, h * 0.55, SILVER, spec=0.9, shin=36, rim=0.35)
    # rim seam
    sdf, _ = c.ellipse(0, 0.12, 0.98, 0.30)
    c.flat(np.abs(sdf + 0.045) - 0.012, (0.35, 0.37, 0.45), 0.6)
    # dome
    sdf, h = c.ellipse(0, -0.05, 0.44, 0.46)
    c.shade(c.clip(sdf, c.y + 0.02), h, GLASS_GREEN, spec=1.0, shin=22, alpha=0.95, ambient=0.3, rim=0.3)
    c.glow(-0.14, -0.30, 0.10, 0.10, (0.9, 1.0, 0.9), 0.7)
    # running lights
    for i, x in enumerate(np.linspace(-0.72, 0.72, 5)):
        col = (1.0, 0.85, 0.2) if i % 2 == 0 else (1.0, 0.25, 0.2)
        c.glow(x, 0.30, 0.09, 0.07, col, 0.9, power=3)
        sdf, h = c.ellipse(x, 0.30, 0.055, 0.04)
        c.shade(sdf, h, col, spec=0.8, ambient=0.8)
    return c.image()


def war_machine(size: int) -> Image.Image:
    """Enemy row 1: copper manta-wing machine with a cobra-neck green eye lamp."""
    c = Canvas(size, size)
    c.glow(0, -0.55, 0.35, 0.35, (0.3, 1.0, 0.4), 0.35)
    # wings
    sdf, h = c.polygon([(-0.98, 0.45), (-0.45, -0.05), (0.45, -0.05), (0.98, 0.45), (0.62, 0.62), (0.0, 0.42), (-0.62, 0.62)], bevel=0.08)
    c.shade(sdf, h, COPPER, spec=0.55, shin=24)
    # wing veins
    for sx in (-1, 1):
        sdf, _ = c.capsule(sx * 0.2, 0.2, sx * 0.85, 0.47, 0.012)
        c.flat(sdf, (0.45, 0.22, 0.10), 0.8)
    # body
    sdf, h = c.ellipse(0, 0.18, 0.36, 0.30)
    c.shade(sdf, h, COPPER, spec=0.7, shin=30, rim=0.4)
    # neck
    sdf, h = c.capsule(0, 0.0, 0, -0.55, 0.07)
    c.shade(sdf, h, DARK_STEEL, spec=0.5, rim=0.4)
    # hood
    sdf, h = c.ellipse(0, -0.60, 0.30, 0.24)
    c.shade(sdf, h, COPPER, spec=0.7, shin=30, rim=0.4)
    # eye lamp
    sdf, h = c.ellipse(0, -0.60, 0.17, 0.15)
    c.shade(sdf, h, (0.3, 1.0, 0.45), spec=1.0, shin=16, ambient=0.75)
    c.glow(-0.04, -0.64, 0.06, 0.06, (1, 1, 1), 0.8)
    return c.image()


def orb(size: int) -> Image.Image:
    """Enemy row 2: dark-chrome Sputnik orb with a gold ring, antennae and a red beacon."""
    c = Canvas(size, size)
    # antennae
    for ang in (-0.8, -2.35, 0.8, 2.35):
        sdf, h = c.capsule(0, 0, 0.92 * np.cos(ang), 0.92 * np.sin(ang), 0.03)
        c.shade(sdf, h, SILVER, spec=0.7, rim=0.4)
    # back half of ring
    sdf, _ = c.ellipse(0, 0.05, 0.85, 0.26)
    ring = np.abs(sdf) - 0.05
    c.shade(c.clip(ring, c.y - 0.05), np.clip(0.05 - np.abs(sdf), 0, None), GOLD, spec=0.9, shin=34)
    # sphere
    sdf, h = c.ellipse(0, 0, 0.52, 0.52)
    c.shade(sdf, h, DARK_STEEL, spec=1.0, shin=40, rim=0.5)
    # panel seam
    c.flat(np.abs(c.y + 0.12 * c.x) - 0.012 + np.maximum(sdf + 0.04, 0) * 1e3, (0.25, 0.27, 0.33), 0.7)
    # front half of ring
    c.shade(c.clip(ring, 0.05 - c.y), np.clip(0.05 - np.abs(sdf), 0, None), GOLD, spec=0.9, shin=34)
    # beacon
    c.glow(0, -0.30, 0.16, 0.16, (1.0, 0.2, 0.15), 0.9, power=3)
    sdf, h = c.ellipse(0, -0.30, 0.09, 0.09)
    c.shade(sdf, h, (1.0, 0.25, 0.2), spec=1.0, shin=16, ambient=0.8)
    return c.image()


def laser(w: int, h: int, player: int) -> Image.Image:
    """Shot: white-hot core with a coloured halo, tapered head."""
    c = Canvas(w, h)
    acc = PLAYER_COLORS[player]
    c.glow(0, 0.0, 0.95, 1.8, acc, 0.7, power=2.2)
    c.glow(0, -1.1, 0.8, 0.7, (1.0, 1.0, 1.0), 0.5, power=2.5)
    sdf, hh = c.capsule(0, -1.2, 0, 1.4, 0.42)
    c.shade(sdf, hh, acc, spec=0.0, ambient=1.0)
    sdf, hh = c.capsule(0, -1.3, 0, 1.2, 0.2)
    c.shade(sdf, hh, (1.0, 1.0, 0.95), spec=0.0, ambient=1.0)
    return c.image()


# ------------------------------------------------------------------------------- sheets
def starfield(w, h, seed=3, density=0.004):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3))
    img[..., :] = (0.02, 0.02, 0.06)
    n = int(w * h * density)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    br = rng.uniform(0.25, 1.0, n) ** 2
    img[ys, xs] = br[:, None] * np.array([0.9, 0.9, 1.0])
    return Image.fromarray((img * 255).astype(np.uint8), "RGB").convert("RGBA")


def all_sprites(size=64):
    return {
        "rocket_p0": rocket(size, 0), "rocket_p1": rocket(size, 1),
        "enemy_row0_saucer": saucer(size), "enemy_row1_warmachine": war_machine(size), "enemy_row2_orb": orb(size),
        "laser_p0": laser(size // 2, size, 0), "laser_p1": laser(size // 2, size, 1),
    }


def preview_sheet(out: Path):
    big = {k: (rocket(160, 0) if k == "rocket_p0" else rocket(160, 1) if k == "rocket_p1" else saucer(160) if "saucer" in k
               else war_machine(160) if "warmachine" in k else orb(160) if "orb" in k else laser(80, 160, int(k[-1])))
           for k in all_sprites(64)}
    # in-game pixel sizes at 240x240: ship 14 px wide, invader 17x14, laser ~4x12
    small = {"rocket_p0": rocket(16, 0), "rocket_p1": rocket(16, 1), "enemy_row0_saucer": saucer(18),
             "enemy_row1_warmachine": war_machine(18), "enemy_row2_orb": orb(18), "laser_p0": laser(6, 14, 0), "laser_p1": laser(6, 14, 1)}
    cols = len(big); cw, rowh = 190, 330
    sheet = starfield(cols * cw, rowh)
    d = ImageDraw.Draw(sheet)
    for i, k in enumerate(big):
        x0 = i * cw
        b = big[k]; sheet.alpha_composite(b, (x0 + (cw - b.width) // 2, 20))
        s = small[k].resize((small[k].width * 5, small[k].height * 5), Image.NEAREST)
        sheet.alpha_composite(s, (x0 + (cw - s.width) // 2, 205))
        d.text((x0 + 10, 300), k, fill=(200, 200, 220, 255))
        d.text((x0 + 10, 312), "160 px  /  game size x5", fill=(120, 120, 150, 255))
    sheet.save(out)


def mock_frame(out: Path, scale=3):
    """A 240x240 game frame drawn with the sprites, upscaled for inspection."""
    W = 240
    fr = starfield(W, W, seed=5)
    inv = {0: saucer(18), 1: war_machine(18), 2: orb(18)}
    for r in range(3):
        for col in range(6):
            x, y = 0.08 + 0.12 * col + 0.05, 0.08 + 0.08 * r + 0.02
            sp = inv[r]
            fr.alpha_composite(sp, (int(x * W) - sp.width // 2, int(y * W) - sp.height // 2))
    for p, (x, y) in enumerate([(0.3, 0.9), (0.7, 0.86)]):
        sp = rocket(16, p); fr.alpha_composite(sp, (int(x * W) - 8, int(y * W) - 8))
        lz = laser(6, 14, p); fr.alpha_composite(lz, (int(x * W) - 3, int((y - 0.25) * W) - 7))
    fr = fr.resize((W * scale, W * scale), Image.LANCZOS)
    fr.save(out)


def main():
    out = Path("assets/sprites"); out.mkdir(parents=True, exist_ok=True)
    for k, im in all_sprites(64).items():
        im.save(out / f"{k}.png")
    preview_sheet(out / "preview_sheet.png")
    mock_frame(out / "preview_frame.png")
    print(f"saved {len(all_sprites(64))} sprites + preview_sheet.png + preview_frame.png in {out}")


if __name__ == "__main__":
    main()
