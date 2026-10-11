"""Game client: local input + server state -> local prediction -> render.

    python -m nge.client --player 0 --server 127.0.0.1:9999 [--classic] [--loss 0.3 --latency 80]
    python -m nge.client --player 1 --bot chaser --headless      # scripted opponent

Per tick the client
  1. drains STATE packets; on the newest one it *rewinds* to the authoritative
     state and *replays* its buffered local inputs up to the present (classic
     client-side prediction + server reconciliation),
  2. reads the local input (keyboard / joypad / bot), buffers it by tick and
     sends the last INPUT_HISTORY inputs to the server (redundancy vs. loss),
  3. steps its own copy of the world model one tick (remote player's input is
     held at the last value the server reported),
  4. renders the predicted state, smoothing out corrections visually.

A small controller keeps the client's tick a few ticks ahead of the server so
inputs arrive before the server needs them (`slack` reported by the server).
"""
from __future__ import annotations

import argparse
import socket
import time
from collections import deque

import numpy as np

from . import config as C
from . import protocol as P
from . import state as S
from .netsim import LossySocket, add_net_args
from .sim import ClassicSim


# ---------------------------------------------------------------- input sources
class KeyboardJoypadInput:
    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.joystick.init()
        self.joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
        if self.joy:
            self.joy.init()
            print(f"[client] joypad: {self.joy.get_name()}")

    def read(self) -> np.ndarray:
        pg = self.pg
        keys = pg.key.get_pressed()
        ax = float(keys[pg.K_RIGHT] or keys[pg.K_d]) - float(keys[pg.K_LEFT] or keys[pg.K_a])
        ay = float(keys[pg.K_DOWN] or keys[pg.K_s]) - float(keys[pg.K_UP] or keys[pg.K_w])
        fire = float(keys[pg.K_SPACE])
        if self.joy is not None:
            jx, jy = self.joy.get_axis(0), self.joy.get_axis(1)
            if abs(jx) > 0.15 or abs(jy) > 0.15:
                ax, ay = jx, jy
            if any(self.joy.get_button(b) for b in range(min(4, self.joy.get_numbuttons()))):
                fire = 1.0
        return np.array([ax, ay, fire], dtype=np.float32)


class BotInput:
    def __init__(self, kind: str, me: int, seed: int = 0):
        self.kind, self.me = kind, me
        self.rng = np.random.default_rng(seed)
        self.cur = np.zeros(3, dtype=np.float32)

    def read(self, state: np.ndarray) -> np.ndarray:
        sh = S.ships(state[None, :])[0]
        if self.kind == "chaser":
            d = sh[1 - self.me, 0:2] - sh[self.me, 0:2]
            self.cur[0:2] = np.clip(np.sign(d) + self.rng.normal(0, 0.2, 2), -1, 1)
        elif self.kind == "random":
            if self.rng.random() < 0.08:
                self.cur[0:2] = self.rng.integers(-1, 2, 2)
        else:
            self.cur[0:2] = 0.0
        self.cur[2] = float(self.rng.random() < 0.3)
        return self.cur.copy()


# ---------------------------------------------------------------- client core
class Client:
    def __init__(self, args):
        self.args = args
        self.me = args.player
        host, port = args.server.rsplit(":", 1)
        self.server_addr = (host, int(port))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        self.net = LossySocket(sock, args.loss, args.latency, args.jitter, seed=args.player + 7)
        if args.classic:
            self.sim = ClassicSim()
        else:
            from .model import NeuralSim
            import torch
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                args.device = "cpu"
            self.sim = NeuralSim(args.ckpt, args.device)

        self.state = S.initial_state(1)
        self.tick: int | None = None                 # None until first server packet
        self.lead = args.lead
        self.inputs: dict[int, np.ndarray] = {}      # own inputs by tick (unacked buffer)
        self.last_own = np.zeros(3, dtype=np.float32)
        self.remote_last = np.zeros(3, dtype=np.float32)
        self.pred_hist: dict[int, np.ndarray] = {}   # predicted state by tick, for drift measurement
        self.ghost: np.ndarray | None = None
        self.last_pkt: P.StatePacket | None = None
        self.visual_off = np.zeros((C.N_PLAYERS, 2), dtype=np.float32)
        self.tick_adjust = 0                         # +n: run extra ticks, -n: skip ticks
        self.last_adjust_tick = -10**9
        # stats
        self.drift_own = deque(maxlen=60)            # |predicted - server| own ship pos at reconcile time
        self.drift_all = deque(maxlen=60)
        self.corr = 0.0
        self.pkts = 0
        self.gaps = 0
        self.last_srv_tick = None
        self.replay_steps = deque(maxlen=60)

    # ---- helpers
    def full_input(self, own: np.ndarray) -> np.ndarray:
        u = np.zeros((1, S.INPUT_DIM), dtype=np.float32)
        u[0, self.me * 3:self.me * 3 + 3] = own
        u[0, (1 - self.me) * 3:(1 - self.me) * 3 + 3] = self.remote_last
        return u

    def ship_pos(self, st: np.ndarray) -> np.ndarray:
        return S.ships(st[None, :])[0][:, 0:2].copy()

    # ---- reconciliation
    def on_server_state(self, pkt: P.StatePacket):
        self.pkts += 1
        if self.last_srv_tick is not None and pkt.tick > self.last_srv_tick + 1:
            self.gaps += pkt.tick - self.last_srv_tick - 1
        self.last_srv_tick = pkt.tick
        self.last_pkt = pkt
        self.ghost = pkt.state.copy()
        self.remote_last = pkt.last_inputs[(1 - self.me) * 3:(1 - self.me) * 3 + 3].copy()
        T = pkt.tick

        if self.tick is None:
            self.state = pkt.state[None, :].copy()
            self.tick = T + self.lead
            print(f"[client] synced: server tick {T}, local tick {self.tick}")
            return

        # how wrong was our prediction for tick T?
        if T in self.pred_hist:
            pred = self.pred_hist[T]
            d_own = np.abs(self.ship_pos(pred)[self.me] - self.ship_pos(pkt.state)[self.me]).sum()
            self.drift_own.append(d_own)
            self.drift_all.append(np.abs(pred - pkt.state).mean())

        before = self.ship_pos(self.state[0])
        # rewind to authoritative state and replay unacked local inputs
        st = pkt.state[None, :].copy()
        n = 0
        for t in range(T, self.tick):
            own = self.inputs.get(t, self.last_own)
            st = self.sim.step(st, self.full_input(own))
            n += 1
        self.replay_steps.append(n)
        self.state = st
        after = self.ship_pos(self.state[0])
        self.corr = float(np.abs(after - before).sum())
        if not self.args.no_smooth:
            self.visual_off += after - before
        for t in [t for t in self.inputs if t < T]:
            del self.inputs[t]
        for t in [t for t in self.pred_hist if t <= T]:
            del self.pred_hist[t]

        # tick-lead controller: keep ~`lead` ticks of input buffered at the server
        slack = pkt.slack[self.me]
        if T - self.last_adjust_tick >= 5:
            if slack < 1:
                self.tick_adjust = 2
                self.last_adjust_tick = T
            elif slack > self.lead + 3:
                self.tick_adjust = -1
                self.last_adjust_tick = T

    # ---- one simulation tick
    def step_local(self, own: np.ndarray):
        assert self.tick is not None
        self.inputs[self.tick] = own
        self.last_own = own
        self.state = self.sim.step(self.state, self.full_input(own))
        self.tick += 1
        self.pred_hist[self.tick] = self.state[0].copy()

    def send_inputs(self):
        if self.tick is None:
            ents = [(0, 0.0, 0.0, 0.0)]     # hello
        else:
            ticks = sorted(self.inputs)[-P.INPUT_HISTORY:]
            ents = [(t, *self.inputs[t].tolist()) for t in ticks]
        self.net.sendto(P.encode_input(self.me, ents), self.server_addr)

    def frame(self, read_input):
        newest = None
        for data, _ in self.net.poll():
            pkt = P.decode_state(data)
            if pkt and (newest is None or pkt.tick > newest.tick):
                newest = pkt
        if newest is not None and (self.last_srv_tick is None or newest.tick > self.last_srv_tick):
            self.on_server_state(newest)

        if self.tick is None:
            self.send_inputs()
            return

        own = read_input(self.state[0])
        n_steps = 1 + self.tick_adjust
        self.tick_adjust = 0
        for _ in range(max(0, n_steps)):
            self.step_local(own)
        self.send_inputs()
        self.visual_off *= 0.75

    def hud(self) -> list[str]:
        p = self.last_pkt
        srv = p.tick if p else "-"
        slack = p.slack[self.me] if p else "-"
        missed = p.missed[self.me] if p else "-"
        d_own = np.mean(self.drift_own) if self.drift_own else 0.0
        d_all = np.mean(self.drift_all) if self.drift_all else 0.0
        a = self.args
        return [
            f"P{self.me} {'(' + a.bot + ' bot)' if a.bot != 'none' else ''}  local sim: {self.sim.name}   tick {self.tick}  server tick {srv}  ahead {self.tick - p.tick if p and self.tick else '-'}",
            f"server: input slack {slack} ticks   ticks w/o my input {missed}   replay {np.mean(self.replay_steps) if self.replay_steps else 0:.1f} steps/pkt   pkts {self.pkts}  gaps {self.gaps}",
            f"netsim: loss {a.loss:.0%}  latency {a.latency:.0f}ms  jitter {a.jitter:.0f}ms   smoothing {'off' if a.no_smooth else 'on'}  ghost {'on' if a.ghost else 'off'}",
            f"drift at reconcile: own ship {d_own:.4f} (radius {C.SHIP_RADIUS})   mean |state err| {d_all:.5f}   last correction {self.corr:.4f}",
            f"invaders alive {int(S.invaders(self.state)[0].sum())}   keys: arrows/WASD move, space fire, G ghost, H smoothing, Esc quit",
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:9999")
    ap.add_argument("--player", type=int, required=True, choices=range(C.N_PLAYERS))
    ap.add_argument("--classic", action="store_true", help="predict locally with the numpy reference sim instead of the neural model")
    ap.add_argument("--ckpt", default="checkpoints/world_model.pt")
    ap.add_argument("--device", default="cuda", help="device for the local neural sim and renderer (cuda falls back to cpu)")
    ap.add_argument("--render", default="neural", choices=["neural", "classic"], help="draw frames with the renderer network or with vector shapes")
    ap.add_argument("--renderer-ckpt", default="checkpoints/renderer.pt")
    ap.add_argument("--force-render", action="store_true", help="use the neural renderer even if it is slow on this device")
    ap.add_argument("--screenshot", default="", help="save a PNG of the window at --screenshot-tick and exit")
    ap.add_argument("--screenshot-tick", type=int, default=300)
    ap.add_argument("--lead", type=int, default=3, help="initial number of ticks to run ahead of the server")
    ap.add_argument("--bot", default="none", choices=["none", "random", "chaser", "idle"])
    ap.add_argument("--headless", action="store_true", help="no window (bots)")
    ap.add_argument("--no-smooth", action="store_true", help="snap to corrections instead of blending them")
    ap.add_argument("--ghost", action="store_true", default=True)
    ap.add_argument("--no-ghost", dest="ghost", action="store_false")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--log-every", type=float, default=0.0, help="print HUD to stdout every N seconds")
    add_net_args(ap)
    args = ap.parse_args()

    cl = Client(args)
    print(f"[client] player {args.player} -> {args.server}  local sim={cl.sim.name}  bot={args.bot}  headless={args.headless}")

    if args.headless:
        bot = BotInput(args.bot if args.bot != "none" else "idle", args.player, seed=args.player)
        next_t = time.perf_counter(); last_log = next_t
        while True:
            cl.frame(bot.read)
            now = time.perf_counter()
            if args.log_every and now - last_log >= args.log_every and cl.tick is not None:
                last_log = now
                print("[client] " + " | ".join(cl.hud()[:4]), flush=True)
            if args.max_ticks and cl.tick is not None and cl.tick >= args.max_ticks:
                break
            next_t += C.DT
            d = next_t - time.perf_counter()
            if d > 0:
                time.sleep(d)
            elif d < -0.5:
                next_t = time.perf_counter()
        return

    import pygame
    from .render import Renderer
    rend = Renderer(args.size, f"neural game engine - player {args.player}")
    nrend = None
    if args.render == "neural":
        try:
            from .renderer import RendererRuntime
            nrend = RendererRuntime(args.renderer_ckpt, args.device)
            nrend.render(cl.state[0]); nrend.render(cl.state[0])      # warm-up
            print(f"[client] renderer: {nrend.name} {nrend.model.H}x{nrend.model.W}, {nrend.last_ms:.2f} ms/frame")
            if nrend.last_ms > 20.0 and not args.force_render:
                print("[client] renderer too slow on this device; starting with classic drawing (press R, or pass --force-render)")
                args.render = "classic"
        except Exception as e:  # noqa: BLE001
            print(f"[client] neural renderer unavailable ({e}) -> classic drawing")
    kb = KeyboardJoypadInput()
    bot = BotInput(args.bot, args.player, seed=args.player) if args.bot != "none" else None
    clock = pygame.time.Clock()
    running = True
    last_log = time.perf_counter()
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_g:
                args.ghost = not args.ghost
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_h:
                args.no_smooth = not args.no_smooth
                cl.visual_off[:] = 0
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r and nrend is not None:
                args.render = "classic" if args.render == "neural" else "neural"
        cl.frame(bot.read if bot else (lambda st: kb.read()))
        hud = cl.hud() if cl.tick is not None else [f"P{args.player} waiting for server {args.server} ..."]
        pos = cl.ship_pos(cl.state[0]) - cl.visual_off
        frame = None
        if nrend is not None and args.render == "neural":
            rs = cl.state[0].copy()
            S.ships(rs[None, :])[0][:, 0:2] = pos          # draw ships at the smoothed positions
            frame = nrend.render(rs)
            hud = hud + [f"renderer: {nrend.name} {nrend.last_ms:.2f} ms   (R toggles classic drawing)"]
        rend.draw(cl.state[0], pos, cl.ghost, cl.me, hud, args.ghost, frame)
        if args.screenshot and cl.tick is not None and cl.tick >= args.screenshot_tick:
            rend.screenshot(args.screenshot)
            print(f"[client] saved {args.screenshot}")
            running = False
        if args.log_every and time.perf_counter() - last_log >= args.log_every and cl.tick is not None:
            last_log = time.perf_counter()
            print("[client] " + " | ".join(hud[:4]), flush=True)
        if args.max_ticks and cl.tick is not None and cl.tick >= args.max_ticks:
            running = False
        clock.tick(C.TICK_RATE)
    pygame.quit()


if __name__ == "__main__":
    main()
