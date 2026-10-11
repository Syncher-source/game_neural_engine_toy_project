"""Authoritative world server.

    python -m nge.server --port 9999 [--classic] [--loss 0.2 --latency 50]

Every tick the server
  1. drains input packets and stores them per (player, tick),
  2. picks each player's input for the current tick (holding the last known
     input if it has not arrived -> `missed`),
  3. steps the world model on the GPU,
  4. broadcasts the new authoritative state to every connected client.

The server never looks at local devices: its only inputs are remote player
inputs, its only output is the network state.
"""
from __future__ import annotations

import argparse
import socket
import time

import numpy as np

from . import config as C
from . import protocol as P
from . import state as S
from .netsim import LossySocket, add_net_args
from .sim import ClassicSim


class Player:
    def __init__(self):
        self.addr = None
        self.inputs: dict[int, np.ndarray] = {}
        self.last = np.zeros(S.INPUT_STRIDE, dtype=np.float32)
        self.missed = 0
        self.received = 0
        self.last_seen = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--classic", action="store_true", help="use the numpy reference sim instead of the neural model")
    ap.add_argument("--ckpt", default="checkpoints/world_model.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=float, default=1.0, help="seconds between status lines (0 = quiet)")
    ap.add_argument("--max-ticks", type=int, default=0, help="stop after N ticks (0 = run forever)")
    add_net_args(ap)
    args = ap.parse_args()

    if args.classic:
        sim = ClassicSim()
    else:
        from .model import NeuralSim
        sim = NeuralSim(args.ckpt, args.device, threads=None)
    print(f"[server] sim={sim.name} tick={C.TICK_RATE}Hz listening on {args.bind}:{args.port} "
          f"netsim(loss={args.loss} latency={args.latency}ms jitter={args.jitter}ms)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    net = LossySocket(sock, args.loss, args.latency, args.jitter, seed=1)

    players = [Player() for _ in range(C.N_PLAYERS)]
    state = S.initial_state(1)
    tick = 0
    u = np.zeros((1, S.INPUT_DIM), dtype=np.float32)
    next_t = time.perf_counter()
    last_log = next_t
    step_ms_acc = 0.0; step_n = 0

    while True:
        # 1. network in
        now = time.perf_counter()
        for data, addr in net.poll():
            dec = P.decode_input(data)
            if dec is None:
                continue
            pid, ents = dec
            if not 0 <= pid < C.N_PLAYERS:
                continue
            pl = players[pid]
            pl.addr = addr
            pl.last_seen = now
            pl.received += 1
            for t, ax, ay, fire in ents:
                if t >= tick:
                    pl.inputs[t] = np.array([ax, ay, fire], dtype=np.float32)

        # 2. inputs for this tick (hold-last on loss)
        slack = []
        for p, pl in enumerate(players):
            if pl.addr is None or now - pl.last_seen > 3.0:
                pl.last[:] = 0.0
                slack.append(0)
            else:
                inp = pl.inputs.pop(tick, None)
                if inp is None:
                    pl.missed += 1
                else:
                    pl.last = inp
                slack.append((max(pl.inputs) - tick) if pl.inputs else 0)
            for t in [t for t in pl.inputs if t < tick]:
                del pl.inputs[t]
            u[0, p * S.INPUT_STRIDE:(p + 1) * S.INPUT_STRIDE] = pl.last

        # 3. world step
        t0 = time.perf_counter()
        state = sim.step(state, u)
        step_ms_acc += (time.perf_counter() - t0) * 1e3; step_n += 1
        tick += 1

        # 4. broadcast
        pkt = P.encode_state(tick, slack, [pl.missed for pl in players], u[0], state[0])
        for pl in players:
            if pl.addr is not None:
                net.sendto(pkt, pl.addr)

        if args.log_every and now - last_log >= args.log_every:
            last_log = now
            sh = S.ships(state)[0]
            who = " ".join(f"p{p}:{'on ' if pl.addr else 'off'} rx={pl.received} missed={pl.missed} slack={slack[p]}" for p, pl in enumerate(players))
            print(f"[server] tick={tick} step={step_ms_acc / max(step_n, 1):.2f}ms | {who} | "
                  f"ships ({sh[0, 0]:.2f},{sh[0, 1]:.2f}) ({sh[1, 0]:.2f},{sh[1, 1]:.2f}) alive={int(S.invaders(state)[0].sum())}", flush=True)
            step_ms_acc = 0.0; step_n = 0
        if args.max_ticks and tick >= args.max_ticks:
            break

        # fixed-rate pacing
        next_t += C.DT
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.5:
            next_t = time.perf_counter()


if __name__ == "__main__":
    main()
