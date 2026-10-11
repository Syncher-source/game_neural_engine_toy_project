"""Network-condition simulator wrapped around a non-blocking UDP socket.

Adds configurable packet loss, fixed latency and jitter on both directions so
the prediction/reconciliation logic can be stressed on a single machine.
"""
from __future__ import annotations

import heapq
import random
import socket
import time


class LossySocket:
    def __init__(self, sock: socket.socket, loss: float = 0.0, latency_ms: float = 0.0, jitter_ms: float = 0.0, seed: int | None = None):
        self.sock = sock
        self.sock.setblocking(False)
        self.loss = loss
        self.latency = latency_ms / 1000.0
        self.jitter = jitter_ms / 1000.0
        self.rng = random.Random(seed)
        self._out: list[tuple[float, int, bytes, tuple]] = []
        self._in: list[tuple[float, int, bytes, tuple]] = []
        self._seq = 0
        self.sent = self.dropped = self.received = 0

    def _delay(self) -> float:
        return max(0.0, self.latency + self.rng.uniform(-self.jitter, self.jitter))

    def sendto(self, data: bytes, addr) -> None:
        self.sent += 1
        if self.loss > 0 and self.rng.random() < self.loss:
            self.dropped += 1
            return
        if self.latency == 0 and self.jitter == 0:
            self._raw_send(data, addr)
            return
        self._seq += 1
        heapq.heappush(self._out, (time.perf_counter() + self._delay(), self._seq, data, addr))

    def _raw_send(self, data, addr):
        try:
            self.sock.sendto(data, addr)
        except OSError:
            pass

    def poll(self) -> list[tuple[bytes, tuple]]:
        """Flush due outgoing packets and return all due incoming packets."""
        now = time.perf_counter()
        while self._out and self._out[0][0] <= now:
            _, _, data, addr = heapq.heappop(self._out)
            self._raw_send(data, addr)
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            if self.loss > 0 and self.rng.random() < self.loss:
                self.dropped += 1
                continue
            self._seq += 1
            heapq.heappush(self._in, (now + self._delay(), self._seq, data, addr))
        out = []
        while self._in and self._in[0][0] <= now:
            _, _, data, addr = heapq.heappop(self._in)
            out.append((data, addr))
            self.received += 1
        return out


def add_net_args(ap) -> None:
    ap.add_argument("--loss", type=float, default=0.0, help="packet loss probability per direction, 0..1")
    ap.add_argument("--latency", type=float, default=0.0, help="added one-way latency in ms")
    ap.add_argument("--jitter", type=float, default=0.0, help="latency jitter +- ms")
