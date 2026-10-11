"""Wire format (UDP, big-endian).

Client -> server  INPUT packet
    'NI' player:u8 count:u8 then count x (tick:u32 ax:i8 ay:i8 fire:u8)
    The client resends its last INPUT_HISTORY inputs every tick so a single
    lost packet does not lose an input.

Server -> client  STATE packet
    'NS' tick:u32
    per player: slack:i16 missed:u16   (slack = buffered input ticks ahead of the
                                        server, missed = total ticks where the
                                        input had to be held/extrapolated)
    last_inputs: INPUT_DIM x i8        (what the server applied on the last tick,
                                        lets clients extrapolate the remote ship)
    state: STATE_DIM x f32
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from . import config as C
from . import state as S

MAGIC_INPUT = b"NI"
MAGIC_STATE = b"NS"
INPUT_HISTORY = 8

_IN_HDR = struct.Struct(">2sBB")
_IN_ENT = struct.Struct(">Ibbb")
_ST_HDR = struct.Struct(">2sI" + "hH" * C.N_PLAYERS + "b" * S.INPUT_DIM)


def q8(v: float) -> int:
    return int(max(-127, min(127, round(float(v) * 127))))


def encode_input(player: int, entries: list[tuple[int, float, float, float]]) -> bytes:
    entries = entries[-INPUT_HISTORY:]
    buf = bytearray(_IN_HDR.pack(MAGIC_INPUT, player, len(entries)))
    for tick, ax, ay, fire in entries:
        buf += _IN_ENT.pack(tick & 0xFFFFFFFF, q8(ax), q8(ay), 1 if fire > 0.5 else 0)
    return bytes(buf)


def decode_input(data: bytes):
    """Returns (player, [(tick, ax, ay, fire), ...]) or None if not an input packet."""
    if len(data) < _IN_HDR.size or data[:2] != MAGIC_INPUT:
        return None
    _, player, count = _IN_HDR.unpack_from(data, 0)
    off = _IN_HDR.size
    ents = []
    for _ in range(count):
        if off + _IN_ENT.size > len(data):
            break
        tick, ax, ay, fire = _IN_ENT.unpack_from(data, off)
        off += _IN_ENT.size
        ents.append((tick, ax / 127.0, ay / 127.0, float(fire)))
    return player, ents


@dataclass
class StatePacket:
    tick: int
    slack: list[int]
    missed: list[int]
    last_inputs: np.ndarray   # (INPUT_DIM,)
    state: np.ndarray         # (STATE_DIM,)


def encode_state(tick: int, slack: list[int], missed: list[int], last_inputs: np.ndarray, state: np.ndarray) -> bytes:
    fields = [MAGIC_STATE, tick & 0xFFFFFFFF]
    for p in range(C.N_PLAYERS):
        fields += [int(max(-32768, min(32767, slack[p]))), int(missed[p]) & 0xFFFF]
    fields += [q8(v) for v in last_inputs]
    return _ST_HDR.pack(*fields) + np.asarray(state, dtype=">f4").tobytes()


def decode_state(data: bytes) -> StatePacket | None:
    if len(data) != _ST_HDR.size + 4 * S.STATE_DIM or data[:2] != MAGIC_STATE:
        return None
    f = _ST_HDR.unpack_from(data, 0)
    tick = f[1]
    slack = [f[2 + 2 * p] for p in range(C.N_PLAYERS)]
    missed = [f[3 + 2 * p] for p in range(C.N_PLAYERS)]
    li = np.array(f[2 + 2 * C.N_PLAYERS:], dtype=np.float32) / 127.0
    st = np.frombuffer(data, dtype=">f4", offset=_ST_HDR.size).astype(np.float32)
    return StatePacket(tick, slack, missed, li, st)
