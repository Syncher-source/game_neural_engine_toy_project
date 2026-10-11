"""Export the world model weights for the deterministic C++ kernel (`cpp/src/det_sim.cpp`).

Plain little-endian file, fp32 arrays in a fixed order; no framework on the consumer side.
The C++ side transposes the matrices on load for the GPU/SIMD layout; the accumulation order
(over the input index, ascending, starting from the bias) is fixed in `det_math.h`.

    header: "NGEW", u32 version=1, i32 model_dim, state_dim, hidden, layers, n_cont, n_bin, f32 ln_eps
    arrays: s_mean[state_dim], s_std[state_dim],
            inp_w[hidden, model_dim], inp_b[hidden],
            per layer: ln_w[hidden], ln_b[hidden], w1[hidden, hidden], b1[hidden], w2[hidden, hidden], b2[hidden],
            out_w[n_cont + n_bin, hidden], out_b[n_cont + n_bin],
            d_std[n_cont], cont_lo[n_cont], cont_hi[n_cont]
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

from . import state as S
from .model import WorldModel, load_model

MAGIC = b"NGEW"
VERSION = 1


def export_det(model: WorldModel, path: str | Path) -> int:
    sd = {k: v.detach().cpu().contiguous().numpy().astype("<f4") for k, v in model.state_dict().items()}
    eps = float(model.blocks[0][0].eps)
    out = [struct.pack("<4sIiiiiiif", MAGIC, VERSION, S.MODEL_DIM, S.STATE_DIM, model.hidden, model.layers, S.N_CONT, S.N_BIN, eps)]
    arrays = [sd["s_mean"], sd["s_std"], sd["inp.weight"], sd["inp.bias"]]
    for i in range(model.layers):
        arrays += [sd[f"blocks.{i}.0.weight"], sd[f"blocks.{i}.0.bias"], sd[f"blocks.{i}.1.weight"], sd[f"blocks.{i}.1.bias"],
                   sd[f"blocks.{i}.3.weight"], sd[f"blocks.{i}.3.bias"]]
    arrays += [sd["out.weight"], sd["out.bias"], sd["d_std"], sd["cont_lo"], sd["cont_hi"]]
    assert sd["inp.weight"].shape == (model.hidden, S.MODEL_DIM) and sd["out.weight"].shape == (S.N_CONT + S.N_BIN, model.hidden)
    out += [np.ascontiguousarray(a).tobytes() for a in arrays]
    data = b"".join(out)
    Path(path).write_bytes(data)
    return len(data)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/world_model.pt")
    ap.add_argument("--out", default="checkpoints/world_model.nge")
    args = ap.parse_args()
    n = export_det(load_model(args.ckpt, "cpu"), args.out)
    print(f"saved {args.out} ({n / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
