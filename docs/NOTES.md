# Working notes (2026-10-11)

## Where we are

* Game binaries (`nge_server`, `nge_client`, `nge_bot`) contain no hand-written rules or
  drawing; both networks take the same 55-float vector; renderer ignores the input slots;
  D3D12/CUDA zero-readback frame path works on the user's PC (renderer ~2 ms/frame).
* Git with LFS for `*.pt`/`*.onnx`. `nge_test` cross-checks C++ vs Python; `nge_determinism`
  hashes a 600-tick scripted rollout.

## Determinism baseline (nge_determinism, 600 ticks, seed 1, ORT 1.31)

| run | hash | step |
|---|---|---|
| CPU provider, Linux (WSL, x86) | `7de44c2d7a11fa31` | 0.19 ms |
| CPU provider, Windows (same PC) | `7de44c2d7a11fa31` (identical) | 0.17 ms |
| CUDA provider, RTX 4090, TF32 off | `61e023bac2e7c6c6` | 0.52 ms |

* Each provider is repeatable with itself.
* CUDA vs CPU: differ from tick 0; max |diff| 1.18e-4 over 600 ticks (0.4 % of a ship radius);
  0 binary-flag mismatches. Bounded drift, not divergence, because the server resyncs clients.
* GPU slower than CPU at batch 1: launch/sync overhead dominates a 55-wide MLP.

## Decisions

* Snap-vs-smooth correction is a per-game policy, not engine design. Keep the correction
  available as data (`corr_`, visual offset in `client_core.cpp`); current 25 %/frame blend is
  a placeholder default.
* Protocol stays split: clients send only local inputs; server sends state + applied inputs.
* Skip "state quantisation" (roadmap level 2): with ~1e-4 differences any usable grid is
  straddled on most ticks, so it reduces but never guarantees. Go straight to level 3.

## Deterministic world model (level 3) — DONE 2026-10-11, verified in WSL

Implemented as designed (`det_math.h`, `det_sim.cpp`, `det_kernel.cu`, `sim_factory.h`,
`export_det.py`; `--sim det` on server/client/bot; `.nge` weights tracked by LFS). Changes to
the design made while implementing:
* CPU path uses AVX2+FMA intrinsics (lane = one GPU thread's fma chain); scalar reference kept
  as `det(cpu-scalar)` for the tests. Needs AVX2+FMA (checked at startup).
* Driver API + NVRTC are loaded dynamically from our own minimal declarations (`cuda_dyn.h`):
  no CUDA headers or import libs needed; Linux fetches only the NVRTC redist (deployed to bin/,
  builtins preloaded by soname).
* Quantisation step skipped (agreed).

Results (RTX 4090, WSL, 600 ticks, seed 1): det(cpu) == det(cpu-scalar) == det(cuda), hash
`edadd50658a76fda`; vs ORT CPU max diff 6.9e-4 (per-step 7.6e-6). Steps: det(cpu) 0.26 ms,
det(cuda) 0.29 ms (3 synchronous driver calls per step under WSL; ORT CPU is 0.16 ms).
Bot match with server det(cuda) + bots det(cpu)/det(cuda) under 10 % loss: normal.

Windows (MSVC, same PC) confirmed 2026-10-11: det(cpu) 0.22 ms, det(cuda) 0.31 ms, all three
rows hash `edadd50658a76fda`. `--sim det` is now the default; `--sim onnx` keeps ONNX Runtime.

### Next
* Optional perf: pinned host memory / one mapped buffer to cut the per-step driver calls;
  the CPU path could also be made faster than ORT (0.16 ms) with 2 threads or AVX-512.
* With server and clients bit-identical, the drift at reconcile for the OWN ship should be
  exactly 0 whenever the server had all the client's inputs; a check for that could be added to
  the HUD/bot log (currently "drift at reconcile" mixes in the unknown remote input).

## Graphics: 1950s sprites + 720p renderer (2026-10-11, done)

* `nge/sprites.py` procedural shaded sprites (user approved the look); `nge/raster_sprites.py`
  GPU compositor; renderer trained on 160 px crops of the 720x720 frame.
* Run 1 (12k steps, object-centred crops only): invaders excellent (90 % object px), ships ok
  but with full-height vertical streaks at the ship's x (never asked about those pixels),
  lasers not learned at all (0.5 %). Checkpoint kept as `renderer_720_run1.pt`.
* Runs 2-4 (fine-tunes): negative crops + band weighting fixed the streaks; a fatter bolt,
  per-object relative scale (bullets 0.025) and rel-coordinate Fourier features made lasers
  learnable; a radial-distance channel removed ghosts between same-row bolts. Final:
  41.7 dB, objects 96 %, lasers 78 %. Class-balanced loss was tried and rejected (destroys the
  background). Intermediate checkpoints `renderer_720_run{1,2,3}.pt` are gitignored.
* Remaining: very faint trace between two same-row bolts; could fine-tune longer. The 720
  ONNX is 85 MB because the coordinate features are a constant; generating them in-graph
  would shrink it to the weights.
* Frame time at 720: 17 ms fp32 / 11 ms fp16 (torch, RTX 4090) for the unchanged network.
* Client: F12 saves `screenshot_N.png` next to the exe (own PNG writer, `png_write.h`);
  README embeds `docs/screenshot.png` (stand-in rendered in Python until the user takes a
  real one on Windows).
* Publishing: MIT `LICENSE` (copyright holder written as "lpall": the user may want their
  full name), `THIRD_PARTY_NOTICES.md`, pyproject licence metadata. No remote configured yet.

## Parked

* World-model artifact: one bullet occasionally kills 3 invaders (flag-transition accuracy
  ~92 %). Levers: more samples around hit events, a loss term penalising >1 kill per bullet,
  closed-loop fine-tuning on the model's own rollouts.
* Client: no error threshold before replay; no jitter buffer / interpolation for the remote
  ship.
