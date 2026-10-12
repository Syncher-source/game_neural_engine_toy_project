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

## Rules: measurement, structured events, closed-loop fine-tuning (2026-10-12, done)

* `nge/rules.py`: 10 rules checked on closed-loop rollouts; classic sim scores 0 (after fixing
  the checks for same-slot respawn, cooldown decay order, spawn position vs moved ship, wall
  contact). Two rules carry gameplay tolerances (|v| < 0.02 is "stopped", cooldown jump > 0.05).
* World model v2 (`MODEL_VERSION = 2`): event head = act[6] | dir | reset | hit[6 x 19]; alive'
  derived (`WorldModel.events`), mirrored in the ONNX graph (ArgMax/Equal/ReduceMax) and in
  `det_sim.cpp::finish`. `.nge` format v2. `nge_test` allows <= 0.5 % det-vs-ORT rows to differ
  (argmax near-ties); det-vs-det stays bit-exact (verified CPU == CUDA again).
* Training targets: `sim.step_events` / `data.generate(events=True)` give per-bullet hit index
  and wave reset.
* `nge/finetune.py`: on-policy rollouts + teacher one-step targets; best round by violation
  index (ratios floored at +5/1000). Lessons: hit-weight 8 during fine-tuning triples ghost
  kills in one round; lr 2e-4 constant destroys fidelity; the working recipe is lr 2e-5 cosine,
  hit-weight 1, teacher ratio 5. Gains are real but partial (see README table); spawn
  placement and cooldown precision regress a little.
* Checkpoints: `world_model.pt` = v2+ft (shipped), `world_model_v2_pre.pt` (v2 before ft),
  `world_model_v1.pt` (old head) kept locally, gitignored.
* Ideas not done: structural spawn head (per player: which free slot to fire into, gated by
  fire & cooldown), which would attack the largest remaining rate (spawn without fire 75/1000);
  a rule-violation penalty inside the fine-tuning loss (differentiable surrogates).

## State extension: decided against a wire latent, for renderer-local state (2026-10-12)

* The 49-slot explicit state stays the only wire format and the only thing the world model
  advances. A learned latent appended to the protocol was discussed (hybrid: explicit core +
  latent block) and rejected by the user: the network interface must stay clean for debugging
  and bandwidth, and a shared latent space is hard to keep stable across model versions.
* Instead, physics nobody needs to synchronise lives in the renderer, locally: explosions,
  trails, flashes, particles. Design sketch (not started):
  - renderer becomes recurrent: `(state, hidden_prev, dt) -> (frame, hidden)`; hidden state is
    local, never on the wire, needs no determinism, may be fp16; spatial (low-res feature map)
    rather than a vector so particles have positions to live in;
  - effects are triggered by *observed* transitions between rendered frames (alive 1->0,
    bullet active 1->0), so the world model and the protocol are untouched;
  - runs at frame rate, not tick rate; rewind/replay never feeds the renderer, only the final
    predicted state per frame, so replays cannot re-fire effects; mispredicted kills produce a
    visual-only ghost explosion (acceptable; alternative: drive effects from the authoritative
    stream, one RTT late);
  - training: particle system added to the teacher rasterizer (`raster_sprites.py`), renderer
    trained on short sequences with the recurrence unrolled; keep heavy weight on current-state
    object pixels and a narrow recurrent channel so the recurrence cannot lag ship positions.

### Teacher particle system (2026-10-12, done, look pending user approval)

* `nge/effects.py`: `Effects(B, device)` holds up to 8 explosions per world as (centre, age,
  row, seed); `observe(prev, cur)` spawns one per invader alive in `prev` and dead in `cur`, at
  its `prev` position; `advance(dt)`; `render(X, Y, px)` -> premultiplied RGBA layer. Every
  particle is analytic in the age (drag: `v tau (1 - exp(-t/tau))`), so frame rate does not
  change the look and there is no integration state. 0.7 s per explosion: flash (150 ms),
  shock ring (350 ms, radius ~0.06 = one invader), 18 sparks (streaks, per-spark lifetime and
  flicker, hot white -> row colour), 5 debris chunks of the row's material (shrink after
  0.35 s). Per-row palette: green/silver, copper, gold-red/dark steel.
* `raster_sprites.rasterize(s, H, W, window, fx=None)` now returns `(rgb, obj_a, laser_a, fx_a)`;
  the effects layer is composited after lasers and before ships. Without `fx` the output is
  bit-identical to before (checked against the committed version, full frame and crops).
* Cost (RTX 4090): 720 full frame B=16 with 3 explosions 65 ms/batch; 32 crops of 160 px with 2
  explosions each 100 ms/batch (mostly per-crop Python overhead, same as today's trainer).
* `python -m nge.effects` writes `assets/sprites/explosion_sheet.png` (3 rows x 10 frames,
  50 ms apart), `explosion_frame.png` (720, from a scripted kill rollout) and `explosion.gif`
  (360 px, 60 fps, 30 Hz sim: effects spawn from observed transitions only).
* Known limits: the last invader of a wave never explodes (the sim respawns the wave in the
  same tick, so no 1 -> 0 transition is observable: consistent with what the renderer sees);
  `_sub(fx, b)` views one world of a batch for per-crop rendering; the trainer does not use
  effects yet (unpacks and ignores `fx_a`).

### Renderer with local effects state (2026-10-12, done, look soft)

* Design chosen over a generic spatial recurrent hidden state: `h` = per invader slot
  [age since last alive, cx, cy] (FX_DIM = 54), updated by a fixed rule in the graph
  (`renderer.step_fx`: alive -> age 0 and centre follows the slot; dead -> age += dt saturating
  at 0.7 s, centre frozen). Reasons: exact, debuggable, no sequence training (sample (state, h)
  pairs directly and let the teacher draw from the same h via `Effects.from_local`), and the
  frozen centre keeps the explosion where the kill happened while the formation moves on.
  The teacher's particle seed is fixed (FX_SEED) so the target is a deterministic function of
  (state, h); per-slot variety could later come from a per-slot rotation of the relative coords.
* Network, attempt 1: +18 "objects" (slot centres from h) with the bullets' rel-coord Fourier
  features and 5 age features per slot (360 channels, 653k params, 23 ms/frame). 8000 steps
  lr 1e-4: explosion px 0.275 but visually only a faint average blob, no ring/sparks. Cause: no
  weight sharing across slots, each slot's explosion learned from 1/18 of the data.
* Network, attempt 2 (current): per pixel, argmax over slots of life x in-reach picks the one
  exploding slot; gather its rel coords (FX_SCALE 0.06, clamp +-2 = +-0.12 world), Fourier
  features, radial distance and 7 age features (life, sin/cos at 1, 2, 4 cycles), all
  multiplied by an in-reach mask. 22 extra channels, 567k params, 20 ms/frame fp32. ArgMax +
  GatherElements export fine (ORT CPU verified via nge_test). Old checkpoints load with the
  new channels zero-initialised (`grow_first_layer`), bit-for-bit the old behaviour.
  Result after 8000 steps: explosion px 0.39, PSNR 35.6, but visually still only a swelling,
  fading blob: no ring, no sparks, no row colour. Cause: a spark is ~0.14 clamped units wide,
  8x finer than the finest rel-coord Fourier feature (period 1 unit); nothing to build it from.
* Attempt 3 (current): same selection, plus a polar basis for the chosen explosion: cos/sin of
  k x angle for k = 1..16 (angle-addition recurrence from rx/r, ry/r: no atan2 in ONNX) and
  radial sin/cos at 0.5..8 cycles per unit (period down to 5 px at 720). Sparks are radial
  streaks and the ring a circle, both separable in these. 64 extra channels, 578k params,
  24 ms/frame fp32. Export + nge_test verified on the smoke checkpoint.
  Result (8000 steps from `renderer.pt`, `renderer_fx_run3.pt`): explosion px 0.457, PSNR
  36.1, objects 0.946, lasers 0.808. Visually the first real explosion: ring, radial spark
  pattern, row colour, correct place and time in the rollout, but blurred/low-order, and a
  faint square box around each explosion: the hard square in-reach mask (|rx|,|ry| < 2) made
  the features discontinuous and the 3x3 convs spread the step.
* Attempt 4 (shipped as `checkpoints/renderer.pt`; the sprite-only one kept locally as
  `renderer_nofx.pt`): circular reach (FX_REACH 2.2) with a soft edge over the last 0.4 units
  multiplying every explosion feature, so they go to zero continuously. 12000 more steps from
  `renderer_fx_run3.pt`, lr 1e-4, fx weight 15, p_slot 0.3. Result: explosion px 0.552, PSNR
  37.5, objects 0.958, lasers 0.846 (both above the sprite-only 0.96 / 0.78). Visually: flash,
  ring in the row colour, radial spark star, fade, right place and time; still soft, sparks not
  resolved individually, faint concentric texture (the radial Fourier features showing through).
  The box artifact is gone. Exported `renderer.onnx` (720) and `renderer_480.onnx`, vectors with
  4 match starts + 4 random states; nge_test: 0 pixels differ at both sizes.
* Where to go next if the look matters: (a) the laser lesson again: a fatter spark in the
  teacher (rs 0.0042 -> ~0.007, maybe 12 sparks) would be learnable at this resolution (the
  user approved the current look, so this is their call); (b) longer training / a wider first
  layer; (c) per-slot rotation of the polar angle for variety, free once the pattern is learned.
* Export contract for every renderer graph, old or new: inputs x, h, dt; outputs image,
  h_next. `renderer_test.bin` gained FX_DIM in the header and the h rows. C++ `NeuralRenderer`
  keeps h, feeds dt, loops h_next (readback and IO-binding paths; h_next bound to a host
  buffer), `reset_fx()` after the warm-up frames; `nge_test` also checks the ageing rule through
  the graph. Verified in WSL (ORT CPU) against both the old and a fx checkpoint: 0 pixels differ.
* Training: lr 3e-4 one-cycle, fx weight 25, p_slot 0.3 gave unreadable metrics (obj/PSNR
  included explosion pixels); metrics now exclude explosion px from the object score. Runs:
  attempt 1 above; attempt 2: lr 1e-4, fx weight 10, p_slot 0.25, 8000 steps from `renderer.pt`.

## Parked

* Client: no error threshold before replay; no jitter buffer / interpolation for the remote
  ship.
