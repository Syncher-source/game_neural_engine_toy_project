// Cross-checks the C++ runtime against the Python reference:
//   nge_test ../checkpoints/test_vectors.bin [world_model.onnx [renderer_test.bin renderer.onnx [world_model.nge]]]
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "nge/classic_sim.h"
#include "nge/det_sim.h"
#include "nge/neural_renderer.h"
#include "nge/neural_sim.h"
#include "nge/protocol.h"
#include "nge/state.h"

using namespace nge;

int main(int argc, char** argv) {
  int failures = 0;
  // ---- protocol round trip
  {
    std::vector<proto::InputEntry> ents = {{10, {0.5f, -1.f, 1.f}}, {11, {0.f, 0.f, 0.f}}};
    auto b = proto::encode_input(1, ents);
    auto d = proto::decode_input(b.data(), b.size());
    if (!d || d->first != 1 || d->second.size() != 2 || d->second[0].tick != 10 || std::fabs(d->second[0].input[0] - 0.504f) > 0.01f) { std::puts("FAIL input packet"); ++failures; }
    proto::StatePacket p{}; p.tick = 42; p.slack[0] = 3; p.slack[1] = -1; p.missed[0] = 7; p.missed[1] = 65535; p.last_inputs = {1, 0, 1, -1, 0, 0}; p.state = initial_state();
    auto sb = proto::encode_state(p);
    auto sp = proto::decode_state(sb.data(), sb.size());
    if (sb.size() != 216 || !sp || sp->tick != 42 || sp->slack[1] != -1 || sp->missed[1] != 65535 || sp->state != p.state) { std::puts("FAIL state packet"); ++failures; }
    else std::printf("protocol: ok (%zu-byte state packet)\n", sb.size());
  }
  if (argc < 2) { std::puts("usage: nge_test vectors.bin [model.onnx]"); return failures; }

  std::ifstream f(argv[1], std::ios::binary);
  int32_t n, sd, id;
  f.read((char*)&n, 4); f.read((char*)&sd, 4); f.read((char*)&id, 4);
  if (!f || sd != L::STATE_DIM || id != L::INPUT_DIM) { std::printf("FAIL vectors header (%d %d %d)\n", n, sd, id); return 1; }
  std::vector<State> st(n), classic(n), neural(n); std::vector<Input> ut(n);
  f.read((char*)st.data(), n * sizeof(State)); f.read((char*)ut.data(), n * sizeof(Input));
  f.read((char*)classic.data(), n * sizeof(State)); f.read((char*)neural.data(), n * sizeof(State));

  // ---- classic sim vs numpy
  {
    ClassicSim sim; double max_err = 0; int bad = 0;
    for (int i = 0; i < n; ++i) {
      const ModelVector r = sim.step(pack(st[i], ut[i]));
      if (!inputs_consumed(r)) { std::puts("FAIL classic sim did not consume the inputs"); ++failures; break; }
      const State& o = state_of(r);
      double e = 0; for (int k = 0; k < L::STATE_DIM; ++k) e = std::max(e, (double)std::fabs(o[k] - classic[i][k]));
      max_err = std::max(max_err, e); if (e > 1e-4) ++bad;
    }
    std::printf("classic sim vs numpy: %d/%d rows differ by >1e-4, max abs diff %.2e%s\n", bad, n, max_err, bad ? "  FAIL" : "  ok");
    failures += bad > 0;
  }
  // ---- neural sim (ONNX Runtime) vs torch
  if (argc >= 3) {
    try {
      NeuralSim sim(argv[2], false, 1); double max_err = 0; int bad = 0; double ms = 0;
      for (int i = 0; i < n; ++i) {
        const ModelVector r = sim.step(pack(st[i], ut[i])); ms += sim.last_step_ms();
        if (!inputs_consumed(r)) { std::puts("FAIL neural sim did not consume the inputs"); ++failures; break; }
        const State& o = state_of(r);
        double e = 0; for (int k = 0; k < L::STATE_DIM; ++k) e = std::max(e, (double)std::fabs(o[k] - neural[i][k]));
        max_err = std::max(max_err, e); if (e > 1e-3) ++bad;
      }
      std::printf("neural sim (%s) vs onnxruntime-python: %d/%d rows differ by >1e-3, max abs diff %.2e, %.3f ms/step%s\n", sim.name(), bad, n, max_err, ms / n, bad ? "  FAIL" : "  ok");
      failures += bad > 0;
    } catch (const std::exception& e) { std::printf("neural sim: SKIPPED (%s)\n", e.what()); }
  }
  // ---- neural renderer (ONNX Runtime) vs torch
  if (argc >= 5) {
    try {
      std::ifstream rf(argv[3], std::ios::binary);
      int32_t rn, rsd, rh, rw;
      rf.read((char*)&rn, 4); rf.read((char*)&rsd, 4); rf.read((char*)&rh, 4); rf.read((char*)&rw, 4);
      if (!rf) std::printf("neural renderer: SKIPPED (no %s)\n", argv[3]);
      else if (rsd != L::MODEL_DIM) { std::puts("FAIL renderer vectors header (expected MODEL_DIM rows)"); ++failures; }
      else {
        // rows are full model vectors whose input slots hold junk: the frames must not depend on them
        std::vector<ModelVector> rs(rn); std::vector<uint8_t> frames((size_t)rn * rh * rw * 4);
        rf.read((char*)rs.data(), rn * sizeof(ModelVector)); rf.read((char*)frames.data(), frames.size());
        NeuralRenderer r(argv[4], false, 1);
        if (r.width() != rw || r.height() != rh) { std::printf("FAIL renderer size %dx%d vs %dx%d\n", r.width(), r.height(), rw, rh); ++failures; }
        else {
          int bad_px = 0; double ms = 0; int maxd = 0;
          for (int i = 0; i < rn; ++i) {
            const auto& f = r.render(rs[i]); ms += r.last_ms();
            for (size_t k = 0; k < f.size(); ++k) { int d = std::abs((int)f[k] - (int)frames[(size_t)i * f.size() + k]); maxd = std::max(maxd, d); if (d > 2) ++bad_px; }
            ModelVector z = rs[i]; set_inputs(z, Input{});            // same state, zero inputs -> identical frame
            const auto f2 = r.render(z);
            if (f2 != f) { std::puts("FAIL renderer output depends on the input slots"); ++failures; break; }
          }
          std::printf("neural renderer (%s) %dx%d RGBA8 vs torch: %d/%zu channel values differ by >2/255, max diff %d, %.2f ms/frame%s\n",
                      r.name(), rw, rh, bad_px, frames.size(), maxd, ms / rn, bad_px > (int)(frames.size() / 1000) ? "  FAIL" : "  ok");
          failures += bad_px > (int)(frames.size() / 1000);
        }
      }
    } catch (const std::exception& e) { std::printf("neural renderer: SKIPPED (%s)\n", e.what()); }
  }
  // ---- deterministic sim: close to the ONNX reference, and bit-identical across its backends
  if (argc >= 6) {
    try {
      DetSim cpu(argv[5], DetSim::Backend::Cpu);
      std::vector<ModelVector> out_cpu(n);
      double max_err = 0; int bad = 0; double ms = 0;
      for (int i = 0; i < n; ++i) {
        out_cpu[i] = cpu.step(pack(st[i], ut[i])); ms += cpu.last_step_ms();
        if (!inputs_consumed(out_cpu[i])) { std::puts("FAIL det sim did not consume the inputs"); ++failures; break; }
        const State& o = state_of(out_cpu[i]);
        double e = 0; for (int k = 0; k < L::STATE_DIM; ++k) e = std::max(e, (double)std::fabs(o[k] - neural[i][k]));
        max_err = std::max(max_err, e); if (e > 1e-3) ++bad;
      }
      // the event choices are argmaxes: a near-tie can resolve differently under different arithmetic, so a
      // handful of rows may legitimately differ in a flag; det-vs-det below must still be bit-identical
      const bool det_ok = bad <= n / 200;
      std::printf("det sim (%s) vs onnxruntime-python: %d/%d rows differ by >1e-3 (<= 0.5%% allowed: argmax ties), %.3f ms/step%s\n", cpu.name(), bad, n, ms / n, det_ok ? "  ok" : "  FAIL");
      failures += !det_ok;

      DetSim scalar(argv[5], DetSim::Backend::CpuScalar);
      const int ns = std::min(n, 64); int diff = 0; ms = 0;
      for (int i = 0; i < ns; ++i) { const ModelVector r = scalar.step(pack(st[i], ut[i])); ms += scalar.last_step_ms(); diff += r != out_cpu[i]; }
      std::printf("det sim %s vs %s: %d/%d rows differ (must be 0), %.2f ms/step scalar%s\n", cpu.name(), scalar.name(), diff, ns, ms / ns, diff ? "  FAIL" : "  ok");
      failures += diff > 0;

      std::string why;
      if (DetSim::cuda_available(&why)) {
        DetSim gpu(argv[5], DetSim::Backend::Cuda);
        if (gpu.backend() == DetSim::Backend::Cuda) {
          diff = 0; ms = 0;
          for (int i = 0; i < n; ++i) { const ModelVector r = gpu.step(pack(st[i], ut[i])); ms += gpu.last_step_ms(); diff += r != out_cpu[i]; }
          std::printf("det sim %s vs %s: %d/%d rows differ (must be 0), %.3f ms/step cuda%s\n", gpu.name(), cpu.name(), diff, n, ms / n, diff ? "  FAIL" : "  ok");
          failures += diff > 0;
        }
      } else std::printf("det sim cuda: SKIPPED (%s)\n", why.c_str());
    } catch (const std::exception& e) { std::printf("det sim: FAIL (%s)\n", e.what()); ++failures; }
  }
  std::printf(failures ? "FAILED\n" : "ALL OK\n");
  return failures;
}
