// Determinism check for the world model: roll N ticks from the canonical start with a scripted,
// platform-independent input sequence and hash every intermediate state, for every backend:
//   onnx(cpu), onnx(cuda), det(cpu), det(cpu-scalar), det(cuda)
// Compare the printed hashes between machines / GPUs: identical hash = bit-identical rollout.
//   nge_determinism [--model checkpoints/world_model.onnx] [--det checkpoints/world_model.nge] [--ticks 600] [--seed 1] [--cpu-only] [--no-scalar]
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "nge/args.h"
#include "nge/assets.h"
#include "nge/det_sim.h"
#include "nge/neural_sim.h"
#include "nge/state.h"

using namespace nge;

namespace {

// 64-bit LCG -> inputs that are exactly representable and identical on every IEEE platform.
struct ScriptedInputs {
  uint64_t x;
  explicit ScriptedInputs(uint64_t seed) : x(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint64_t next() { x = x * 6364136223846793005ull + 1442695040888963407ull; return x >> 33; }
  Input cur{};
  Input at(uint32_t tick) {
    if (tick % 5 == 0)                              // new stick values every 5 ticks, per player
      for (int p = 0; p < L::N_PLAYERS; ++p) {
        cur[p * 3 + 0] = (float)((int)(next() % 201) - 100) / 100.0f;
        cur[p * 3 + 1] = (float)((int)(next() % 201) - 100) / 100.0f;
      }
    for (int p = 0; p < L::N_PLAYERS; ++p) cur[p * 3 + 2] = (next() % 3 == 0) ? 1.0f : 0.0f;
    return cur;
  }
};

inline uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
  const unsigned char* b = (const unsigned char*)data;
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}

struct Rollout {
  std::string name;
  std::vector<State> states;
  uint64_t hash = 14695981039346656037ull;
  double ms = 0;
  bool repeatable = false;
};

Rollout run(ISim& sim, int ticks, uint64_t seed) {
  Rollout r; r.name = sim.name();
  ModelVector w = pack(initial_state(), Input{});
  ScriptedInputs in(seed);
  for (int t = 0; t < ticks; ++t) {
    set_inputs(w, in.at((uint32_t)t));
    w = sim.step(w);
    r.ms += sim.last_step_ms();
    r.states.push_back(state_of(w));
    r.hash = fnv1a(r.hash, state_of(w).data(), sizeof(State));
  }
  return r;
}

Rollout measure(ISim& sim, int ticks, uint64_t seed) {
  Rollout r = run(sim, ticks, seed);
  r.repeatable = run(sim, ticks, seed).hash == r.hash;
  const State& f = r.states.back();
  std::printf("  %-16s hash %016" PRIx64 "   step %7.3f ms   %s   final ships (%.6f,%.6f) (%.6f,%.6f) alive %d\n", r.name.c_str(), r.hash,
              r.ms / ticks, r.repeatable ? "repeatable" : "NOT REPEATABLE", ship(f, 0, 0), ship(f, 0, 1), ship(f, 1, 0), ship(f, 1, 1), invaders_alive(f));
  return r;
}

void compare(const Rollout& a, const Rollout& b, bool must_match) {
  int first = -1; double max_diff = 0; int flag_mismatch = 0;
  const int ticks = (int)a.states.size();
  for (int t = 0; t < ticks; ++t) {
    for (int k = 0; k < L::STATE_DIM; ++k) {
      const double d = std::fabs((double)a.states[t][k] - (double)b.states[t][k]);
      if (d > 0 && first < 0) first = t;
      max_diff = std::fmax(max_diff, d);
    }
    for (int k = 0; k < L::N_BIN; ++k) if (a.states[t][L::BIN_IDX[k]] != b.states[t][L::BIN_IDX[k]]) ++flag_mismatch;
  }
  if (first < 0) std::printf("  %s vs %s: bit-identical over %d ticks%s\n", a.name.c_str(), b.name.c_str(), ticks, must_match ? "  ok" : "");
  else std::printf("  %s vs %s: first difference at tick %d, max |diff| %.3g, %d binary-flag mismatches%s\n", a.name.c_str(), b.name.c_str(), first,
                   max_diff, flag_mismatch, must_match ? "  FAIL (must be bit-identical)" : "  (expected: different arithmetic)");
}

}  // namespace

int main(int argc, char** argv) {
  Args a(argc, argv);
  const std::string onnx = find_asset(a.get("--model", "checkpoints/world_model.onnx"));
  const std::string det = find_asset(a.get("--det", "checkpoints/world_model.nge"));
  const int ticks = a.geti("--ticks", 600);
  const uint64_t seed = (uint64_t)a.geti("--seed", 1);
  const bool gpu = !a.has("--cpu-only");
  std::printf("[determinism] %d ticks, scripted inputs seed %" PRIu64 "\n", ticks, seed);
  std::vector<Rollout> onnx_rows, det_rows;

  std::printf("ONNX Runtime (%s):\n", onnx.c_str());
  try {
    NeuralSim cpu(onnx, false, 1);
    onnx_rows.push_back(measure(cpu, ticks, seed));
    if (gpu) {
      NeuralSim cuda(onnx, true, 1);
      if (std::strcmp(cuda.name(), "neural(cuda)") == 0) onnx_rows.push_back(measure(cuda, ticks, seed));
      else std::printf("  neural(cuda)     not available on this machine\n");
    }
  } catch (const std::exception& e) { std::printf("  skipped: %s\n", e.what()); }

  std::printf("deterministic kernel (%s):\n", det.c_str());
  try {
    DetSim cpu(det, DetSim::Backend::Cpu);
    det_rows.push_back(measure(cpu, ticks, seed));
    if (!a.has("--no-scalar")) { DetSim sc(det, DetSim::Backend::CpuScalar); det_rows.push_back(measure(sc, ticks, seed)); }
    if (gpu) {
      std::string why;
      if (DetSim::cuda_available(&why)) {
        DetSim cuda(det, DetSim::Backend::Cuda);
        if (cuda.backend() == DetSim::Backend::Cuda) det_rows.push_back(measure(cuda, ticks, seed));
      } else std::printf("  det(cuda)        not available: %s\n", why.c_str());
    }
  } catch (const std::exception& e) { std::printf("  skipped: %s\n", e.what()); }

  std::printf("comparisons:\n");
  if (onnx_rows.size() >= 2) compare(onnx_rows[0], onnx_rows[1], false);
  for (size_t i = 1; i < det_rows.size(); ++i) compare(det_rows[0], det_rows[i], true);
  if (!onnx_rows.empty() && !det_rows.empty()) compare(det_rows[0], onnx_rows[0], false);
  return 0;
}
