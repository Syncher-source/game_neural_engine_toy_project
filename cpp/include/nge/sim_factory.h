// One place that turns command-line options into a world model, shared by server, client, bot:
//   --sim det|onnx     det: deterministic fixed-order kernel (default)   onnx: ONNX Runtime
//   --model <path>     default checkpoints/world_model.onnx or .nge depending on --sim
//   --cpu              CPU instead of CUDA
//   --classic          hand-written reference sim (only if built with NGE_WITH_CLASSIC_SIM)
//   --det-scalar       (det only) scalar CPU reference path, slow; for cross-checks
// Exits with code 2 and a message when the requested model cannot be loaded: the runtime
// programs never fall back to hand-written rules.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "nge/args.h"
#include "nge/assets.h"
#include "nge/det_sim.h"
#include "nge/neural_sim.h"
#include "nge/state.h"
#if NGE_WITH_CLASSIC_SIM
#include "nge/classic_sim.h"
#endif

namespace nge {

inline std::unique_ptr<ISim> make_sim(const Args& a, const char* who) {
  if (a.has("--classic")) {
#if NGE_WITH_CLASSIC_SIM
    return std::make_unique<ClassicSim>();
#else
    std::fprintf(stderr, "[%s] --classic is not available: built without the hand-written simulation (NGE_WITH_CLASSIC_SIM=OFF)\n", who);
    std::exit(2);
#endif
  }
  const std::string kind = a.get("--sim", "det");
  try {
    if (kind == "det") {
      const DetSim::Backend be = a.has("--det-scalar") ? DetSim::Backend::CpuScalar : a.has("--cpu") ? DetSim::Backend::Cpu : DetSim::Backend::Cuda;
      return std::make_unique<DetSim>(find_asset(a.get("--model", "checkpoints/world_model.nge")), be);
    }
    if (kind != "onnx") { std::fprintf(stderr, "[%s] unknown --sim %s (onnx | det)\n", who, kind.c_str()); std::exit(2); }
    return std::make_unique<NeuralSim>(find_asset(a.get("--model", "checkpoints/world_model.onnx")), !a.has("--cpu"), 1);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] cannot load the world model: %s\n", who, e.what());
    std::exit(2);
  }
}

}  // namespace nge
