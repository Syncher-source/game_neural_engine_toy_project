// ONNX Runtime backed world model. CUDA execution provider when available, CPU otherwise.
#pragma once
#include <memory>
#include <string>

#include "nge/state.h"

namespace nge {

class NeuralSim : public ISim {
 public:
  // Throws std::runtime_error if the binary was built without ONNX Runtime or the model fails to load.
  NeuralSim(const std::string& onnx_path, bool want_cuda, int threads = 1);
  ~NeuralSim() override;
  ModelVector step(const ModelVector& x) override;   // ONNX: "x"[MODEL_DIM] -> "x_next"[MODEL_DIM]
  const char* name() const override { return name_.c_str(); }
  double last_step_ms() const override { return last_ms_; }
  static bool available();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string name_;
  double last_ms_ = 0.0;
};

}  // namespace nge
