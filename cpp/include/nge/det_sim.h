// Deterministic world model: our own fixed-order fp32 implementation of the network, on the CPU
// (AVX2 + FMA) or on the GPU (CUDA kernel compiled at runtime with NVRTC). Same vector in ->
// same bits out on every machine. See det_math.h for the arithmetic contract.
#pragma once
#include <memory>
#include <string>

#include "nge/state.h"

namespace nge {

class DetSim : public ISim {
 public:
  enum class Backend { Cpu, CpuScalar, Cuda };
  // nge_path: weights written by `python -m nge.export_det`. With Backend::Cuda, falls back to
  // the CPU when no driver/NVRTC is available (prints why). Throws std::runtime_error on bad files
  // or a CPU without FMA.
  explicit DetSim(const std::string& nge_path, Backend want = Backend::Cpu);
  ~DetSim() override;
  ModelVector step(const ModelVector& x) override;
  const char* name() const override { return name_.c_str(); }
  double last_step_ms() const override { return last_ms_; }
  Backend backend() const;
  static bool cpu_supported();                 // AVX2 + FMA present
  static bool cuda_available(std::string* why = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string name_;
  double last_ms_ = 0.0;
};

}  // namespace nge
