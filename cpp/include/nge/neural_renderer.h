// ONNX Runtime backed neural renderer: world state -> RGBA8 frame (CUDA EP, CPU fallback).
// The ONNX graph already outputs uint8 RGBA (B, H, W, 4); the conversion is part of the network.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nge/state.h"

namespace nge {

class NeuralRenderer {
 public:
  NeuralRenderer(const std::string& onnx_path, bool want_cuda, int threads = 1);   // throws std::runtime_error
  ~NeuralRenderer();
  int width() const { return w_; }
  int height() const { return h_; }
  bool on_cuda() const { return cuda_; }
  // Readback path: RGBA8, row-major, height x width x 4 bytes (host memory). The renderer takes
  // the same vector type as the world model and ignores its input slots by construction.
  const std::vector<uint8_t>& render(const ModelVector& x);
  // GPU-resident path (CUDA provider only): runs the network with its output bound to a CUDA
  // buffer and returns the device pointer of the RGBA8 frame (pitch = width*4). nullptr if
  // unavailable. The buffer is reused by the next call.
  const void* render_gpu(const ModelVector& x);
  const char* name() const { return name_.c_str(); }
  double last_ms() const { return last_ms_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string name_;
  bool cuda_ = false;
  int w_ = 0, h_ = 0;
  std::vector<uint8_t> frame_;
  double last_ms_ = 0.0;
};

}  // namespace nge
