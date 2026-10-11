#include "nge/neural_sim.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#if NGE_HAVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace nge {

#if NGE_HAVE_ORT
struct NeuralSim::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "nge"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
};

static std::basic_string<ORTCHAR_T> to_ort_path(const std::string& p) {
#ifdef _WIN32
  return std::wstring(p.begin(), p.end());
#else
  return p;
#endif
}

bool NeuralSim::available() { return true; }

NeuralSim::NeuralSim(const std::string& onnx_path, bool want_cuda, int threads) : impl_(new Impl) {
  impl_->opts.SetIntraOpNumThreads(threads);
  impl_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  name_ = "neural(cpu)";
  if (want_cuda) {
    try {
      OrtCUDAProviderOptionsV2* cuda = nullptr;
      Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
      // use_tf32=0: full fp32 matmuls (TF32 would round mantissas to 10 bits and differ per GPU)
      const char* keys[] = {"device_id", "use_tf32", "cudnn_conv_algo_search"};
      const char* vals[] = {"0", "0", "DEFAULT"};
      Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda, keys, vals, 3));
      impl_->opts.AppendExecutionProvider_CUDA_V2(*cuda);
      Ort::GetApi().ReleaseCUDAProviderOptions(cuda);
      name_ = "neural(cuda)";
    } catch (const Ort::Exception& e) {
      std::fprintf(stderr, "[neural] CUDA execution provider unavailable (%s); using CPU\n", e.what());
    }
  }
  try {
    impl_->session = std::make_unique<Ort::Session>(impl_->env, to_ort_path(onnx_path).c_str(), impl_->opts);
  } catch (const Ort::Exception& e) {
    if (name_ == "neural(cuda)") {
      // typically: onnxruntime_providers_cuda.dll present but CUDA / cuDNN runtime DLLs missing on PATH
      std::fprintf(stderr, "[neural] session with CUDA provider failed (%s); retrying on CPU\n", e.what());
      Ort::SessionOptions cpu_opts;
      cpu_opts.SetIntraOpNumThreads(threads);
      cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      impl_->opts = std::move(cpu_opts);
      name_ = "neural(cpu)";
      try {
        impl_->session = std::make_unique<Ort::Session>(impl_->env, to_ort_path(onnx_path).c_str(), impl_->opts);
      } catch (const Ort::Exception& e2) {
        throw std::runtime_error(std::string("failed to load ") + onnx_path + ": " + e2.what());
      }
    } else {
      throw std::runtime_error(std::string("failed to load ") + onnx_path + ": " + e.what());
    }
  }
}

NeuralSim::~NeuralSim() = default;

ModelVector NeuralSim::step(const ModelVector& x) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t x_shape[2] = {1, L::MODEL_DIM};
  Ort::Value in = Ort::Value::CreateTensor<float>(impl_->mem, const_cast<float*>(x.data()), x.size(), x_shape, 2);
  const char* in_names[] = {"x"};
  const char* out_names[] = {"x_next"};
  auto out = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
  ModelVector n;
  std::memcpy(n.data(), out[0].GetTensorData<float>(), sizeof(ModelVector));
  last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return n;
}
#else
struct NeuralSim::Impl {};
bool NeuralSim::available() { return false; }
NeuralSim::NeuralSim(const std::string&, bool, int) {
  throw std::runtime_error("built without ONNX Runtime (configure with -DONNXRUNTIME_DIR=...)");
}
NeuralSim::~NeuralSim() = default;
ModelVector NeuralSim::step(const ModelVector& x) { return x; }
#endif

}  // namespace nge
