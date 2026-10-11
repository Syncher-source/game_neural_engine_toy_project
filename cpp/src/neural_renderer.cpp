#include "nge/neural_renderer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#if NGE_HAVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace nge {

#if NGE_HAVE_ORT
struct NeuralRenderer::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "nge-render"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo cpu_mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  // GPU output path
  std::unique_ptr<Ort::Allocator> cuda_alloc;
  std::unique_ptr<Ort::IoBinding> binding;
  Ort::Value gpu_out{nullptr};
  bool gpu_path_failed = false;
};

static std::basic_string<ORTCHAR_T> to_ort_path(const std::string& p) {
#ifdef _WIN32
  return std::wstring(p.begin(), p.end());
#else
  return p;
#endif
}

static std::unique_ptr<Ort::Session> make_session(Ort::Env& env, const std::string& path, bool cuda, int threads, std::string& name, bool& on_cuda) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  name = "neural(cpu)"; on_cuda = false;
  if (cuda) {
    try {
      OrtCUDAProviderOptionsV2* c = nullptr;
      Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&c));
      opts.AppendExecutionProvider_CUDA_V2(*c);
      Ort::GetApi().ReleaseCUDAProviderOptions(c);
      name = "neural(cuda)"; on_cuda = true;
    } catch (const Ort::Exception& e) {
      std::fprintf(stderr, "[render] CUDA execution provider unavailable (%s); using CPU\n", e.what());
    }
  }
  return std::make_unique<Ort::Session>(env, to_ort_path(path).c_str(), opts);
}

NeuralRenderer::NeuralRenderer(const std::string& onnx_path, bool want_cuda, int threads) : impl_(new Impl) {
  try {
    impl_->session = make_session(impl_->env, onnx_path, want_cuda, threads, name_, cuda_);
  } catch (const Ort::Exception& e) {
    if (want_cuda) {
      std::fprintf(stderr, "[render] session with CUDA provider failed (%s); retrying on CPU\n", e.what());
      try { impl_->session = make_session(impl_->env, onnx_path, false, threads, name_, cuda_); }
      catch (const Ort::Exception& e2) { throw std::runtime_error(std::string("failed to load ") + onnx_path + ": " + e2.what()); }
    } else {
      throw std::runtime_error(std::string("failed to load ") + onnx_path + ": " + e.what());
    }
  }
  Ort::TypeInfo type_info = impl_->session->GetOutputTypeInfo(0);   // must outlive the shape view below
  auto info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();   // (B, H, W, 4)
  const auto etype = info.GetElementType();
  if (shape.size() != 4 || shape[3] != 4 || etype != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    char b[160];
    std::snprintf(b, sizeof b, "renderer output must be uint8 (B, H, W, 4); got rank %zu, last dim %lld, element type %d. Re-export with python -m nge.export_onnx",
                  shape.size(), shape.size() == 4 ? (long long)shape[3] : -1LL, (int)etype);
    throw std::runtime_error(b);
  }
  if (shape[1] <= 0 || shape[2] <= 0) {
    // symbolic H/W in the model: probe with one run
    ModelVector probe = pack(initial_state(), Input{});
    const int64_t ishape[2] = {1, L::MODEL_DIM};
    Ort::Value in = Ort::Value::CreateTensor<float>(impl_->cpu_mem, probe.data(), probe.size(), ishape, 2);
    const char* in_names[] = {"x"}; const char* out_names[] = {"image"};
    auto out = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
  }
  h_ = (int)shape[1]; w_ = (int)shape[2];
  frame_.resize((size_t)w_ * h_ * 4);
}

NeuralRenderer::~NeuralRenderer() = default;

const std::vector<uint8_t>& NeuralRenderer::render(const ModelVector& x) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t shape[2] = {1, L::MODEL_DIM};
  Ort::Value in = Ort::Value::CreateTensor<float>(impl_->cpu_mem, const_cast<float*>(x.data()), x.size(), shape, 2);
  const char* in_names[] = {"x"};
  const char* out_names[] = {"image"};
  auto out = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
  std::memcpy(frame_.data(), out[0].GetTensorData<uint8_t>(), frame_.size());
  last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return frame_;
}

const void* NeuralRenderer::render_gpu(const ModelVector& x) {
  if (!cuda_ || impl_->gpu_path_failed) return nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  try {
    if (!impl_->binding) {
      Ort::MemoryInfo cuda_mem("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
      impl_->cuda_alloc = std::make_unique<Ort::Allocator>(*impl_->session, cuda_mem);
      const int64_t oshape[4] = {1, h_, w_, 4};
      impl_->gpu_out = Ort::Value::CreateTensor(*impl_->cuda_alloc, oshape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      impl_->binding = std::make_unique<Ort::IoBinding>(*impl_->session);
      impl_->binding->BindOutput("image", impl_->gpu_out);
    }
    const int64_t shape[2] = {1, L::MODEL_DIM};
    Ort::Value in = Ort::Value::CreateTensor<float>(impl_->cpu_mem, const_cast<float*>(x.data()), x.size(), shape, 2);
    impl_->binding->BindInput("x", in);
    impl_->session->Run(Ort::RunOptions{nullptr}, *impl_->binding);   // synchronises the CUDA stream before returning
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return impl_->gpu_out.GetTensorData<uint8_t>();
  } catch (const Ort::Exception& e) {
    std::fprintf(stderr, "[render] GPU output path failed (%s); using readback\n", e.what());
    impl_->gpu_path_failed = true;
    return nullptr;
  }
}
#else
struct NeuralRenderer::Impl {};
NeuralRenderer::NeuralRenderer(const std::string&, bool, int) { throw std::runtime_error("built without ONNX Runtime"); }
NeuralRenderer::~NeuralRenderer() = default;
const std::vector<uint8_t>& NeuralRenderer::render(const ModelVector&) { return frame_; }
const void* NeuralRenderer::render_gpu(const ModelVector&) { return nullptr; }
#endif

}  // namespace nge
