#include "nge/gpu_interop.h"

#if defined(_WIN32) && NGE_HAVE_CUDART
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <cuda_runtime_api.h>
#include <SDL3/SDL.h>

#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace nge {
namespace {

#define NGE_CU(call)                                                                        \
  do {                                                                                      \
    cudaError_t _e = (call);                                                                \
    if (_e != cudaSuccess) { error = std::string(#call) + ": " + cudaGetErrorString(_e); return false; } \
  } while (0)
#define NGE_HR(call)                                                                        \
  do {                                                                                      \
    HRESULT _hr = (call);                                                                   \
    if (FAILED(_hr)) { char b[64]; std::snprintf(b, sizeof b, "0x%08lx", (unsigned long)_hr); error = std::string(#call) + " failed " + b; return false; } \
  } while (0)

class D3D12CudaInterop final : public GpuFrameInterop {
 public:
  ~D3D12CudaInterop() override {
    if (stream_) cudaStreamSynchronize(stream_);
    if (sdl_tex_) SDL_DestroyTexture(sdl_tex_);
    if (cuda_done_sem_) cudaDestroyExternalSemaphore(cuda_done_sem_);
    if (gfx_done_sem_) cudaDestroyExternalSemaphore(gfx_done_sem_);
    if (mip_) cudaFreeMipmappedArray(mip_);
    if (ext_mem_) cudaDestroyExternalMemory(ext_mem_);
    if (stream_) cudaStreamDestroy(stream_);
    if (tex_handle_) CloseHandle(tex_handle_);
    if (cuda_done_handle_) CloseHandle(cuda_done_handle_);
    if (gfx_done_handle_) CloseHandle(gfx_done_handle_);
  }

  bool init(SDL_Renderer* ren, int w, int h, std::string& error) {
    w_ = w; h_ = h;
    const char* drv = SDL_GetRendererName(ren);
    if (!drv || std::strcmp(drv, "direct3d12") != 0) { error = std::string("SDL renderer is '") + (drv ? drv : "?") + "', need direct3d12"; return false; }
    SDL_PropertiesID rp = SDL_GetRendererProperties(ren);
    device_ = (ID3D12Device*)SDL_GetPointerProperty(rp, SDL_PROP_RENDERER_D3D12_DEVICE_POINTER, nullptr);
    queue_ = (ID3D12CommandQueue*)SDL_GetPointerProperty(rp, SDL_PROP_RENDERER_D3D12_COMMAND_QUEUE_POINTER, nullptr);
    if (!device_ || !queue_) { error = "SDL did not expose the D3D12 device/queue"; return false; }

    // CUDA device must be the same adapter as the D3D12 device (compare LUIDs); the ONNX
    // Runtime CUDA provider uses device 0, so that one has to match.
    const LUID luid = device_->GetAdapterLuid();
    int n = 0;
    NGE_CU(cudaGetDeviceCount(&n));
    int match = -1;
    for (int i = 0; i < n; ++i) {
      cudaDeviceProp p{};
      if (cudaGetDeviceProperties(&p, i) == cudaSuccess && std::memcmp(p.luid, &luid, sizeof luid) == 0) { match = i; break; }
    }
    if (match != 0) { error = match < 0 ? "no CUDA device matches the D3D12 adapter" : "the D3D12 adapter is not CUDA device 0 (used by ONNX Runtime)"; return false; }
    NGE_CU(cudaSetDevice(0));
    NGE_CU(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    // ---- shared texture (initial state COPY_DEST matches what SDL assumes for wrapped textures)
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = (UINT64)w; rd.Height = (UINT)h; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    NGE_HR(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex_)));
    NGE_HR(device_->CreateSharedHandle(tex_.Get(), nullptr, GENERIC_ALL, nullptr, &tex_handle_));
    const D3D12_RESOURCE_ALLOCATION_INFO ai = device_->GetResourceAllocationInfo(0, 1, &rd);

    cudaExternalMemoryHandleDesc md{};
    md.type = cudaExternalMemoryHandleTypeD3D12Resource;
    md.handle.win32.handle = tex_handle_;
    md.size = ai.SizeInBytes;
    md.flags = cudaExternalMemoryDedicated;
    NGE_CU(cudaImportExternalMemory(&ext_mem_, &md));
    cudaExternalMemoryMipmappedArrayDesc ad{};
    ad.offset = 0;
    ad.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    ad.extent.width = (size_t)w; ad.extent.height = (size_t)h; ad.extent.depth = 0;
    ad.flags = 0;
    ad.numLevels = 1;
    NGE_CU(cudaExternalMemoryGetMappedMipmappedArray(&mip_, ext_mem_, &ad));
    NGE_CU(cudaGetMipmappedArrayLevel(&array_, mip_, 0));

    // ---- shared fences: CUDA -> graphics ("cuda_done") and graphics -> CUDA ("gfx_done")
    NGE_HR(device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&cuda_done_)));
    NGE_HR(device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&gfx_done_)));
    NGE_HR(device_->CreateSharedHandle(cuda_done_.Get(), nullptr, GENERIC_ALL, nullptr, &cuda_done_handle_));
    NGE_HR(device_->CreateSharedHandle(gfx_done_.Get(), nullptr, GENERIC_ALL, nullptr, &gfx_done_handle_));
    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    sd.handle.win32.handle = cuda_done_handle_;
    NGE_CU(cudaImportExternalSemaphore(&cuda_done_sem_, &sd));
    sd.handle.win32.handle = gfx_done_handle_;
    NGE_CU(cudaImportExternalSemaphore(&gfx_done_sem_, &sd));

    // ---- wrap the resource as an SDL texture
    SDL_PropertiesID tp = SDL_CreateProperties();
    SDL_SetNumberProperty(tp, SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, SDL_PIXELFORMAT_RGBA32);
    SDL_SetNumberProperty(tp, SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER, SDL_TEXTUREACCESS_STATIC);
    SDL_SetNumberProperty(tp, SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, w);
    SDL_SetNumberProperty(tp, SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, h);
    SDL_SetPointerProperty(tp, SDL_PROP_TEXTURE_CREATE_D3D12_TEXTURE_POINTER, tex_.Get());
    sdl_tex_ = SDL_CreateTextureWithProperties(ren, tp);
    SDL_DestroyProperties(tp);
    if (!sdl_tex_) { error = std::string("SDL_CreateTextureWithProperties: ") + SDL_GetError(); return false; }
    SDL_SetTextureScaleMode(sdl_tex_, SDL_SCALEMODE_NEAREST);
    return true;
  }

  SDL_Texture* texture() const override { return sdl_tex_; }

  void begin_frame() override {
    // the network is about to overwrite its output buffer: our copy out of it must be done
    cudaStreamSynchronize(stream_);
  }

  void submit(const void* device_rgba, size_t pitch) override {
    ++frame_;
    if (frame_ > 1) {
      cudaExternalSemaphoreWaitParams wp{};
      wp.params.fence.value = frame_ - 1;          // graphics finished drawing the previous frame
      cudaWaitExternalSemaphoresAsync(&gfx_done_sem_, &wp, 1, stream_);
    }
    cudaMemcpy2DToArrayAsync(array_, 0, 0, device_rgba, pitch, (size_t)w_ * 4, (size_t)h_, cudaMemcpyDeviceToDevice, stream_);
    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = frame_;
    cudaSignalExternalSemaphoresAsync(&cuda_done_sem_, &sp, 1, stream_);
    queue_->Wait(cuda_done_.Get(), frame_);        // everything SDL submits from now on waits for the copy
  }

  void end_frame() override { queue_->Signal(gfx_done_.Get(), frame_); }
  const char* name() const override { return "d3d12+cuda (no readback)"; }

 private:
  int w_ = 0, h_ = 0;
  ID3D12Device* device_ = nullptr;        // owned by SDL
  ID3D12CommandQueue* queue_ = nullptr;   // owned by SDL
  ComPtr<ID3D12Resource> tex_;
  ComPtr<ID3D12Fence> cuda_done_, gfx_done_;
  HANDLE tex_handle_ = nullptr, cuda_done_handle_ = nullptr, gfx_done_handle_ = nullptr;
  cudaExternalMemory_t ext_mem_ = nullptr;
  cudaMipmappedArray_t mip_ = nullptr;
  cudaArray_t array_ = nullptr;
  cudaExternalSemaphore_t cuda_done_sem_ = nullptr, gfx_done_sem_ = nullptr;
  cudaStream_t stream_ = nullptr;
  SDL_Texture* sdl_tex_ = nullptr;
  unsigned long long frame_ = 0;
};

}  // namespace

bool GpuFrameInterop::supported() { return true; }

std::unique_ptr<GpuFrameInterop> GpuFrameInterop::create(SDL_Renderer* renderer, int width, int height, std::string& error) {
  auto p = std::make_unique<D3D12CudaInterop>();
  if (!p->init(renderer, width, height, error)) return nullptr;
  return p;
}

}  // namespace nge

#else

namespace nge {
bool GpuFrameInterop::supported() { return false; }
std::unique_ptr<GpuFrameInterop> GpuFrameInterop::create(SDL_Renderer*, int, int, std::string& error) {
  error = "built without D3D12/CUDA interop (Windows + CUDA runtime headers required)";
  return nullptr;
}
}  // namespace nge

#endif
