// GPU-resident frame path: the renderer network's RGBA8 output (CUDA memory, ONNX Runtime CUDA
// provider) is copied device-to-device into a Direct3D 12 texture shared with CUDA and drawn by
// SDL's D3D12 renderer. Shared fences order the two queues. No CPU readback.
//
// Only available on Windows builds with CUDA runtime headers (NGE_HAVE_CUDART); elsewhere
// create() returns nullptr and the client falls back to the readback path.
#pragma once
#include <cstddef>
#include <memory>
#include <string>

struct SDL_Renderer;
struct SDL_Texture;

namespace nge {

class GpuFrameInterop {
 public:
  virtual ~GpuFrameInterop() = default;
  // Requires an SDL renderer created with the "direct3d12" driver.
  static std::unique_ptr<GpuFrameInterop> create(SDL_Renderer* renderer, int width, int height, std::string& error);
  static bool supported();

  virtual SDL_Texture* texture() const = 0;
  // Before running the renderer network: makes sure the previous frame's copy out of the
  // network's output buffer has completed (the network reuses that buffer).
  virtual void begin_frame() = 0;
  // After the network ran: copy its device RGBA8 buffer into the shared texture (on the CUDA
  // side, waits for the graphics queue to be done with the previous frame) and make the
  // graphics queue wait for the copy.
  virtual void submit(const void* device_rgba, size_t pitch_bytes) = 0;
  // After SDL_RenderPresent: signal that the graphics queue is done with the texture.
  virtual void end_frame() = 0;
  virtual const char* name() const = 0;
};

}  // namespace nge
