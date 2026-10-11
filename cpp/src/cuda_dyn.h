// Minimal, header-free bindings to the CUDA driver API and NVRTC, resolved at runtime with
// dlopen/LoadLibrary. Only what the deterministic kernel needs. No CUDA headers or import
// libraries are required to build; a machine without an NVIDIA driver simply reports
// "unavailable" and the caller uses the CPU path.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace nge::cu {

using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUstream = struct CUstream_st*;
using CUdeviceptr = unsigned long long;
using nvrtcProgram = struct _nvrtcProgram*;
using nvrtcResult = int;

constexpr int ATTR_CC_MAJOR = 75, ATTR_CC_MINOR = 76;

struct Api {
  // driver
  CUresult (*cuInit)(unsigned) = nullptr;
  CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
  CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
  CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
  CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
  CUresult (*cuCtxSynchronize)() = nullptr;
  CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
  CUresult (*cuModuleUnload)(CUmodule) = nullptr;
  CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
  CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
  CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
  CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**) = nullptr;
  CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
  // nvrtc
  nvrtcResult (*nvrtcVersion)(int*, int*) = nullptr;
  nvrtcResult (*nvrtcGetNumSupportedArchs)(int*) = nullptr;
  nvrtcResult (*nvrtcGetSupportedArchs)(int*) = nullptr;
  nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*) = nullptr;
  nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
  nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
  nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
  const char* (*nvrtcGetErrorString)(nvrtcResult) = nullptr;

  bool ok = false;
  std::string error;      // why loading failed
};

// Loads the driver (nvcuda.dll / libcuda.so.1) and NVRTC (nvrtc64_120_0.dll / libnvrtc.so.12,
// looked up next to the executable first). Idempotent; returns the process-wide table.
const Api& api();

}  // namespace nge::cu
