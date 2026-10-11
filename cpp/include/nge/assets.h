// Locates data files (models) relative to the working directory or the executable, so the
// programs run from any directory: tries <path>, then <exe dir>/../.. up to 6 levels /<path>.
#pragma once
#include <filesystem>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace nge {

inline std::filesystem::path exe_dir() {
#ifdef _WIN32
  wchar_t buf[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  return n ? std::filesystem::path(buf, buf + n).parent_path() : std::filesystem::current_path();
#else
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  return n > 0 ? std::filesystem::path(std::string(buf, (size_t)n)).parent_path() : std::filesystem::current_path();
#endif
}

// `rel` like "checkpoints/world_model.onnx". Returns the first existing candidate, else `rel` unchanged.
inline std::string find_asset(const std::string& rel) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::exists(rel, ec)) return rel;
  fs::path dir = exe_dir();
  for (int i = 0; i < 7; ++i) {
    const fs::path cand = dir / rel;
    if (fs::exists(cand, ec)) return cand.string();
    const fs::path parent = dir.parent_path();
    if (parent == dir) break;
    dir = parent;
  }
  return rel;
}

}  // namespace nge
