// Minimal dependency-free PNG writer (RGBA8, stored/uncompressed deflate blocks). Used for the
// client's screenshot key; files are larger than a compressed PNG but every viewer reads them.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace nge::png {

inline uint32_t crc32(const uint8_t* d, size_t n, uint32_t c = 0xFFFFFFFFu) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) { uint32_t x = i; for (int k = 0; k < 8; ++k) x = (x & 1) ? 0xEDB88320u ^ (x >> 1) : x >> 1; table[i] = x; }
    init = true;
  }
  for (size_t i = 0; i < n; ++i) c = table[(c ^ d[i]) & 0xFF] ^ (c >> 8);
  return c;
}

inline void put32(std::vector<uint8_t>& v, uint32_t x) { v.push_back((uint8_t)(x >> 24)); v.push_back((uint8_t)(x >> 16)); v.push_back((uint8_t)(x >> 8)); v.push_back((uint8_t)x); }

inline void chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
  put32(out, (uint32_t)data.size());
  std::vector<uint8_t> td(type, type + 4);
  td.insert(td.end(), data.begin(), data.end());
  out.insert(out.end(), td.begin(), td.end());
  put32(out, crc32(td.data(), td.size()) ^ 0xFFFFFFFFu);
}

// rgba: w*h*4 bytes, row-major, pitch bytes per row. Returns false if the file cannot be written.
inline bool write_rgba(const std::string& path, int w, int h, const uint8_t* rgba, int pitch) {
  std::vector<uint8_t> raw;
  raw.reserve((size_t)h * (w * 4 + 1));
  for (int y = 0; y < h; ++y) { raw.push_back(0); raw.insert(raw.end(), rgba + (size_t)y * pitch, rgba + (size_t)y * pitch + (size_t)w * 4); }
  // zlib stream with stored blocks
  std::vector<uint8_t> z = {0x78, 0x01};
  size_t pos = 0;
  while (pos < raw.size()) {
    const uint16_t n = (uint16_t)std::min<size_t>(65535, raw.size() - pos);
    const uint16_t nn = (uint16_t)~n;
    z.push_back(pos + n == raw.size() ? 1 : 0);
    z.push_back((uint8_t)(n & 0xFF)); z.push_back((uint8_t)(n >> 8)); z.push_back((uint8_t)(nn & 0xFF)); z.push_back((uint8_t)(nn >> 8));
    z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
    pos += n;
  }
  uint32_t a = 1, b = 0;
  for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
  put32(z, (b << 16) | a);
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<uint8_t> ihdr; put32(ihdr, (uint32_t)w); put32(ihdr, (uint32_t)h);
  ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});          // 8-bit RGBA, deflate, no filter, no interlace
  chunk(out, "IHDR", ihdr); chunk(out, "IDAT", z); chunk(out, "IEND", {});
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
  std::fclose(f);
  return ok;
}

}  // namespace nge::png
