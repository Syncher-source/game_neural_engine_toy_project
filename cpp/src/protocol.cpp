#include "nge/protocol.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nge::proto {
namespace {

inline int8_t q8(float v) { return (int8_t)std::max(-127, std::min(127, (int)std::lround(v * 127.0f))); }
inline void put_u16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v >> 8); b.push_back(v & 0xFF); }
inline void put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 3; i >= 0; --i) b.push_back((v >> (8 * i)) & 0xFF); }
inline void put_f32(std::vector<uint8_t>& b, float f) { uint32_t v; std::memcpy(&v, &f, 4); put_u32(b, v); }
inline uint16_t get_u16(const uint8_t* p) { return (uint16_t)((p[0] << 8) | p[1]); }
inline uint32_t get_u32(const uint8_t* p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
inline float get_f32(const uint8_t* p) { uint32_t v = get_u32(p); float f; std::memcpy(&f, &v, 4); return f; }

constexpr size_t IN_HDR = 4, IN_ENT = 7;
constexpr size_t ST_HDR = 2 + 4 + 4 * L::N_PLAYERS + L::INPUT_DIM;
constexpr size_t ST_LEN = ST_HDR + 4 * L::STATE_DIM;

}  // namespace

std::vector<uint8_t> encode_input(int player, const std::vector<InputEntry>& entries) {
  const size_t first = entries.size() > (size_t)INPUT_HISTORY ? entries.size() - INPUT_HISTORY : 0;
  std::vector<uint8_t> b;
  b.reserve(IN_HDR + IN_ENT * (entries.size() - first));
  b.push_back('N'); b.push_back('I'); b.push_back((uint8_t)player); b.push_back((uint8_t)(entries.size() - first));
  for (size_t i = first; i < entries.size(); ++i) {
    put_u32(b, entries[i].tick);
    b.push_back((uint8_t)q8(entries[i].input[0]));
    b.push_back((uint8_t)q8(entries[i].input[1]));
    b.push_back(entries[i].input[2] > 0.5f ? 1 : 0);
  }
  return b;
}

std::optional<std::pair<int, std::vector<InputEntry>>> decode_input(const uint8_t* d, size_t len) {
  if (len < IN_HDR || d[0] != 'N' || d[1] != 'I') return std::nullopt;
  const int player = d[2], count = d[3];
  std::vector<InputEntry> ents;
  size_t off = IN_HDR;
  for (int i = 0; i < count && off + IN_ENT <= len; ++i, off += IN_ENT) {
    InputEntry e;
    e.tick = get_u32(d + off);
    e.input = {(int8_t)d[off + 4] / 127.0f, (int8_t)d[off + 5] / 127.0f, (float)(int8_t)d[off + 6]};
    ents.push_back(e);
  }
  return std::make_pair(player, std::move(ents));
}

std::vector<uint8_t> encode_state(const StatePacket& p) {
  std::vector<uint8_t> b;
  b.reserve(ST_LEN);
  b.push_back('N'); b.push_back('S');
  put_u32(b, p.tick);
  for (int i = 0; i < L::N_PLAYERS; ++i) { put_u16(b, (uint16_t)p.slack[i]); put_u16(b, p.missed[i]); }
  for (float v : p.last_inputs) b.push_back((uint8_t)q8(v));
  for (float v : p.state) put_f32(b, v);
  return b;
}

std::optional<StatePacket> decode_state(const uint8_t* d, size_t len) {
  if (len != ST_LEN || d[0] != 'N' || d[1] != 'S') return std::nullopt;
  StatePacket p;
  p.tick = get_u32(d + 2);
  size_t off = 6;
  for (int i = 0; i < L::N_PLAYERS; ++i) { p.slack[i] = (int16_t)get_u16(d + off); p.missed[i] = get_u16(d + off + 2); off += 4; }
  for (int i = 0; i < L::INPUT_DIM; ++i) p.last_inputs[i] = (int8_t)d[off++] / 127.0f;
  for (int i = 0; i < L::STATE_DIM; ++i) { p.state[i] = get_f32(d + off); off += 4; }
  return p;
}

}  // namespace nge::proto
