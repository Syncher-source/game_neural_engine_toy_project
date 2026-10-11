// Wire format, byte-compatible with nge/protocol.py (big-endian).
#pragma once
#include <cstdint>
#include <optional>
#include <vector>

#include "nge/state.h"

namespace nge::proto {

inline constexpr int INPUT_HISTORY = 8;

struct InputEntry {
  uint32_t tick;
  PlayerInput input;   // ax, ay in [-1,1], fire 0/1
};

struct StatePacket {
  uint32_t tick;
  int16_t slack[L::N_PLAYERS];
  uint16_t missed[L::N_PLAYERS];
  Input last_inputs;
  State state;
};

std::vector<uint8_t> encode_input(int player, const std::vector<InputEntry>& entries);
std::optional<std::pair<int, std::vector<InputEntry>>> decode_input(const uint8_t* data, size_t len);
std::vector<uint8_t> encode_state(const StatePacket& p);
std::optional<StatePacket> decode_state(const uint8_t* data, size_t len);

}  // namespace nge::proto
