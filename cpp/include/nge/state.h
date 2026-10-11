// World state: a flat float vector shared by the neural model, the wire and the renderers.
// No game rules live here: only the layout, the start arrangement and input merging.
#pragma once
#include <array>
#include <cstdint>

#include "nge/layout.h"

namespace nge {

using State = std::array<float, L::STATE_DIM>;
using Input = std::array<float, L::INPUT_DIM>;      // all players: [ax, ay, fire] x N_PLAYERS
using PlayerInput = std::array<float, L::INPUT_STRIDE>;

// The one vector type the models work on: [world state | all player inputs].
//   server: [authoritative world state | remote players' inputs]
//   client: [local world state (network state + local prediction) | local input, last known remote input]
// World model: ModelVector -> ModelVector, same type, input slots come back zeroed ("consumed").
// Renderer:    ModelVector -> frame, the input slots are ignored by construction.
using ModelVector = std::array<float, L::MODEL_DIM>;
ModelVector pack(const State& s, const Input& u);
const State& state_of(const ModelVector& x);     // view of the state part
State& state_of(ModelVector& x);
Input inputs_of(const ModelVector& x);
void set_inputs(ModelVector& x, const Input& u);
bool inputs_consumed(const ModelVector& x);       // all input slots zero

inline float& ship(State& s, int p, int k) { return s[L::SHIP_OFF + p * L::SHIP_STRIDE + k]; }
inline float ship(const State& s, int p, int k) { return s[L::SHIP_OFF + p * L::SHIP_STRIDE + k]; }
inline float& bullet(State& s, int p, int b, int k) { return s[L::BUL_OFF + (p * L::N_BULLETS + b) * L::BUL_STRIDE + k]; }
inline float bullet(const State& s, int p, int b, int k) { return s[L::BUL_OFF + (p * L::N_BULLETS + b) * L::BUL_STRIDE + k]; }
inline float& form(State& s, int k) { return s[L::FORM_OFF + k]; }
inline float form(const State& s, int k) { return s[L::FORM_OFF + k]; }
inline float& inv(State& s, int i) { return s[L::INV_OFF + i]; }
inline float inv(const State& s, int i) { return s[L::INV_OFF + i]; }

State initial_state();
int invaders_alive(const State& s);
Input merge_inputs(int me, const PlayerInput& own, const PlayerInput& remote);

// Common step interface for the classic reference sim and the neural world model.
class ISim {
 public:
  virtual ~ISim() = default;
  // Advance one tick: [state | inputs] -> [next state | 0].
  virtual ModelVector step(const ModelVector& x) = 0;
  virtual const char* name() const = 0;
  virtual double last_step_ms() const { return 0.0; }
};

}  // namespace nge
