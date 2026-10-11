#include "nge/state.h"

#include <algorithm>
#include <cmath>

namespace nge {

State initial_state() {
  State s{};
  ship(s, 0, 0) = 0.3f; ship(s, 0, 1) = 0.9f;
  ship(s, 1, 0) = 0.7f; ship(s, 1, 1) = 0.9f;
  form(s, 2) = 1.0f;
  for (int i = 0; i < L::N_INVADERS; ++i) inv(s, i) = 1.0f;
  return s;
}

int invaders_alive(const State& s) {
  int n = 0;
  for (int i = 0; i < L::N_INVADERS; ++i) n += inv(s, i) > 0.5f;
  return n;
}

ModelVector pack(const State& s, const Input& u) {
  static_assert(L::MODEL_DIM == L::STATE_DIM + L::INPUT_DIM);
  ModelVector x;
  std::copy(s.begin(), s.end(), x.begin());
  std::copy(u.begin(), u.end(), x.begin() + L::STATE_DIM);
  return x;
}

const State& state_of(const ModelVector& x) { return *reinterpret_cast<const State*>(x.data()); }
State& state_of(ModelVector& x) { return *reinterpret_cast<State*>(x.data()); }

void set_inputs(ModelVector& x, const Input& u) { std::copy(u.begin(), u.end(), x.begin() + L::STATE_DIM); }

bool inputs_consumed(const ModelVector& x) {
  for (int i = L::STATE_DIM; i < L::MODEL_DIM; ++i) if (x[i] != 0.0f) return false;
  return true;
}

Input inputs_of(const ModelVector& x) {
  Input u;
  std::copy(x.begin() + L::STATE_DIM, x.end(), u.begin());
  return u;
}

Input merge_inputs(int me, const PlayerInput& own, const PlayerInput& remote) {
  Input u{};
  for (int k = 0; k < L::INPUT_STRIDE; ++k) {
    u[me * L::INPUT_STRIDE + k] = own[k];
    u[(1 - me) * L::INPUT_STRIDE + k] = remote[k];
  }
  return u;
}

}  // namespace nge
