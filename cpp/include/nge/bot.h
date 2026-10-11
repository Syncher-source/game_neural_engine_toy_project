// Scripted input policies for headless clients / testing.
#pragma once
#include <algorithm>
#include <random>
#include <string>

#include "nge/state.h"

namespace nge {

class BotInput {
 public:
  BotInput(std::string kind, int me, unsigned seed) : kind_(std::move(kind)), me_(me), rng_(seed) {}
  PlayerInput read(const State& s) {
    std::uniform_real_distribution<float> U(0.f, 1.f);
    if (kind_ == "chaser") {
      std::normal_distribution<float> N(0.f, 0.2f);
      for (int k = 0; k < 2; ++k) {
        const float d = ship(s, 1 - me_, k) - ship(s, me_, k);
        cur_[k] = std::clamp((d > 0 ? 1.f : d < 0 ? -1.f : 0.f) + N(rng_), -1.f, 1.f);
      }
    } else if (kind_ == "random") {
      if (U(rng_) < 0.08f) { std::uniform_int_distribution<int> D(-1, 1); cur_[0] = (float)D(rng_); cur_[1] = (float)D(rng_); }
    } else {
      cur_[0] = cur_[1] = 0.f;
    }
    cur_[2] = U(rng_) < 0.3f ? 1.f : 0.f;
    return cur_;
  }

 private:
  std::string kind_;
  int me_;
  std::mt19937 rng_;
  PlayerInput cur_{};
};

}  // namespace nge
