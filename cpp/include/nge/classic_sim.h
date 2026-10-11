// Hand-written reference simulation (port of nge/sim.py).
//
// This is the game logic the neural world model was trained to imitate. It exists for the
// cross-check test and for A/B comparisons; it is NOT linked into the server or clients unless
// the build is configured with -DNGE_WITH_CLASSIC_SIM=ON, so the shipped binaries contain no
// transition rule other than the network.
#pragma once
#include "nge/state.h"

namespace nge {

// Invader centre derived from the formation offset (game geometry; used by the reference sim
// and the vector renderer only).
void invader_pos(const State& s, int i, double& x, double& y);

class ClassicSim : public ISim {
 public:
  ModelVector step(const ModelVector& x) override;
  static State step_state(const ModelVector& x);
  const char* name() const override { return "classic"; }
};

}  // namespace nge
