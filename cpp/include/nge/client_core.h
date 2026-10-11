// Client-side prediction + server reconciliation, independent of rendering/input (mirrors nge/client.py).
#pragma once
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nge/net.h"
#include "nge/protocol.h"
#include "nge/state.h"

namespace nge {

struct ClientConfig {
  int player = 0;
  int lead = 3;            // initial ticks to run ahead of the server
  bool smooth = true;      // blend corrections visually instead of snapping
  bool ghost = true;       // draw the last authoritative server state
};

class ClientCore {
 public:
  ClientCore(ClientConfig cfg, std::unique_ptr<ISim> sim, net::UdpSocket& sock, net::Addr server, net::NetParams np);

  // One tick: receive + reconcile, buffer `own` input, send inputs, predict one tick.
  void frame(const PlayerInput& own);

  bool synced() const { return tick_.has_value(); }
  uint32_t tick() const { return tick_.value_or(0); }
  const State& state() const { return state_of(world_); }
  const ModelVector& world() const { return world_; }
  const std::optional<State>& ghost() const { return ghost_; }
  const ISim& sim() const { return *sim_; }
  ClientConfig& config() { return cfg_; }
  void reset_visual() { for (auto& v : visual_off_) v = {0.f, 0.f}; }
  std::array<float, 2> render_pos(int p) const;      // predicted position minus smoothing offset
  std::vector<std::string> hud() const;

 private:
  void on_server_state(const proto::StatePacket& pkt);
  void step_local(const PlayerInput& own);
  void send_inputs();
  static double mean(const std::deque<double>& d);

  ClientConfig cfg_;
  std::unique_ptr<ISim> sim_;
  net::LossySocket net_;
  net::Addr server_;
  ModelVector world_;                          // [local world state | inputs], the vector the local model steps
  std::optional<uint32_t> tick_;
  std::map<uint32_t, PlayerInput> inputs_;   // unacked own inputs by tick
  PlayerInput last_own_{}, remote_last_{};
  std::map<uint32_t, State> pred_hist_;
  std::optional<State> ghost_;
  std::optional<proto::StatePacket> last_pkt_;
  std::array<std::array<float, 2>, L::N_PLAYERS> visual_off_{};
  int tick_adjust_ = 0;
  int64_t last_adjust_tick_ = -1000000;
  // stats
  std::deque<double> drift_own_, drift_all_, replay_steps_;
  double corr_ = 0.0;
  uint64_t pkts_ = 0, gaps_ = 0;
  std::optional<uint32_t> last_srv_tick_;
};

}  // namespace nge
