#include "nge/client_core.h"

#include <cmath>
#include <cstdio>

namespace nge {

ClientCore::ClientCore(ClientConfig cfg, std::unique_ptr<ISim> sim, net::UdpSocket& sock, net::Addr server, net::NetParams np)
    : cfg_(cfg), sim_(std::move(sim)), net_(sock, np, 7u + cfg.player), server_(server), world_(pack(initial_state(), Input{})) {}

double ClientCore::mean(const std::deque<double>& d) {
  if (d.empty()) return 0.0;
  double s = 0; for (double v : d) s += v; return s / d.size();
}

static void push60(std::deque<double>& d, double v) { d.push_back(v); if (d.size() > 60) d.pop_front(); }

std::array<float, 2> ClientCore::render_pos(int p) const {
  const State& s = state_of(world_);
  return {ship(s, p, 0) - visual_off_[p][0], ship(s, p, 1) - visual_off_[p][1]};
}

void ClientCore::on_server_state(const proto::StatePacket& pkt) {
  ++pkts_;
  if (last_srv_tick_ && pkt.tick > *last_srv_tick_ + 1) gaps_ += pkt.tick - *last_srv_tick_ - 1;
  last_srv_tick_ = pkt.tick;
  last_pkt_ = pkt;
  ghost_ = pkt.state;
  const int other = 1 - cfg_.player;
  for (int k = 0; k < L::INPUT_STRIDE; ++k) remote_last_[k] = pkt.last_inputs[other * L::INPUT_STRIDE + k];
  const uint32_t T = pkt.tick;

  if (!tick_) {
    world_ = pack(pkt.state, Input{});
    tick_ = T + cfg_.lead;
    std::printf("[client] synced: server tick %u, local tick %u\n", T, *tick_);
    return;
  }

  // how wrong was our prediction for tick T?
  if (auto it = pred_hist_.find(T); it != pred_hist_.end()) {
    const State& pred = it->second;
    const int me = cfg_.player;
    push60(drift_own_, std::fabs(ship(pred, me, 0) - ship(pkt.state, me, 0)) + std::fabs(ship(pred, me, 1) - ship(pkt.state, me, 1)));
    double e = 0; for (int i = 0; i < L::STATE_DIM; ++i) e += std::fabs(pred[i] - pkt.state[i]);
    push60(drift_all_, e / L::STATE_DIM);
  }

  std::array<std::array<float, 2>, L::N_PLAYERS> before;
  for (int p = 0; p < L::N_PLAYERS; ++p) before[p] = {ship(state(), p, 0), ship(state(), p, 1)};

  // rewind to the authoritative state, replay unacked local inputs through the local model
  ModelVector w = pack(pkt.state, Input{});
  int n = 0;
  for (uint32_t t = T; t < *tick_; ++t, ++n) {
    auto it = inputs_.find(t);
    const PlayerInput& own = it != inputs_.end() ? it->second : last_own_;
    set_inputs(w, merge_inputs(cfg_.player, own, remote_last_));
    w = sim_->step(w);
  }
  push60(replay_steps_, n);
  world_ = w;
  corr_ = 0.0;
  for (int p = 0; p < L::N_PLAYERS; ++p) {
    const float dx = ship(state(), p, 0) - before[p][0], dy = ship(state(), p, 1) - before[p][1];
    if (p == cfg_.player) corr_ = std::fabs(dx) + std::fabs(dy);
    if (cfg_.smooth) { visual_off_[p][0] += dx; visual_off_[p][1] += dy; }
  }
  inputs_.erase(inputs_.begin(), inputs_.lower_bound(T));
  pred_hist_.erase(pred_hist_.begin(), pred_hist_.upper_bound(T));

  // tick-lead controller: keep a few ticks of our input buffered at the server
  const int slack = pkt.slack[cfg_.player];
  if ((int64_t)T - last_adjust_tick_ >= 5) {
    if (slack < 1) { tick_adjust_ = 2; last_adjust_tick_ = T; }
    else if (slack > cfg_.lead + 3) { tick_adjust_ = -1; last_adjust_tick_ = T; }
  }
}

void ClientCore::step_local(const PlayerInput& own) {
  inputs_[*tick_] = own;
  last_own_ = own;
  set_inputs(world_, merge_inputs(cfg_.player, own, remote_last_));   // [local world state | local input, last remote input]
  world_ = sim_->step(world_);                                        // inputs consumed
  ++*tick_;
  pred_hist_[*tick_] = state();
}

void ClientCore::send_inputs() {
  std::vector<proto::InputEntry> ents;
  if (!tick_) {
    ents.push_back({0, {0.f, 0.f, 0.f}});   // hello
  } else {
    auto it = inputs_.size() > (size_t)proto::INPUT_HISTORY ? std::prev(inputs_.end(), proto::INPUT_HISTORY) : inputs_.begin();
    for (; it != inputs_.end(); ++it) ents.push_back({it->first, it->second});
  }
  net_.send(proto::encode_input(cfg_.player, ents), server_);
}

void ClientCore::frame(const PlayerInput& own) {
  std::optional<proto::StatePacket> newest;
  for (auto& [data, from] : net_.poll()) {
    auto pkt = proto::decode_state(data.data(), data.size());
    if (pkt && (!newest || pkt->tick > newest->tick)) newest = *pkt;
  }
  if (newest && (!last_srv_tick_ || newest->tick > *last_srv_tick_)) on_server_state(*newest);

  if (!tick_) { send_inputs(); return; }
  const int n_steps = 1 + tick_adjust_;
  tick_adjust_ = 0;
  for (int i = 0; i < n_steps; ++i) step_local(own);
  send_inputs();
  for (auto& v : visual_off_) { v[0] *= 0.75f; v[1] *= 0.75f; }
}

std::vector<std::string> ClientCore::hud() const {
  char b[256];
  std::vector<std::string> out;
  const auto& p = last_pkt_;
  std::snprintf(b, sizeof b, "P%d  local sim: %s   tick %u  server tick %u  ahead %d", cfg_.player, sim_->name(), tick(),
                p ? p->tick : 0, p ? (int)tick() - (int)p->tick : 0);
  out.push_back(b);
  std::snprintf(b, sizeof b, "server: input slack %d ticks   ticks w/o my input %u   replay %.1f steps/pkt   pkts %llu  gaps %llu",
                p ? p->slack[cfg_.player] : 0, p ? p->missed[cfg_.player] : 0, mean(replay_steps_),
                (unsigned long long)pkts_, (unsigned long long)gaps_);
  out.push_back(b);
  std::snprintf(b, sizeof b, "drift at reconcile: own ship %.4f (radius %.3f)   mean |state err| %.5f   last correction %.4f",
                mean(drift_own_), L::SHIP_RADIUS, mean(drift_all_), corr_);
  out.push_back(b);
  std::snprintf(b, sizeof b, "invaders alive %d   smoothing %s  ghost %s   sim step %.2f ms", invaders_alive(state()),
                cfg_.smooth ? "on" : "off", cfg_.ghost ? "on" : "off", sim_->last_step_ms());
  out.push_back(b);
  return out;
}

}  // namespace nge
