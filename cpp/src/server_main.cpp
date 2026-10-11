// Authoritative world server (C++ port of nge/server.py). Same wire format: Python clients can connect.
//   nge_server [--port 9999] [--sim det|onnx] [--model path] [--classic] [--cpu] [--loss 0.2 --latency 40 --jitter 5]
#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <thread>

#include "nge/args.h"
#include "nge/assets.h"
#include "nge/net.h"
#include "nge/protocol.h"
#include "nge/sim_factory.h"
#include "nge/state.h"

using namespace nge;

struct Player {
  net::Addr addr;
  std::map<uint32_t, PlayerInput> inputs;
  PlayerInput last{};
  uint16_t missed = 0;
  uint64_t received = 0;
  double last_seen = 0;
};

int main(int argc, char** argv) {
  Args a(argc, argv);
  const int port = a.geti("--port", 9999);
  net::NetParams np{a.getd("--loss", 0), a.getd("--latency", 0), a.getd("--jitter", 0)};
  const int max_ticks = a.geti("--max-ticks", 0);
  const double log_every = a.getd("--log-every", 1.0);

  std::unique_ptr<ISim> sim = make_sim(a, "server");

  net::UdpSocket sock;
  if (!sock.bind("0.0.0.0", (uint16_t)port)) { std::fprintf(stderr, "bind failed on port %d\n", port); return 1; }
  net::LossySocket net(sock, np, 1);
  std::printf("[server] sim=%s tick=%dHz listening on 0.0.0.0:%d netsim(loss=%.2f latency=%.0fms jitter=%.0fms)%s\n",
              sim->name(), L::TICK_RATE, port, np.loss, np.latency_ms, np.jitter_ms,
              NGE_WITH_CLASSIC_SIM ? "" : "  [binary contains no hand-written game rules]");

  Player players[L::N_PLAYERS];
  ModelVector world = pack(initial_state(), Input{});   // the one vector: [world state | inputs]
  uint32_t tick = 0;
  Input u{};
  using clock = std::chrono::steady_clock;
  auto next = clock::now();
  double last_log = net::now_s(), step_acc = 0; int step_n = 0;
  const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(L::DT));

  for (;;) {
    const double now = net::now_s();
    for (auto& [data, from] : net.poll()) {
      auto dec = proto::decode_input(data.data(), data.size());
      if (!dec || dec->first < 0 || dec->first >= L::N_PLAYERS) continue;
      Player& pl = players[dec->first];
      pl.addr = from; pl.last_seen = now; ++pl.received;
      for (auto& e : dec->second) if (e.tick >= tick) pl.inputs[e.tick] = e.input;
    }
    int16_t slack[L::N_PLAYERS]; uint16_t missed[L::N_PLAYERS];
    for (int p = 0; p < L::N_PLAYERS; ++p) {
      Player& pl = players[p];
      if (!pl.addr.valid() || now - pl.last_seen > 3.0) { pl.last = {}; slack[p] = 0; }
      else {
        auto it = pl.inputs.find(tick);
        if (it == pl.inputs.end()) ++pl.missed; else { pl.last = it->second; pl.inputs.erase(it); }
        slack[p] = pl.inputs.empty() ? 0 : (int16_t)((int64_t)pl.inputs.rbegin()->first - tick);
        pl.inputs.erase(pl.inputs.begin(), pl.inputs.lower_bound(tick));
      }
      missed[p] = pl.missed;
      for (int k = 0; k < L::INPUT_STRIDE; ++k) u[p * L::INPUT_STRIDE + k] = pl.last[k];
    }
    const auto t0 = clock::now();
    set_inputs(world, u);                        // write the remote players' inputs into the vector ...
    world = sim->step(world);                    // ... step it; the inputs come back consumed (zero)
    step_acc += std::chrono::duration<double, std::milli>(clock::now() - t0).count(); ++step_n;
    ++tick;
    const State& state = state_of(world);

    proto::StatePacket pkt; pkt.tick = tick; pkt.last_inputs = u; pkt.state = state;
    for (int p = 0; p < L::N_PLAYERS; ++p) { pkt.slack[p] = slack[p]; pkt.missed[p] = missed[p]; }
    const auto bytes = proto::encode_state(pkt);
    for (auto& pl : players) if (pl.addr.valid()) net.send(bytes, pl.addr);

    if (log_every > 0 && now - last_log >= log_every) {
      last_log = now;
      std::printf("[server] tick=%u step=%.2fms |", tick, step_acc / std::max(step_n, 1));
      for (int p = 0; p < L::N_PLAYERS; ++p)
        std::printf(" p%d:%s rx=%llu missed=%u slack=%d", p, players[p].addr.valid() ? "on " : "off",
                    (unsigned long long)players[p].received, players[p].missed, slack[p]);
      std::printf(" | ships (%.2f,%.2f) (%.2f,%.2f) alive=%d\n", ship(state, 0, 0), ship(state, 0, 1), ship(state, 1, 0), ship(state, 1, 1), invaders_alive(state));
      std::fflush(stdout);
      step_acc = 0; step_n = 0;
    }
    if (max_ticks && tick >= (uint32_t)max_ticks) break;
    next += dt;
    if (next < clock::now() - std::chrono::milliseconds(500)) next = clock::now();
    std::this_thread::sleep_until(next);
  }
  return 0;
}
