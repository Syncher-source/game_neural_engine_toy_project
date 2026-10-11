// Headless scripted client (no SDL needed) for testing the netcode.
//   nge_bot --player 1 [--server 127.0.0.1:9999] [--bot chaser|random|idle] [--sim det|onnx] [--classic] [--model path] [--loss .. --latency ..]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

#include "nge/args.h"
#include "nge/assets.h"
#include "nge/bot.h"
#include "nge/client_core.h"
#include "nge/sim_factory.h"

using namespace nge;

int main(int argc, char** argv) {
  Args a(argc, argv);
  ClientConfig cfg; cfg.player = a.geti("--player", 1); cfg.lead = a.geti("--lead", 3);
  const std::string srv = a.get("--server", "127.0.0.1:9999");
  const auto colon = srv.rfind(':');
  net::Addr server = net::resolve(srv.substr(0, colon), (uint16_t)std::atoi(srv.c_str() + colon + 1));
  if (!server.valid()) { std::fprintf(stderr, "cannot resolve %s\n", srv.c_str()); return 1; }
  net::NetParams np{a.getd("--loss", 0), a.getd("--latency", 0), a.getd("--jitter", 0)};
  net::UdpSocket sock; sock.bind("0.0.0.0", 0);
  ClientCore core(cfg, make_sim(a, "bot"), sock, server, np);
  BotInput bot(a.get("--bot", "random"), cfg.player, (unsigned)cfg.player);
  const int max_ticks = a.geti("--max-ticks", 0);
  const double log_every = a.getd("--log-every", 0.0);
  std::printf("[client] player %d -> %s  local sim=%s  bot=%s\n", cfg.player, server.str().c_str(), core.sim().name(), a.get("--bot", "random").c_str());

  using clock = std::chrono::steady_clock;
  auto next = clock::now();
  const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(L::DT));
  double last_log = net::now_s();
  for (;;) {
    core.frame(bot.read(core.state()));
    if (log_every > 0 && net::now_s() - last_log >= log_every && core.synced()) {
      last_log = net::now_s();
      std::printf("[client]");
      for (auto& l : core.hud()) std::printf(" %s |", l.c_str());
      std::printf("\n"); std::fflush(stdout);
    }
    if (max_ticks && core.synced() && core.tick() >= (uint32_t)max_ticks) break;
    next += dt;
    if (next < clock::now() - std::chrono::milliseconds(500)) next = clock::now();
    std::this_thread::sleep_until(next);
  }
  return 0;
}
