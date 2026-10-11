// Non-blocking UDP socket (Winsock / BSD) plus a loss/latency/jitter simulator.
#pragma once
#include <cstdint>
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace nge::net {

struct Addr {
  uint32_t ip = 0;   // host byte order
  uint16_t port = 0;
  bool valid() const { return port != 0; }
  bool operator==(const Addr& o) const { return ip == o.ip && port == o.port; }
  std::string str() const;
};
Addr resolve(const std::string& host, uint16_t port);   // numeric IPv4 or DNS name

class UdpSocket {
 public:
  UdpSocket();
  ~UdpSocket();
  UdpSocket(const UdpSocket&) = delete;
  UdpSocket& operator=(const UdpSocket&) = delete;
  bool bind(const std::string& host, uint16_t port);
  bool send(const std::vector<uint8_t>& data, const Addr& to);
  bool recv(std::vector<uint8_t>& data, Addr& from);   // false when nothing is pending
 private:
  intptr_t fd_ = -1;
};

struct NetParams {
  double loss = 0.0;        // drop probability per direction
  double latency_ms = 0.0;  // one-way added latency
  double jitter_ms = 0.0;
};

// Mirrors nge/netsim.py: packets are dropped / delayed on both send and receive.
class LossySocket {
 public:
  LossySocket(UdpSocket& sock, NetParams p, unsigned seed);
  void send(const std::vector<uint8_t>& data, const Addr& to);
  std::vector<std::pair<std::vector<uint8_t>, Addr>> poll();   // flush due sends, return due receives
  uint64_t sent = 0, dropped = 0, received = 0;

 private:
  struct Pending { double due; uint64_t seq; std::vector<uint8_t> data; Addr addr;
                   bool operator>(const Pending& o) const { return due != o.due ? due > o.due : seq > o.seq; } };
  double delay();
  UdpSocket& sock_;
  NetParams p_;
  std::mt19937 rng_;
  std::priority_queue<Pending, std::vector<Pending>, std::greater<Pending>> out_, in_;
  uint64_t seq_ = 0;
};

double now_s();   // monotonic seconds

}  // namespace nge::net
