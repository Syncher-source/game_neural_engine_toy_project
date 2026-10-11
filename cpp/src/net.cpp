#include "nge/net.h"

#include <chrono>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
static void close_sock(sock_t s) { closesocket(s); }
static void set_nonblocking(sock_t s) { u_long nb = 1; ioctlsocket(s, FIONBIO, &nb); }
static void net_init() { static bool done = false; if (!done) { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); done = true; } }
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
static void close_sock(sock_t s) { ::close(s); }
static void set_nonblocking(sock_t s) { fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK); }
static void net_init() {}
#endif

namespace nge::net {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string Addr::str() const {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%u.%u.%u.%u:%u", (ip >> 24) & 255, (ip >> 16) & 255, (ip >> 8) & 255, ip & 255, port);
  return buf;
}

Addr resolve(const std::string& host, uint16_t port) {
  net_init();
  Addr a; a.port = port;
  addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_DGRAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) == 0 && res) {
    a.ip = ntohl(((sockaddr_in*)res->ai_addr)->sin_addr.s_addr);
    freeaddrinfo(res);
  } else {
    a.port = 0;
  }
  return a;
}

static sockaddr_in to_sockaddr(const Addr& a) {
  sockaddr_in sa{}; sa.sin_family = AF_INET; sa.sin_port = htons(a.port); sa.sin_addr.s_addr = htonl(a.ip);
  return sa;
}

UdpSocket::UdpSocket() {
  net_init();
  fd_ = (intptr_t)::socket(AF_INET, SOCK_DGRAM, 0);
  set_nonblocking((sock_t)fd_);
}
UdpSocket::~UdpSocket() { if (fd_ >= 0) close_sock((sock_t)fd_); }

bool UdpSocket::bind(const std::string& host, uint16_t port) {
  Addr a = host.empty() || host == "0.0.0.0" ? Addr{0, port} : resolve(host, port);
  if (!host.empty() && host != "0.0.0.0" && !a.valid()) return false;
  a.port = port;
  sockaddr_in sa = to_sockaddr(a);
  return ::bind((sock_t)fd_, (sockaddr*)&sa, sizeof sa) == 0;
}

bool UdpSocket::send(const std::vector<uint8_t>& data, const Addr& to) {
  sockaddr_in sa = to_sockaddr(to);
  return ::sendto((sock_t)fd_, (const char*)data.data(), (int)data.size(), 0, (sockaddr*)&sa, sizeof sa) >= 0;
}

bool UdpSocket::recv(std::vector<uint8_t>& data, Addr& from) {
  data.resize(2048);
  sockaddr_in sa{}; socklen_t sl = sizeof sa;
  const auto n = ::recvfrom((sock_t)fd_, (char*)data.data(), (int)data.size(), 0, (sockaddr*)&sa, &sl);
  if (n < 0) { data.clear(); return false; }
  data.resize((size_t)n);
  from.ip = ntohl(sa.sin_addr.s_addr); from.port = ntohs(sa.sin_port);
  return true;
}

LossySocket::LossySocket(UdpSocket& sock, NetParams p, unsigned seed) : sock_(sock), p_(p), rng_(seed) {}

double LossySocket::delay() {
  const double j = p_.jitter_ms > 0 ? std::uniform_real_distribution<double>(-p_.jitter_ms, p_.jitter_ms)(rng_) : 0.0;
  return std::max(0.0, (p_.latency_ms + j) / 1000.0);
}

void LossySocket::send(const std::vector<uint8_t>& data, const Addr& to) {
  ++sent;
  if (p_.loss > 0 && std::uniform_real_distribution<double>(0, 1)(rng_) < p_.loss) { ++dropped; return; }
  if (p_.latency_ms == 0 && p_.jitter_ms == 0) { sock_.send(data, to); return; }
  out_.push(Pending{now_s() + delay(), ++seq_, data, to});
}

std::vector<std::pair<std::vector<uint8_t>, Addr>> LossySocket::poll() {
  const double now = now_s();
  while (!out_.empty() && out_.top().due <= now) { sock_.send(out_.top().data, out_.top().addr); out_.pop(); }
  std::vector<uint8_t> buf; Addr from;
  while (sock_.recv(buf, from)) {
    if (p_.loss > 0 && std::uniform_real_distribution<double>(0, 1)(rng_) < p_.loss) { ++dropped; continue; }
    in_.push(Pending{now + delay(), ++seq_, buf, from});
  }
  std::vector<std::pair<std::vector<uint8_t>, Addr>> out;
  while (!in_.empty() && in_.top().due <= now) { out.emplace_back(in_.top().data, in_.top().addr); in_.pop(); ++received; }
  return out;
}

}  // namespace nge::net
