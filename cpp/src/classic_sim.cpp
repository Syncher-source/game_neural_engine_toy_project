#include "nge/classic_sim.h"

#include <algorithm>
#include <cmath>

namespace nge {

void invader_pos(const State& s, int i, double& x, double& y) {
  const int row = i / L::INV_COLS, col = i % L::INV_COLS;
  x = (double)L::INV_X0 + (double)form(s, 0) + col * (double)L::INV_DX;
  y = (double)L::INV_Y0 + (double)form(s, 1) + row * (double)L::INV_DY;
}

// Port of nge/sim.py (single world). Operation order mirrors the numpy code so
// both produce the same trajectories up to float rounding.
ModelVector ClassicSim::step(const ModelVector& x) {
  ModelVector out{};                 // input slots stay zero: consumed
  state_of(out) = step_state(x);
  return out;
}

State ClassicSim::step_state(const ModelVector& x) {
  using namespace L;
  State s = state_of(x);
  const Input u = inputs_of(x);
  float ax[N_PLAYERS], ay[N_PLAYERS];
  bool fire[N_PLAYERS];
  for (int p = 0; p < N_PLAYERS; ++p) {
    ax[p] = std::clamp(u[p * INPUT_STRIDE + 0], -1.0f, 1.0f);
    ay[p] = std::clamp(u[p * INPUT_STRIDE + 1], -1.0f, 1.0f);
    fire[p] = u[p * INPUT_STRIDE + 2] > 0.5f;
  }

  // ---- ship dynamics
  for (int p = 0; p < N_PLAYERS; ++p) {
    float& x = ship(s, p, 0); float& y = ship(s, p, 1);
    float& vx = ship(s, p, 2); float& vy = ship(s, p, 3);
    vx = (vx + ax[p] * SHIP_ACCEL * DT) * SHIP_DAMPING;
    vy = (vy + ay[p] * SHIP_ACCEL * DT) * SHIP_DAMPING;
    const float speed = std::sqrt(vx * vx + vy * vy);
    const float scale = std::min(1.0f, SHIP_MAX_SPEED / std::max(speed, 1e-9f));
    vx *= scale; vy *= scale;
    x += vx * DT; y += vy * DT;
    const bool hx = x < SHIP_X_MIN || x > SHIP_X_MAX;
    const bool hy = y < SHIP_Y_MIN || y > SHIP_Y_MAX;
    x = std::clamp(x, SHIP_X_MIN, SHIP_X_MAX);
    y = std::clamp(y, SHIP_Y_MIN, SHIP_Y_MAX);
    if (hx) vx = 0.0f;
    if (hy) vy = 0.0f;
  }
  // ship-ship collision: both speeds go to zero, ships are separated
  {
    const float dx = ship(s, 1, 0) - ship(s, 0, 0), dy = ship(s, 1, 1) - ship(s, 0, 1);
    const float dist = std::sqrt(dx * dx + dy * dy);
    if (dist < 2 * SHIP_RADIUS) {
      float nx = 1.0f, ny = 0.0f;
      if (dist > 1e-6f) { nx = dx / dist; ny = dy / dist; }
      const float mx = 0.5f * (ship(s, 0, 0) + ship(s, 1, 0)), my = 0.5f * (ship(s, 0, 1) + ship(s, 1, 1));
      ship(s, 0, 0) = mx - nx * SHIP_RADIUS; ship(s, 0, 1) = my - ny * SHIP_RADIUS;
      ship(s, 1, 0) = mx + nx * SHIP_RADIUS; ship(s, 1, 1) = my + ny * SHIP_RADIUS;
      for (int p = 0; p < N_PLAYERS; ++p) {
        ship(s, p, 2) = 0.0f; ship(s, p, 3) = 0.0f;
        ship(s, p, 0) = std::clamp(ship(s, p, 0), SHIP_X_MIN, SHIP_X_MAX);
        ship(s, p, 1) = std::clamp(ship(s, p, 1), SHIP_Y_MIN, SHIP_Y_MAX);
      }
    }
  }
  for (int p = 0; p < N_PLAYERS; ++p) ship(s, p, 4) = std::max(0.0f, ship(s, p, 4) - FIRE_COOLDOWN_DECAY);

  // ---- bullets
  bool active[N_PLAYERS][N_BULLETS];
  for (int p = 0; p < N_PLAYERS; ++p)
    for (int b = 0; b < N_BULLETS; ++b) {
      active[p][b] = bullet(s, p, b, 2) > 0.5f;
      bullet(s, p, b, 1) = active[p][b] ? bullet(s, p, b, 1) - BULLET_SPEED * DT : 0.0f;
      bullet(s, p, b, 0) = active[p][b] ? bullet(s, p, b, 0) : 0.0f;
      if (active[p][b] && bullet(s, p, b, 1) < 0.0f) active[p][b] = false;
    }
  // bullet / invader hits (alive flags are read before any kill is applied)
  bool alive[N_INVADERS], killed[N_INVADERS] = {};
  double ix[N_INVADERS], iy[N_INVADERS];
  for (int i = 0; i < N_INVADERS; ++i) { alive[i] = inv(s, i) > 0.5f; invader_pos(s, i, ix[i], iy[i]); }
  for (int p = 0; p < N_PLAYERS; ++p)
    for (int b = 0; b < N_BULLETS; ++b) {
      if (!active[p][b]) continue;
      const double bx = bullet(s, p, b, 0), by = bullet(s, p, b, 1);
      bool hit = false;
      for (int i = 0; i < N_INVADERS; ++i)
        if (alive[i] && std::fabs(bx - ix[i]) < (double)INV_HALF_W && std::fabs(by - iy[i]) < (double)INV_HALF_H) { killed[i] = true; hit = true; }
      if (hit) active[p][b] = false;
    }
  for (int i = 0; i < N_INVADERS; ++i) if (killed[i]) inv(s, i) = 0.0f;
  for (int p = 0; p < N_PLAYERS; ++p)
    for (int b = 0; b < N_BULLETS; ++b) {
      bullet(s, p, b, 2) = active[p][b] ? 1.0f : 0.0f;
      if (!active[p][b]) { bullet(s, p, b, 0) = 0.0f; bullet(s, p, b, 1) = 0.0f; }
    }
  // firing: cooldown expired and a free slot
  for (int p = 0; p < N_PLAYERS; ++p) {
    int slot = -1;
    for (int b = 0; b < N_BULLETS; ++b) if (!active[p][b]) { slot = b; break; }
    if (fire[p] && ship(s, p, 4) <= 0.0f && slot >= 0) {
      bullet(s, p, slot, 0) = ship(s, p, 0);
      bullet(s, p, slot, 1) = ship(s, p, 1) - SHIP_RADIUS;
      bullet(s, p, slot, 2) = 1.0f;
      ship(s, p, 4) = 1.0f;
    }
  }

  // ---- formation
  {
    const bool right = form(s, 2) > 0.5f;
    form(s, 0) += (right ? 1.0f : -1.0f) * FORM_SPEED * DT;
    const bool br = form(s, 0) > FORM_OX_MAX, bl = form(s, 0) < 0.0f;
    form(s, 0) = std::clamp(form(s, 0), 0.0f, FORM_OX_MAX);
    if (br) form(s, 2) = 0.0f; else if (bl) form(s, 2) = 1.0f;
    if (br || bl) form(s, 1) += FORM_STEP_DOWN;
    if (form(s, 1) > FORM_OY_MAX + 1e-4f) form(s, 1) = 0.0f;
  }
  if (invaders_alive(s) == 0) {
    for (int i = 0; i < N_INVADERS; ++i) inv(s, i) = 1.0f;
    form(s, 1) = 0.0f;
  }
  return s;
}

}  // namespace nge
