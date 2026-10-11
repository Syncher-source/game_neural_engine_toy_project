#include "nge/classic_render.h"

#include <SDL3/SDL.h>

#include "nge/classic_sim.h"

namespace nge::classic_render {
namespace {

const SDL_FColor SHIP_COL[2] = {{0.31f, 0.78f, 1.f, 1.f}, {1.f, 0.59f, 0.31f, 1.f}};
const SDL_FColor GHOST_COL[2] = {{0.16f, 0.39f, 0.5f, 1.f}, {0.5f, 0.29f, 0.16f, 1.f}};
const Uint8 INV_COL[3][3] = {{180, 255, 120}, {255, 230, 100}, {255, 120, 160}};

void ship_poly(SDL_Renderer* ren, int size, float x, float y, SDL_FColor c, bool outline) {
  const float cx = x * size, cy = y * size, R = L::SHIP_RADIUS * size;
  const SDL_FPoint pts[5] = {{cx, cy - R}, {cx - R, cy + R * 0.9f}, {cx, cy + R * 0.4f}, {cx + R, cy + R * 0.9f}, {cx, cy - R}};
  if (outline) {
    SDL_SetRenderDrawColorFloat(ren, c.r, c.g, c.b, c.a);
    SDL_RenderLines(ren, pts, 5);
  } else {
    SDL_Vertex v[4];
    for (int i = 0; i < 4; ++i) { v[i].position = pts[i]; v[i].color = c; v[i].tex_coord = {0, 0}; }
    const int idx[6] = {0, 1, 2, 0, 2, 3};
    SDL_RenderGeometry(ren, nullptr, v, 4, idx, 6);
  }
}

void rect(SDL_Renderer* ren, int size, float x, float y, float hw, float hh, Uint8 r, Uint8 g, Uint8 b) {
  SDL_SetRenderDrawColor(ren, r, g, b, 255);
  SDL_FRect fr{(x - hw) * size, (y - hh) * size, 2 * hw * size, 2 * hh * size};
  SDL_RenderFillRect(ren, &fr);
}

}  // namespace

void draw_world(SDL_Renderer* ren, int size, const State& s, const Pos& ship_pos) {
  SDL_SetRenderDrawColor(ren, 40, 40, 70, 255);
  SDL_RenderLine(ren, 0, L::SHIP_Y_MIN * size, (float)size, L::SHIP_Y_MIN * size);
  for (int i = 0; i < L::N_INVADERS; ++i)
    if (inv(s, i) > 0.5f) {
      double x, y; invader_pos(s, i, x, y);
      const auto& c = INV_COL[i / L::INV_COLS];
      rect(ren, size, (float)x, (float)y, L::INV_HALF_W, L::INV_HALF_H, c[0], c[1], c[2]);
    }
  for (int p = 0; p < L::N_PLAYERS; ++p)
    for (int b = 0; b < L::N_BULLETS; ++b)
      if (bullet(s, p, b, 2) > 0.5f) rect(ren, size, bullet(s, p, b, 0), bullet(s, p, b, 1), 2.f / size, 6.f / size, 255, 255, 255);
  for (int p = 0; p < L::N_PLAYERS; ++p) ship_poly(ren, size, ship_pos[p][0], ship_pos[p][1], SHIP_COL[p], false);
}

void draw_overlays(SDL_Renderer* ren, int size, const std::optional<State>& ghost, int own, const Pos& ship_pos) {
  if (ghost)
    for (int p = 0; p < L::N_PLAYERS; ++p) ship_poly(ren, size, ship(*ghost, p, 0), ship(*ghost, p, 1), GHOST_COL[p], true);
  ship_poly(ren, size, ship_pos[own][0], ship_pos[own][1], {1, 1, 1, 1}, true);
}

}  // namespace nge::classic_render
