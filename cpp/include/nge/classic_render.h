// Hand-written vector drawing of the world (the look the renderer network was trained to
// reproduce) plus debug overlays. Compiled into nge_client only with -DNGE_WITH_CLASSIC_RENDER=ON;
// the default game build draws exclusively what the renderer network outputs.
#pragma once
#include <array>
#include <optional>

#include "nge/state.h"

struct SDL_Renderer;

namespace nge::classic_render {

using Pos = std::array<std::array<float, 2>, L::N_PLAYERS>;

void draw_world(SDL_Renderer* ren, int size, const State& s, const Pos& ship_pos);
void draw_overlays(SDL_Renderer* ren, int size, const std::optional<State>& ghost, int own, const Pos& ship_pos);

}  // namespace nge::classic_render
