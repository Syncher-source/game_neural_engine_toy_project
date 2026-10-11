// Graphical client: SDL3 window + keyboard/gamepad, ClientCore for prediction/reconciliation,
// the renderer network for every frame.
//   nge_client --player 0 [--server host:port] [--sim det|onnx] [--model path] [--renderer path] [--cpu] [--no-interop]
//              [--bot chaser] [--loss .. --latency .. --jitter ..] [--lead N] [--no-smooth]
// Only with -DNGE_WITH_CLASSIC_SIM=ON:    --classic (hand-written simulation)
// Only with -DNGE_WITH_CLASSIC_RENDER=ON: --render classic, R key, G ghost overlay (vector drawing)
// On Windows the SDL renderer is Direct3D 12 and the renderer network's output stays on the GPU
// (shared D3D12 texture); --no-interop forces the CPU readback path.
// Keys: arrows/WASD move, space fire, H toggle correction smoothing, F12 screenshot (PNG next to the exe), Esc quit.
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "nge/args.h"
#include "nge/assets.h"
#include "nge/bot.h"
#include "nge/client_core.h"
#include "nge/gpu_interop.h"
#include "nge/neural_renderer.h"
#include "nge/png_write.h"
#include "nge/sim_factory.h"
#if NGE_WITH_CLASSIC_SIM
#include "nge/classic_sim.h"
#endif
#if NGE_WITH_CLASSIC_RENDER
#include "nge/classic_render.h"
#endif

using namespace nge;

int main(int argc, char** argv) {
  Args a(argc, argv);
  ClientConfig cfg; cfg.player = a.geti("--player", 0); cfg.lead = a.geti("--lead", 3);
  cfg.smooth = !a.has("--no-smooth"); cfg.ghost = !a.has("--no-ghost");
  const std::string srv = a.get("--server", "127.0.0.1:9999");
  const auto colon = srv.rfind(':');
  net::Addr server = net::resolve(srv.substr(0, colon), (uint16_t)std::atoi(srv.c_str() + colon + 1));
  if (!server.valid()) { std::fprintf(stderr, "cannot resolve %s\n", srv.c_str()); return 1; }
  net::NetParams np{a.getd("--loss", 0), a.getd("--latency", 0), a.getd("--jitter", 0)};
  net::UdpSocket sock; sock.bind("0.0.0.0", 0);
  ClientCore core(cfg, make_sim(a, "client"), sock, server, np);
  std::unique_ptr<BotInput> bot;
  if (a.has("--bot")) bot = std::make_unique<BotInput>(a.get("--bot", "random"), cfg.player, (unsigned)cfg.player);

  // ---- window + SDL renderer (Direct3D 12 preferred so the frame can be shared with CUDA)
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) { std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1; }
  const int size = a.geti("--size", 720);
  const std::string title = "neural game engine - player " + std::to_string(cfg.player);
  SDL_Window* win = SDL_CreateWindow(title.c_str(), size, size + 90, 0);
  if (!win) { std::fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }
  SDL_Renderer* ren = nullptr;
  if (GpuFrameInterop::supported() && !a.has("--no-interop")) ren = SDL_CreateRenderer(win, "direct3d12");
  if (!ren) ren = SDL_CreateRenderer(win, nullptr);
  if (!ren) { std::fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError()); return 1; }
  std::printf("[client] SDL renderer: %s\n", SDL_GetRendererName(ren));
  SDL_SetRenderVSync(ren, 0);

  // ---- renderer network (mandatory: it is the only thing that draws the world)
  std::unique_ptr<NeuralRenderer> nrend;
  try {
    nrend = std::make_unique<NeuralRenderer>(find_asset(a.get("--renderer", "checkpoints/renderer.onnx")), !a.has("--cpu"), 1);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[client] cannot load the renderer network: %s\n", e.what());
    return 2;
  }
  std::unique_ptr<GpuFrameInterop> interop;
  if (nrend->on_cuda() && !a.has("--no-interop")) {
    std::string err;
    interop = GpuFrameInterop::create(ren, nrend->width(), nrend->height(), err);
    if (!interop) std::fprintf(stderr, "[client] GPU interop unavailable (%s); using CPU readback\n", err.c_str());
  }
  if (interop) {   // warm-up through the GPU path; drop to readback if the output binding fails
    interop->begin_frame();
    const void* dev = nrend->render_gpu(core.world());
    if (dev) { interop->submit(dev, (size_t)nrend->width() * 4); interop->end_frame(); }
    else { interop.reset(); std::fprintf(stderr, "[client] GPU output path unavailable; using CPU readback\n"); }
  }
  SDL_Texture* frame_tex = nullptr;   // readback path texture
  if (!interop) {
    frame_tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, nrend->width(), nrend->height());
    SDL_SetTextureScaleMode(frame_tex, SDL_SCALEMODE_NEAREST);
    nrend->render(core.world());
  }
  // Startup timing: a few warm-up frames (kernel selection, GPU clocks), then mean/min over 10.
  double sum = 0, best = 1e9;
  for (int i = 0; i < 15; ++i) {
    if (interop) { interop->begin_frame(); nrend->render_gpu(core.world()); } else nrend->render(core.world());
    if (i >= 5) { sum += nrend->last_ms(); best = std::min(best, nrend->last_ms()); }
  }
  const double mean_ms = sum / 10.0;
  const char* frame_path = interop ? interop->name() : "cpu readback";
  std::printf("[client] renderer: %s %dx%d, %.2f ms/frame (min %.2f), path: %s\n", nrend->name(), nrend->width(), nrend->height(), mean_ms, best, frame_path);
  if (mean_ms > 20.0)
    std::fprintf(stderr, "[client] warning: the renderer takes %.1f ms/frame on this device, above the %d Hz budget\n", mean_ms, L::TICK_RATE);

#if NGE_WITH_CLASSIC_RENDER
  bool vector_mode = a.get("--render", "neural") == "classic";
#endif

  SDL_Gamepad* pad = nullptr;
  { int n = 0; SDL_JoystickID* ids = SDL_GetGamepads(&n); if (ids && n > 0) { pad = SDL_OpenGamepad(ids[0]); if (pad) std::printf("[client] gamepad: %s\n", SDL_GetGamepadName(pad)); } SDL_free(ids); }
  std::printf("[client] player %d -> %s  local sim=%s%s\n", cfg.player, server.str().c_str(), core.sim().name(),
              (NGE_WITH_CLASSIC_SIM || NGE_WITH_CLASSIC_RENDER) ? "" : "  [binary contains no hand-written game rules or drawing]");

  const Uint64 tick_ns = (Uint64)(1e9 / L::TICK_RATE);
  Uint64 next = SDL_GetTicksNS();
  bool running = true;
  bool want_screenshot = false;
  while (running) {
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_EVENT_QUIT) running = false;
      else if (ev.type == SDL_EVENT_KEY_DOWN) {
        if (ev.key.key == SDLK_ESCAPE) running = false;
        else if (ev.key.key == SDLK_H) { core.config().smooth = !core.config().smooth; core.reset_visual(); }
        else if (ev.key.key == SDLK_F12) want_screenshot = true;
#if NGE_WITH_CLASSIC_RENDER
        else if (ev.key.key == SDLK_G) core.config().ghost = !core.config().ghost;
        else if (ev.key.key == SDLK_R) vector_mode = !vector_mode;
#endif
      } else if (ev.type == SDL_EVENT_GAMEPAD_ADDED && !pad) {
        pad = SDL_OpenGamepad(ev.gdevice.which);
        if (pad) std::printf("[client] gamepad: %s\n", SDL_GetGamepadName(pad));
      }
    }
    // ---- local input: keyboard, gamepad, or bot
    PlayerInput own{};
    if (bot) own = bot->read(core.state());
    else {
      const bool* k = SDL_GetKeyboardState(nullptr);
      own[0] = (float)(k[SDL_SCANCODE_RIGHT] || k[SDL_SCANCODE_D]) - (float)(k[SDL_SCANCODE_LEFT] || k[SDL_SCANCODE_A]);
      own[1] = (float)(k[SDL_SCANCODE_DOWN] || k[SDL_SCANCODE_S]) - (float)(k[SDL_SCANCODE_UP] || k[SDL_SCANCODE_W]);
      own[2] = k[SDL_SCANCODE_SPACE] ? 1.f : 0.f;
      if (pad) {
        const float jx = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTX) / 32767.f;
        const float jy = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTY) / 32767.f;
        if (std::fabs(jx) > 0.15f || std::fabs(jy) > 0.15f) { own[0] = jx; own[1] = jy; }
        if (SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_SOUTH) || SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_EAST) ||
            SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 8000) own[2] = 1.f;
      }
    }
    core.frame(own);

    // ---- render: the state (ships at their visually smoothed positions) goes to the renderer network
    SDL_SetRenderDrawColor(ren, 12, 12, 24, 255);
    SDL_RenderClear(ren);
    ModelVector rs = core.world();            // same vector type the world model uses; renderer ignores the input slots
    std::array<std::array<float, 2>, L::N_PLAYERS> spos;
    for (int p = 0; p < L::N_PLAYERS; ++p) { spos[p] = core.render_pos(p); ship(state_of(rs), p, 0) = spos[p][0]; ship(state_of(rs), p, 1) = spos[p][1]; }
    bool neural_frame = true;
#if NGE_WITH_CLASSIC_RENDER
    if (vector_mode) { neural_frame = false; classic_render::draw_world(ren, size, state_of(rs), spos); }
#endif
    if (neural_frame) {
      SDL_Texture* tex = nullptr;
      if (interop) {
        interop->begin_frame();
        const void* dev = nrend->render_gpu(rs);
        if (dev) { interop->submit(dev, (size_t)nrend->width() * 4); tex = interop->texture(); }
      }
      if (!tex) {
        const auto& px = nrend->render(rs);
        if (!frame_tex) { frame_tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, nrend->width(), nrend->height()); SDL_SetTextureScaleMode(frame_tex, SDL_SCALEMODE_NEAREST); }
        SDL_UpdateTexture(frame_tex, nullptr, px.data(), nrend->width() * 4);
        tex = frame_tex;
      }
      SDL_FRect dst{0, 0, (float)size, (float)size};
      SDL_RenderTexture(ren, tex, nullptr, &dst);
    }
#if NGE_WITH_CLASSIC_RENDER
    classic_render::draw_overlays(ren, size, core.config().ghost ? core.ghost() : std::nullopt, cfg.player, spos);
#endif
    // ---- HUD (SDL debug text)
    SDL_SetRenderDrawColor(ren, 200, 200, 220, 255);
    float y = (float)size + 8;
    if (core.synced()) for (auto& l : core.hud()) { SDL_RenderDebugText(ren, 8, y, l.c_str()); y += 14; }
    else SDL_RenderDebugText(ren, 8, y, ("waiting for server " + srv).c_str());
    char nb[220];
    std::snprintf(nb, sizeof nb, "renderer: %s %.2f ms [%s]   netsim: loss %.0f%%  latency %.0fms  jitter %.0fms   keys: WASD/arrows, space, H smooth%s",
                  neural_frame ? nrend->name() : "vector", neural_frame ? nrend->last_ms() : 0.0, neural_frame ? frame_path : "classic drawing",
                  np.loss * 100, np.latency_ms, np.jitter_ms, NGE_WITH_CLASSIC_RENDER ? ", G ghost, R renderer" : "");
    SDL_RenderDebugText(ren, 8, y, nb);
    if (want_screenshot) {
      want_screenshot = false;
      if (SDL_Surface* shot = SDL_RenderReadPixels(ren, nullptr)) {
        SDL_Surface* rgba = SDL_ConvertSurface(shot, SDL_PIXELFORMAT_RGBA32);
        if (rgba) {
          static int shot_n = 0;
          const std::string path = (exe_dir() / ("screenshot_" + std::to_string(++shot_n) + ".png")).string();
          std::printf("[client] %s %s\n", png::write_rgba(path, rgba->w, rgba->h, (const uint8_t*)rgba->pixels, rgba->pitch) ? "screenshot saved:" : "screenshot FAILED:", path.c_str());
          SDL_DestroySurface(rgba);
        }
        SDL_DestroySurface(shot);
      }
    }
    SDL_RenderPresent(ren);
    if (interop && neural_frame) interop->end_frame();

    next += tick_ns;
    const Uint64 now = SDL_GetTicksNS();
    if (next > now) SDL_DelayPrecise(next - now); else if (now - next > 500000000ull) next = now;
  }
  interop.reset();
  if (frame_tex) SDL_DestroyTexture(frame_tex);
  if (pad) SDL_CloseGamepad(pad);
  SDL_DestroyRenderer(ren); SDL_DestroyWindow(win); SDL_Quit();
  return 0;
}
