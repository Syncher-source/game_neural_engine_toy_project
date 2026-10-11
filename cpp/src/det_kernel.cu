// CUDA source of the deterministic world-model step, compiled at runtime by NVRTC with
// --fmad=false --ftz=false --prec-div=true --prec-sqrt=true and the dimensions as -D macros:
// NGE_S (state dim), NGE_M (model dim), NGE_H (hidden), NGE_O (outputs), NGE_L (layers), NGE_EPS.
// One block of NGE_H threads: thread o owns hidden unit o; thread 0 computes the LayerNorm
// statistics sequentially (same order as the CPU). Activations live in shared memory.
#include "nge/det_math.h"

extern "C" __global__ void nge_det_step(const float* __restrict__ x_in, const float* __restrict__ w, float* __restrict__ y_out) {
  __shared__ float xs[NGE_M];
  __shared__ float h[NGE_H];
  __shared__ float a[NGE_H];
  __shared__ float g[NGE_H];
  __shared__ float stats[2];
  const int o = threadIdx.x;

  if (o < NGE_M) xs[o] = o < NGE_S ? (x_in[o] - w[NGE_DET_OFF_S_MEAN(NGE_S, NGE_M, NGE_H, NGE_O) + o]) / w[NGE_DET_OFF_S_STD(NGE_S, NGE_M, NGE_H, NGE_O) + o] : x_in[o];
  __syncthreads();

  h[o] = nge_det_silu(nge_det_linear(w + NGE_DET_OFF_INP_W(NGE_S, NGE_M, NGE_H, NGE_O), NGE_M, NGE_H, o, xs, w[NGE_DET_OFF_INP_B(NGE_S, NGE_M, NGE_H, NGE_O) + o]));
  __syncthreads();

  for (int l = 0; l < NGE_L; ++l) {
    const float* lw = w + NGE_DET_OFF_LAYER(NGE_S, NGE_M, NGE_H, NGE_O, l);
    if (o == 0) nge_det_ln_stats(h, NGE_H, NGE_EPS, &stats[0], &stats[1]);
    __syncthreads();
    a[o] = nge_det_ln_apply(h[o], stats[0], stats[1], lw[NGE_DET_LAYER_LN_W + o], lw[NGE_DET_LAYER_LN_B(NGE_H) + o]);
    __syncthreads();
    g[o] = nge_det_silu(nge_det_linear(lw + NGE_DET_LAYER_W1(NGE_H), NGE_H, NGE_H, o, a, lw[NGE_DET_LAYER_B1(NGE_H) + o]));
    __syncthreads();
    h[o] = h[o] + nge_det_linear(lw + NGE_DET_LAYER_W2(NGE_H), NGE_H, NGE_H, o, g, lw[NGE_DET_LAYER_B2(NGE_H) + o]);
    __syncthreads();
  }

  if (o < NGE_O) y_out[o] = nge_det_linear(w + NGE_DET_OFF_OUT_W(NGE_S, NGE_M, NGE_H, NGE_O, NGE_L), NGE_H, NGE_O, o, h, w[NGE_DET_OFF_OUT_B(NGE_S, NGE_M, NGE_H, NGE_O, NGE_L) + o]);
}
