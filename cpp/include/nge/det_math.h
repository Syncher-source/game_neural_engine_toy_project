// Fixed-order fp32 arithmetic shared by the CPU and the CUDA implementation of the world model
// (`src/det_sim.cpp`, `src/det_kernel.cu`). Compiled both by the host compiler and by NVRTC.
//
// Every operation is an IEEE-754 single-precision operation with round-to-nearest-even, in a
// fixed order: fmaf is one correctly rounded fused multiply-add on both sides, sums are
// sequential, exp is our own polynomial (libm expf and CUDA expf are not bit-compatible),
// division and sqrt are correctly rounded (CUDA: --prec-div=true --prec-sqrt=true, no FTZ;
// host: SSE/AVX scalar ops, no fast-math, no contraction). Same bits in -> same bits out on any
// CPU and any NVIDIA GPU. Do not "simplify" an expression here without re-checking both sides.
#pragma once

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define NGE_DET_FN __device__ __forceinline__
#define NGE_DET_INT_AS_FLOAT(i) __int_as_float(i)
#else
#include <math.h>
#include <string.h>
#define NGE_DET_FN static inline
static inline float nge_det_int_as_float_(int i) { float f; memcpy(&f, &i, 4); return f; }
#define NGE_DET_INT_AS_FLOAT(i) nge_det_int_as_float_(i)
#endif

// exp(x) for the SiLU. Range reduction x = k*ln2 + r, |r| <= ln2/2, degree-7 Horner polynomial
// in fmaf, times 2^k built from the exponent bits (exact). Inputs are clamped to [-80, 80]:
// outside that range the SiLU result is numerically indistinguishable from 0 / x anyway.
NGE_DET_FN float nge_det_exp(float x) {
  x = x < -80.0f ? -80.0f : (x > 80.0f ? 80.0f : x);
  const float k = rintf(x * 1.44269504088896341f);
  float r = fmaf(-k, 0.693145751953125f, x);          // ln2 high part (exact in fp32)
  r = fmaf(-k, 1.42860682030941723e-6f, r);           // ln2 low part
  float p = 1.98412698412698413e-4f;                  // 1/7!
  p = fmaf(p, r, 1.38888888888888889e-3f);            // 1/6!
  p = fmaf(p, r, 8.33333333333333333e-3f);            // 1/5!
  p = fmaf(p, r, 4.16666666666666667e-2f);            // 1/4!
  p = fmaf(p, r, 1.66666666666666667e-1f);            // 1/3!
  p = fmaf(p, r, 0.5f);
  p = fmaf(p, r, 1.0f);
  p = fmaf(p, r, 1.0f);
  return p * NGE_DET_INT_AS_FLOAT(((int)k + 127) << 23);
}

NGE_DET_FN float nge_det_silu(float x) { return x / (1.0f + nge_det_exp(-x)); }

// Linear layer output o: bias, then fmaf over the inputs in ascending order. wT is the weight
// matrix transposed to [n_in][n_out] so that neighbouring outputs read neighbouring addresses
// (coalesced on the GPU, one SIMD vector on the CPU).
NGE_DET_FN float nge_det_linear(const float* wT, int n_in, int n_out, int o, const float* x, float bias) {
  float acc = bias;
  for (int k = 0; k < n_in; ++k) acc = fmaf(wT[k * n_out + o], x[k], acc);
  return acc;
}

// LayerNorm statistics over n values, sequential sums (biased variance, like torch).
NGE_DET_FN void nge_det_ln_stats(const float* x, int n, float eps, float* mean, float* inv_std) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s = s + x[i];
  const float m = s / (float)n;
  float v = 0.0f;
  for (int i = 0; i < n; ++i) { const float d = x[i] - m; v = fmaf(d, d, v); }
  v = v / (float)n;
  *mean = m;
  *inv_std = 1.0f / sqrtf(v + eps);
}

NGE_DET_FN float nge_det_ln_apply(float x, float mean, float inv_std, float w, float b) {
  return fmaf((x - mean) * inv_std, w, b);
}

// Weight blob layout (all fp32, matrices transposed to [in][out]), shared by both sides:
//   s_mean[S], s_std[S], inp_wT[M*H], inp_b[H],
//   per layer: ln_w[H], ln_b[H], w1T[H*H], b1[H], w2T[H*H], b2[H],
//   out_wT[H*O], out_b[O]
// with S = state dim, M = model dim (state + inputs), H = hidden, O = n_cont + n_bin.
#define NGE_DET_OFF_S_MEAN(S, M, H, O) 0
#define NGE_DET_OFF_S_STD(S, M, H, O) (S)
#define NGE_DET_OFF_INP_W(S, M, H, O) (2 * (S))
#define NGE_DET_OFF_INP_B(S, M, H, O) (2 * (S) + (M) * (H))
#define NGE_DET_OFF_LAYER(S, M, H, O, l) (2 * (S) + (M) * (H) + (H) + (l) * (2 * (H) * (H) + 4 * (H)))
#define NGE_DET_LAYER_LN_W 0
#define NGE_DET_LAYER_LN_B(H) (H)
#define NGE_DET_LAYER_W1(H) (2 * (H))
#define NGE_DET_LAYER_B1(H) (2 * (H) + (H) * (H))
#define NGE_DET_LAYER_W2(H) (3 * (H) + (H) * (H))
#define NGE_DET_LAYER_B2(H) (3 * (H) + 2 * (H) * (H))
#define NGE_DET_OFF_OUT_W(S, M, H, O, L) NGE_DET_OFF_LAYER(S, M, H, O, L)
#define NGE_DET_OFF_OUT_B(S, M, H, O, L) (NGE_DET_OFF_LAYER(S, M, H, O, L) + (H) * (O))
#define NGE_DET_BLOB_SIZE(S, M, H, O, L) (NGE_DET_OFF_OUT_B(S, M, H, O, L) + (O))
