#pragma once
// Helios engine glue kernels: embedding gather, stream expand/collapse, residual adds,
// MoE weighted sum, dtype casts, sampling helpers. Raw pointers, stream last.
#include "../cuda/cuda_shim.hpp"

namespace helios { namespace glue {

// Embedding lookup: bf16 table [vocab, hidden] -> fp16 rows [n, hidden].
void embed_gather(const void* embed_bf16, const int* ids, int n, half* out, int hidden, Stream s = 0);

// mHC stream init: streams (R,H,D) fp32 = fp16 h broadcast over the H hyper-streams.
void stream_expand(const half* h, float* streams, int n, int hidden, Stream s = 0);
// Final collapse used when the model has no hc_head weights: mean over the H streams.
void gather_rows(const half* src, const int64_t* idx, half* dst, int n, int h, Stream s = 0);
void scatter_add_rows(float* y, const float* src, const int64_t* idx, const half* w, int n, int h,
                      Stream s = 0);
void stream_mean(const float* streams, half* out, int R, int H, int D, Stream s = 0);

// Residual add: acc += x (fp16).
void add_inplace(half* acc, const half* x, size_t n, Stream s = 0);

// MoE combine: out[t] = sum_k w[t,k] * exp_out[t,k]; exp_out layout [n, k, hidden] fp16,
// w layout [n, k] fp32. Writes fp16 out[n, hidden] (overwrites).
void moe_combine(half* out, const half* exp_out, const float* w, int n, int k, int hidden, Stream s = 0);

// Casts
void cast_f16_f32(const half* src, float* dst, size_t n, Stream s = 0);
void cast_f32_f16(const float* src, half* dst, size_t n, Stream s = 0);
void cast_bf16_f16(const void* src, half* dst, size_t n, Stream s = 0);

// Copy row i of src [rows, width] into dst.
void copy_row(const half* src, half* dst, int row, int width, Stream s = 0);

// Argmax over fp16 logits (device -> device int).
void argmax_f16(const half* x, int n, int* out_idx, Stream s = 0);

}}  // namespace helios::glue