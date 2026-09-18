#pragma once
// Helios engine glue (part 2): fp16-weight GEMM/GEMV, layout transforms, MoE permutation.
#include "../cuda/cuda_shim.hpp"

namespace helios { namespace glue {

// y[M,N] = x[M,K] @ w[N,K]^T (fp16 weights, fp32 accumulate, fp16 or fp32 out).
// add: accumulate into y (y must be fp32, zero-init) instead of overwriting.
void gemm_nt_f16(void* y, const half* x, const half* w, int M, int N, int K,
                 bool y_fp32 = false, bool add = false, Stream s = 0);

// Tiled fp16 NN GEMM: C[M,N] = A[M,K] * B[K,N] (B row-major, coalesced in N).
// lda/ldc default to K/N; pass them for strided views (e.g. one head of a [rows,64,512] tensor).
void gemm_nn_f16(void* C, const half* A, const half* B, int M, int N, int K, Stream s = 0,
                 int ldc = 0, bool c_fp32 = false, int lda = 0);

// f16 transpose: src [rows,cols] -> dst [cols,rows]
void transpose_f16_half(void* dst, const half* src, int rows, int cols, Stream s = 0);

// GEMV single row: y[N] = x[K] @ w[N,K]^T
void gemv_nt_f16(void* y, const half* x, const half* w, int N, int K, bool y_fp32 = false, Stream s = 0);

// Transpose + cast: src [M,F] fp32 -> dst [F,M] bf16 (channel-major for conv kernels).
void transpose_f32_bf16(const float* src, void* dst, int M, int F, Stream s = 0);
// Transpose + cast: src [M,F] bf16 -> dst [M,F]? no: dst [F,M]?? kept 1:1 row copy
void cast_f16_bf16(const half* src, void* dst, size_t n, Stream s = 0);
void cast_f32_bf16(const float* src, void* dst, size_t n, Stream s = 0);

// a += b (fp32)
void add_f32_inplace(float* a, const float* b, size_t n, Stream s = 0);

// Sigmoid in place, fp32.
void sigmoid_f32(float* x, size_t n, Stream s = 0);

// Write one row of a cache tensor: dst[row] = src (fp16).
void scatter_row(half* dst, const half* src, const int* row_idx, int width, Stream s = 0);

// MoE permutation for grouped GEMM:
//   ids [M, topk] int32, weights [M, topk] fp32, M rows
//   expert_count [E+1] int64 (cumulative bincount)
//   token_sorted [M*topk] int64, weight_sorted [M*topk] fp16
//   order_by_rank: if true, expert_count is cumulative and token_sorted grouped by expert id.
// workspace must hold 3*(E+2) int64 (counts, offsets, cursors); provided by the caller.
void moe_permute(const int64_t* ids, const half* weights, int M, int topk, int E,
                 int64_t* expert_count, int64_t* token_sorted, half* weight_sorted,
                 int64_t* workspace, Stream s = 0);

// KDA gate/beta preparation (fp32 dt_bias, matching the torch fp32 reference path):
//   mixed_qkv[f*S + s] = bf16(qkv[s*F + f])           (channel-major for the conv kernel)
//   beta[s*H + h]      = bf16(sigmoid(b[s*H + h]))
//   g[s*H*Dk + h*Dk+d] = lower_bound * sigmoid(exp(a_log[h]) * (f[s*H*Dk + h*Dk + d] + dt_bias[h*Dk+d]))
void kda_transpose_cast(void* dst_bf16, const float* src, int S, int F, Stream s = 0);
void kda_gate(const float* b, const float* f, const float* dt_bias, const float* a_log,
              float lower_bound, void* beta_bf16, float* g, int S, int H, int Dk, Stream s = 0);
void kda_prepare(const float* qkv, const float* b, const float* f,
                 const float* dt_bias, const float* a_log, float lower_bound,
                 void* mixed_qkv_bf16, void* beta_bf16, float* g,
                 int S, int F, int H, int Dk, Stream s = 0);

// swiglu with clamp: out = silu(min(gate, limit)) * clamp(up, -limit, limit)
void swiglu_clamp(const float* gate, const float* up, half* out, size_t n, float limit, Stream s = 0);

// layernorm with weight+bias over dim=128 rows (indexer k_norm), fp16 in/out.
void layernorm_f16(const half* x, const half* w, const half* b, half* y, int rows, int dim,
                   float eps, Stream s = 0);

}}  // namespace helios::glue
