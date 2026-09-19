#pragma once
// GLM-5.3 attention kernels: indexer (lightning top-k + kpool), MLA decode/prefill,
// absorbed q_lat/o_lat GEMVs. All fp16 caches, fp32 accumulation, NoPE (no rotary).
// Raw-pointer launchers, stream last. namespace helios::attn.

#include "helios_shim.cuh"

namespace helios { namespace attn {

// ---- constants from config ----
constexpr int N_HEADS   = 64;    // MLA heads
constexpr int QK_NOPE   = 256;   // q_nope / w_uk out dim
constexpr int KV_LORA   = 512;   // ckv latent dim (= q_lat dim)
constexpr int V_DIM     = 256;   // per-head o dim
constexpr int IDX_HEADS = 32;    // indexer heads
constexpr int IDX_DIM   = 128;   // indexer head dim
constexpr int POOL      = 4;     // kpool compression ratio
constexpr float SCALE   = 0.0625f;   // 1/sqrt(256) for MLA scores
constexpr float IDX_SCALE = 0.08838834764f * 0.17677669530f; // 1/sqrt(128) * 1/sqrt(32)

// ---- absorbed projections ----
// q_lat[m,h,c] = sum_j q_nope[m,h,j] * kv_b[h*512+j][c], c in [0,512)
// kv_b is the raw [32768,512] fp16 tensor: rows h*512..h*512+255 = w_uk[h], +256..+511 = w_uv[h].
void q_latent(
    const half* q_nope,   // [m, 64, 256]
    const half* kv_b,     // [32768, 512]
    half* q_lat,          // out [m, 64, 512]
    int m, Stream s = 0);

// o[m,h,j] = sum_c lat_out[m,h,c] * kv_b[h*512+256+j][c]
void wuv_transpose(const half* kv_b, half* wuv_t, Stream s = 0);
void o_absorb_t(const half* lat_out, const half* wuv_t, half* o, int nrows, Stream s = 0);
void q_latent_t(const half* q_nope, const half* kv_b, half* q_lat, int nrows,
                 Stream s = 0);
void o_absorb(
    const half* lat_out,  // [m, 64, 512]
    const half* kv_b,     // [32768, 512]
    half* o,              // out [m, 64, 256]
    int m, Stream s = 0);

// ---- indexer ----
// scores[m,p] = sum_h w[m,h] * relu(dot(q_idx[m,h,:], pool_k[p,:]) * 128^-0.5) * 32^-0.5
// Visible pools: p*POOL+3 <= q_pos (q_pos passed per row via q_pos array, device).
void indexer_score_mma(const half* q_idx, const half* pool_k_nt, const half* w, const int* q_pos,
                       half* scores, int m, int npools, Stream s = 0);
void indexer_score_legacy(
    const half* q_idx, const half* pool_k, const half* w, const int* q_pos, half* scores,
    int m, int npools, int pk_stride, Stream s = 0);
void indexer_score(
    const half* q_idx,    // [m, 32, 128]
    const half* pool_k,   // [128][pk_stride] transposed pool keys
    const half* w,        // [m, 32] weights_proj outputs
    const int* q_pos,     // [m] device int32 absolute positions
    half* scores,         // out [m, npools] (invisible entries = -inf)
    int m, int npools, int pk_stride, Stream s = 0,
    const half* pool_k_nt = nullptr);   // [npools][128] pool-major mirror: enables the mma path

// Expand top pool indices to raw token indices: pool*4 + 0..3, clipped to [0, cur_pos];
// plus tail tokens (cur_pos-3 .. cur_pos) forced in (dedup handled by attention masking
// via -1 markers). out row length kcnt*POOL + POOL, -1 padded.
void pool_expand(
    const int* pool_idx,  // [m, 512] from dsa_topk (-1 padded)
    const int* q_pos,     // [m]
    int* raw_idx,         // out [m, 512*4 + 4]
    int m, int cur_pos,   // last valid token index (inclusive)
    Stream s = 0);

// Per-row causal variant for prefill chunks: row r clips to its own q_pos[r].
void pool_expand_row(
    const int* pool_idx,  // [m, 512]
    const int* q_pos,     // [m] device int32
    int* raw_idx,         // out [m, 512*4 + 4]
    int m, Stream s = 0);

// Write raw idx plane [k||gate] 256/token and complete pool keys:
//   pool_k[p,c] = 0.25 * sum_i softmax_c(gate[p*4+i] + APE[i]) * k[p*4+i]
// softmax over the 4 members channelwise. Called after tokens are appended; n_new tokens
// starting at pos_start. A pool is written by whichever call writes its LAST member, using the raw
// rows of its earlier members, which are still in the plane (same sequence, position-addressed).
void kpool_write(
    const half* idx_k,    // [n_new, 128] conv'd indexer keys
    const half* idx_g,    // [n_new, 128] indexer gates
    half* raw_plane,      // [Tmax, 256] k||gate interleaved rows
    half* pool_k,         // [128][pk_stride] transposed pool keys
    const float* ape,     // [4, 128] fp32
    int pos_start, int n_new, int pk_stride, Stream s = 0,
    half* pool_k_nt = nullptr);   // optional [pk_stride][128] pool-major mirror (mma B operand)

// ---- MLA sparse decode ----
// Flash-style gather over selected tokens. Each (m,h) row: online softmax fp32 over
// scores q_lat·ckv[t]*SCALE, weighted sum of ckv[t] rows -> lat_out [m,64,512].
// raw_idx rows may contain -1 (skip). kcnt up to 2052.
void mla_sparse_decode(
    const half* q_lat,    // [m, 64, 512]
    const half* ckv,      // [Tmax, 512] latent KV cache
    const int* raw_idx,   // [m, kidx] int32 (-1 = skip)
    int kidx,
    half* lat_out,        // out [m, 64, 512]
    int m, Stream s = 0);

// ---- MLA dense prefill (correctness-first baseline; chunked flash later) ----
// For each query row q in [q0, q0+qlen) attend to ckv[0..q_pos] (causal), causal mask,
// softmax fp32, out lat rows. heads share keys/values (MLA). q_lat [qlen,64,512].
void mla_dense_prefill(
    const half* q_lat,    // [qlen, 64, 512]
    const half* ckv,      // [Tmax, 512]
    half* lat_out,        // out [qlen, 64, 512]
    int qlen, int pos_start, Stream s = 0);

}} // namespace helios::attn