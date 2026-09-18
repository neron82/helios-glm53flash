#include <cstdlib>
// GLM-5.3 attention kernels — see attn.cuh for contracts. fp16 IO, fp32 accum.
#include "attn.cuh"
#include <cuda_fp16.h>

namespace helios { namespace attn {


// ---------------------------------------------------------------------------
// q_latent: q_lat[m,h,c] = sum_j q_nope[m,h,j] * kv_b[h*512+j][c]
// grid (m*64), block 256; thread t owns output cols t and t+256.
__global__ void q_latent_k(const half* __restrict__ qn, const half* __restrict__ kb,
                           half* __restrict__ out, int m) {
  int mh = blockIdx.x % 64, mm = blockIdx.x / 64;
  int t = threadIdx.x;
  __shared__ half q_sm[8 * 256];              // [m,256]
  for (int i = t; i < m * 256; i += 256) q_sm[i] = qn[(i / 256) * 64 * 256 + mh * 256 + (i % 256)];
  __syncthreads();
  float acc[8][2] = {};
  const half* base = kb + (size_t)(mh * 512) * 512;
  #pragma unroll 4
  for (int j = 0; j < 256; j++) {
    float fa = __half2float(base[(size_t)j * 512 + t]);
    float fb = __half2float(base[(size_t)j * 512 + t + 256]);
    for (int r = 0; r < m; r++) {
      float qv = __half2float(q_sm[r * 256 + j]);
      acc[r][0] += qv * fa; acc[r][1] += qv * fb;
    }
  }
  for (int r = 0; r < m; r++) {
    out[(size_t)r * 64 * 512 + mh * 512 + t] = __float2half_rn(acc[r][0]);
    out[(size_t)r * 64 * 512 + mh * 512 + t + 256] = __float2half_rn(acc[r][1]);
  }
}

void q_latent(const half* q_nope, const half* kv_b, half* q_lat, int m, Stream s) {
  HELIOS_ASSERT(m >= 1 && m <= 8, "m range 1..8");
  q_latent_k<<<m * 64, 256, 0, s>>>(q_nope, kv_b, q_lat, m);
  cuda_check(cudaPeekAtLastError());
}

// Batched q_lat for all rows in one launch: the m<=8 kernel is coalesced but each launch costs ~1.4ms
// of latency, so a 2048-row layer paid 256 serial launches. One block per (row, head), 512 threads =
// the 512 q_lat channels, with the q_nope row staged in shared memory (it is shared by all threads).
__global__ void q_latent_t_k(const half* __restrict__ qn, const half* __restrict__ kb,
                             half* __restrict__ out, int nrows) {
  int mh = blockIdx.y, mm = blockIdx.x;
  if (mm >= nrows) return;
  __shared__ half q_sm[256];
  if (threadIdx.x < 256) q_sm[threadIdx.x] = qn[(size_t)mm * 64 * 256 + mh * 256 + threadIdx.x];
  __syncthreads();
  int c = threadIdx.x;                      // 0..511
  const half* w = kb + (size_t)(mh * 512) * 512 + c;   // w_uk[h][j][c], contiguous across threads
  float acc = 0.f;
  #pragma unroll 8
  for (int j = 0; j < 256; j++) acc += __half2float(q_sm[j]) * __half2float(w[(size_t)j * 512]);
  out[(size_t)mm * 64 * 512 + mh * 512 + c] = __float2half_rn(acc);
}

void q_latent_t(const half* q_nope, const half* kv_b, half* q_lat, int nrows, Stream s) {
  dim3 grid(nrows, 64), block(512);
  q_latent_t_k<<<grid, block, 0, s>>>(q_nope, kv_b, q_lat, nrows);
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// o_absorb: o[m,h,j] = sum_c lat_out[m,h,c] * kv_b[h*512+256+j][c]
// grid (m*64), block 256; thread t owns output j=t; loops c 512.
__global__ void o_absorb_k(const half* __restrict__ lo, const half* __restrict__ kb,
                           half* __restrict__ out, int m) {
  int mh = blockIdx.x % 64, mm = blockIdx.x / 64; (void)mm;
  int t = threadIdx.x;
  __shared__ half s_sm[8 * 512];              // [m,512]
  for (int i = t; i < m * 512; i += 256) s_sm[i] = lo[(i / 512) * 64 * 512 + mh * 512 + (i % 512)];
  __syncthreads();
  float acc[8] = {};
  const half* base = kb + (size_t)(mh * 512 + 256 + t) * 512;
  #pragma unroll 4
  for (int c = 0; c < 512; c++) {
    float wf = __half2float(base[c]);
    for (int r = 0; r < m; r++) acc[r] += __half2float(s_sm[r * 512 + c]) * wf;
  }
  for (int r = 0; r < m; r++) out[(size_t)r * 64 * 256 + mh * 256 + t] = __float2half_rn(acc[r]);
}

void o_absorb(const half* lat_out, const half* kv_b, half* o, int m, Stream s) {
  HELIOS_ASSERT(m >= 1 && m <= 8, "m range 1..8");
  o_absorb_k<<<m * 64, 256, 0, s>>>(lat_out, kv_b, o, m);
  cuda_check(cudaPeekAtLastError());
}

// Transposed w_uv: kv_b stores w_uv[h] as [256 j][512 c]; o_absorb_k reads it with a 1KB stride
// between the 256 threads of a block (fully uncoalesced - measured 1.44ms per 8-row launch, ~0.09
// TFLOPS). Storing w_uv^T[h][c][j] makes the inner load contiguous across threads.
__global__ void wuv_transpose_k(const half* __restrict__ kv_b, half* __restrict__ wuv_t,
                                size_t total) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int j = (int)(i % 256);
  int c = (int)((i / 256) % 512);
  int h = (int)(i / (256 * 512));
  wuv_t[i] = kv_b[((size_t)(h * 512 + 256 + j)) * 512 + c];
}

void wuv_transpose(const half* kv_b, half* wuv_t, Stream s) {
  const size_t total = (size_t)64 * 512 * 256;
  wuv_transpose_k<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(kv_b, wuv_t, total);
  cuda_check(cudaPeekAtLastError());
}

// o[m,h,j] = sum_c lat_out[m,h,c] * wuv_t[h][c][j]   (one block per (row, head), 256 threads = j)
__global__ void o_absorb_t_k(const half* __restrict__ lo, const half* __restrict__ wuv_t,
                             half* __restrict__ out, int m, int nrows) {
  int mh = blockIdx.y;
  int mm = blockIdx.x;
  if (mm >= nrows) return;
  int j = threadIdx.x;
  const half* w = wuv_t + (size_t)mh * 512 * 256 + j;
  const half* l = lo + (size_t)mm * 64 * 512 + mh * 512;
  float acc = 0.f;
  #pragma unroll 8
  for (int c = 0; c < 512; c++) acc += __half2float(l[c]) * __half2float(w[(size_t)c * 256]);
  out[(size_t)mm * 64 * 256 + mh * 256 + j] = __float2half_rn(acc);
}

void o_absorb_t(const half* lat_out, const half* wuv_t, half* o, int nrows, Stream s) {
  dim3 grid(nrows, 64), block(256);
  o_absorb_t_k<<<grid, block, 0, s>>>(lat_out, wuv_t, o, nrows, nrows);
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// indexer_score: thread per pool; q_idx [m,32,128] in smem.
__global__ void indexer_score_k(const half* __restrict__ qi, const half* __restrict__ pk,
                                const half* __restrict__ w, const int* __restrict__ q_pos,
                                half* __restrict__ scores, int m, int npools, int pk_stride) {
  int mm = blockIdx.y;
  int p = blockIdx.x * 256 + threadIdx.x;
  __shared__ half q_sm[32 * 128], w_sm[32];
  for (int i = threadIdx.x; i < 32 * 128; i += 256) q_sm[i] = qi[mm * 32 * 128 + i];
  if (threadIdx.x < 32) w_sm[threadIdx.x] = w[mm * 32 + threadIdx.x];
  __syncthreads();
  if (p >= npools) return;
  int pos = q_pos[mm];
  if (p * POOL + POOL - 1 > pos) { scores[(size_t)mm * npools + p] = __float2half(-INFINITY); return; }
  float total = 0.f;
  #pragma unroll
  for (int h = 0; h < 32; h++) {
    float dot = 0.f;
    #pragma unroll 8
    for (int d = 0; d < 128; d++)
      dot += __half2float(pk[(size_t)d * pk_stride + p]) * __half2float(q_sm[h * 128 + d]);
    dot *= 0.08838834764f;                        // 1/sqrt(128)
    if (dot > 0.f) total += __half2float(w_sm[h]) * dot;
  }
  total *= 0.17677669530f;                        // 1/sqrt(32)
  scores[(size_t)mm * npools + p] = __float2half_rn(total);
}

// indexer_score: the fast one-query-per-block kernel below. Pool row reads are coalesced across
// threads (the plane is stored transposed) and the query row is broadcast from shared memory.
//
// Two query-tiling variants were tried and rejected, both timed on batch 1 of a 130k-token prefill
// (legacy = 261ms): register-tiled (TQ=4, pool row held in 128 registers/thread) = 1797ms; and
// shared-memory-tiled (TQ=4, TP=128, dynamic smem) = 822ms. Both cut DRAM traffic 4x and both were
// slower, which identifies this kernel as instruction-throughput-bound: ~4096 MACs per
// (query, pool) pair at ~4 instructions each, so it runs at the issue rate, not at memory speed.
// Cutting it further needs tensor cores (tiled dot product writing [m*32, npools] in blocks, then a
// streaming head reduction), not better locality.
// ---------------------------------------------------------------------------
// Tensor-core indexer scoring. The scalar kernel is bound by its own smem+FMA work (measured: a
// bit-exact row-blocked variant that cut pool-plane traffic 8x changed nothing, so the traffic is
// not the limit - ~5e11 scalar ops per layer at 246k context is). This version runs the
// per-head dot products on the fp16 tensor cores:
//   S[16 queries x 8 pools] += q[16 x 16 dims] x k[16 dims x 8 pools]      (m16n8k16, row.col)
// then applies relu + the per-head weight and accumulates over the 32 heads in fp32.
// Block = 16 query rows x IDX_PT pools; warp w owns pools [p0 + 8w, p0 + 8w + 8); the pool keys are
// staged once per block from the pool-major mirror plane (kt[p][d] is exactly the column-major B
// the mma wants). Padding the smem rows to 136 halves removes the 8-way bank conflict on A and B.
#define IDX_PT 64
#define IDX_LD 136
__global__ void indexer_score_mma_k(const half* __restrict__ qi, const half* __restrict__ pk_nt,
                                    const half* __restrict__ w, const int* __restrict__ q_pos,
                                    half* __restrict__ scores, int rows, int npools) {
  int r0 = blockIdx.y * 16;
  int p0 = blockIdx.x * IDX_PT;
  int nr = min(16, rows - r0);
  int np = min(IDX_PT, npools - p0);
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  int g = lane >> 2, tig = lane & 3;
  __shared__ half kt[IDX_PT][IDX_LD];
  __shared__ half qg[4][16][IDX_LD];
  __shared__ float wg[16][4];
  __shared__ int pos_sm[16];
  if (threadIdx.x < nr) pos_sm[threadIdx.x] = q_pos[r0 + threadIdx.x];
  for (int i = threadIdx.x; i < IDX_PT * 128; i += 256) {
    int pp = i >> 7, d = i & 127;
    kt[pp][d] = (pp < np) ? pk_nt[(size_t)(p0 + pp) * 128 + d] : __float2half(0.f);
  }
  __syncthreads();
  float tot0 = 0.f, tot1 = 0.f, tot2 = 0.f, tot3 = 0.f;
  int maxpos = 0;
  for (int r = 0; r < nr; r++) maxpos = max(maxpos, pos_sm[r]);
  const int col0 = wid * 8 + tig * 2;                  // this lane's two pool columns
  const int pcol0 = p0 + col0, pcol1 = pcol0 + 1;
  for (int hg = 0; hg < 8; hg++) {
    for (int i = threadIdx.x; i < 4 * 16 * 128; i += 256) {
      int h = i >> 11, rr = (i >> 7) & 15, d = i & 127;
      qg[h][rr][d] = qi[((size_t)(r0 + rr) * 32 + hg * 4 + h) * 128 + d];
    }
    for (int i = threadIdx.x; i < 64; i += 256) wg[i >> 2][i & 3] = w[(size_t)(r0 + (i >> 2)) * 32 + hg * 4 + (i & 3)];
    __syncthreads();
    #pragma unroll
    for (int h = 0; h < 4; h++) {
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
      #pragma unroll
      for (int dt = 0; dt < 8; dt++) {
        const int o = dt * 16 + tig * 2;
        uint32_t a0 = *(const uint32_t*)(&qg[h][g][o]);
        uint32_t a1 = *(const uint32_t*)(&qg[h][g + 8][o]);
        uint32_t a2 = *(const uint32_t*)(&qg[h][g][o + 8]);
        uint32_t a3 = *(const uint32_t*)(&qg[h][g + 8][o + 8]);
        uint32_t b0 = *(const uint32_t*)(&kt[wid * 8 + g][o]);
        uint32_t b1 = *(const uint32_t*)(&kt[wid * 8 + g][o + 8]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
      const float s0 = 0.08838834764f;                 // 1/sqrt(128)
      float q0 = c0 * s0, q1 = c1 * s0, q2 = c2 * s0, q3 = c3 * s0;
      tot0 += wg[g][h] * (q0 > 0.f ? q0 : 0.f);
      tot1 += wg[g][h] * (q1 > 0.f ? q1 : 0.f);
      tot2 += wg[g + 8][h] * (q2 > 0.f ? q2 : 0.f);
      tot3 += wg[g + 8][h] * (q3 > 0.f ? q3 : 0.f);
    }
    __syncthreads();
  }
  // Per-(row, column) publish: a pool can be visible for some rows of the block and not others, so
  // the visibility test must be per row (a block-level test wrote values where the scalar masks -inf).
  const float os = 0.17677669530f;                     // 1/sqrt(32)
  const int rows_l[2] = {g, g + 8};
  const int cols_l[2] = {pcol0, pcol1};
  const float vals_l[2][2] = {{tot0, tot1}, {tot2, tot3}};
  #pragma unroll
  for (int ri = 0; ri < 2; ri++) {
    if (rows_l[ri] >= nr) continue;
    #pragma unroll
    for (int ci = 0; ci < 2; ci++) {
      int cc = cols_l[ci];
      if (cc >= npools) continue;
      bool vis = cc * POOL + POOL - 1 <= pos_sm[rows_l[ri]];
      scores[(size_t)(r0 + rows_l[ri]) * npools + cc] =
          vis ? __float2half_rn(vals_l[ri][ci] * os) : __float2half(-INFINITY);
    }
  }
}

void indexer_score_mma(const half* q_idx, const half* pool_k_nt, const half* w, const int* q_pos,
                       half* scores, int m, int npools, Stream s) {
  dim3 g((npools + IDX_PT - 1) / IDX_PT, (m + 15) / 16);
  indexer_score_mma_k<<<g, 256, 0, s>>>(q_idx, pool_k_nt, w, q_pos, scores, m, npools);
  cuda_check(cudaPeekAtLastError());
}

void indexer_score(const half* q_idx, const half* pool_k, const half* w, const int* q_pos,
                   half* scores, int m, int npools, int pk_stride, Stream s,
                   const half* pool_k_nt) {
  // A row-blocked scalar variant was written, verified bit-exact, and measured neutral (259.8 vs
  // 264.4ms at pos 0, 1863 vs 1873ms at pos 24k): the scalar kernel is bound by its own smem+FMA
  // work, not by pool-plane traffic (see STATUS.md). The tensor-core path below is the fix; it needs
  // the pool-major mirror plane, so it falls back to the scalar kernel when that is unavailable.
  static const bool mma = getenv("HELIOS_IDX_MMA") == nullptr || atoi(getenv("HELIOS_IDX_MMA")) != 0;
  if (mma && pool_k_nt) {
    indexer_score_mma(q_idx, pool_k_nt, w, q_pos, scores, m, npools, s);
    return;
  }
  indexer_score_legacy(q_idx, pool_k, w, q_pos, scores, m, npools, pk_stride, s);
}

void indexer_score_legacy(const half* q_idx, const half* pool_k, const half* w, const int* q_pos,
                          half* scores, int m, int npools, int pk_stride, Stream s) {
  dim3 g((npools + 255) / 256, m);
  indexer_score_k<<<g, 256, 0, s>>>(q_idx, pool_k, w, q_pos, scores, m, npools, pk_stride);
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// pool_expand: block per row m, 512 threads; out row = 512*4 pool tokens + 4 tail.
__global__ void pool_expand_k(const int* __restrict__ pidx, const int* __restrict__ q_pos,
                              int* __restrict__ raw, int cur_pos) {
  int mm = blockIdx.x, i = threadIdx.x;
  int* row = raw + (size_t)mm * (512 * POOL + POOL);
  int pi = pidx[(size_t)mm * 512 + i];
  #pragma unroll
  for (int j = 0; j < POOL; j++) {
    int tok = pi >= 0 ? pi * POOL + j : -1;
    row[i * POOL + j] = (tok >= 0 && tok <= cur_pos) ? tok : -1;
  }
  if (i < POOL) {
    int tail = cur_pos - (POOL - 1 - i);          // last 4 tokens ascending
    row[512 * POOL + i] = tail >= 0 ? tail : -1;
  }
}

void pool_expand(const int* pool_idx, const int* q_pos, int* raw_idx, int m, int cur_pos, Stream s) {
  (void)q_pos;
  pool_expand_k<<<m, 512, 0, s>>>(pool_idx, q_pos, raw_idx, cur_pos);
  cuda_check(cudaPeekAtLastError());
}

// Per-row causal expansion: each row uses its own position from q_pos.
__global__ void pool_expand_row_k(const int* __restrict__ pidx, const int* __restrict__ q_pos,
                                  int* __restrict__ raw) {
  int mm = blockIdx.x, i = threadIdx.x;
  int cur = q_pos[mm];
  int* row = raw + (size_t)mm * (512 * POOL + POOL);
  int pi = pidx[(size_t)mm * 512 + i];
  #pragma unroll
  for (int j = 0; j < POOL; j++) {
    int tok = pi >= 0 ? pi * POOL + j : -1;
    row[i * POOL + j] = (tok >= 0 && tok <= cur) ? tok : -1;
  }
  if (i < POOL) {
    int tail = cur - (POOL - 1 - i);
    row[512 * POOL + i] = tail >= 0 ? tail : -1;
  }
}

void pool_expand_row(const int* pool_idx, const int* q_pos, int* raw_idx, int m, Stream s) {
  pool_expand_row_k<<<m, 512, 0, s>>>(pool_idx, q_pos, raw_idx);
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// kpool_write: raw plane rows for all new tokens; pool keys for pools completed this call.
// grid.x = ceil((pos_start+n_new)/4); block 128 threads (one per channel).
__global__ void kpool_write_k(const half* __restrict__ ik, const half* __restrict__ ig,
                              half* __restrict__ raw, half* __restrict__ pk, half* __restrict__ pk_nt,
                              const float* __restrict__ ape,
                              int pos_start, int n_new, int pk_stride) {
  int p = blockIdx.x;
  int c = threadIdx.x;                            // 128 channels
  int t0 = p * POOL;
  for (int j = 0; j < POOL; j++) {
    int pos = t0 + j;
    int src = pos - pos_start;
    if (src >= 0 && src < n_new) {
      raw[(size_t)pos * 256 + c] = ik[src * 128 + c];
      raw[(size_t)pos * 256 + 128 + c] = ig[src * 128 + c];
    }
  }
  if (t0 < pos_start || t0 + POOL - 1 >= pos_start + n_new) return;   // not completed
  float g[POOL], k[POOL];
  float mx = -1e30f;
  #pragma unroll
  for (int i = 0; i < POOL; i++) {
    k[i] = __half2float(raw[(size_t)(t0 + i) * 256 + c]);
    g[i] = __half2float(raw[(size_t)(t0 + i) * 256 + 128 + c]) + ape[i * 128 + c];
    mx = fmaxf(mx, g[i]);
  }
  float sum = 0.f;
  #pragma unroll
  for (int i = 0; i < POOL; i++) { g[i] = __expf(g[i] - mx); sum += g[i]; }
  float out = 0.f;
  #pragma unroll
  for (int i = 0; i < POOL; i++) out += (g[i] / sum) * k[i];
  out *= 0.25f;                                   // mean of softmax-weighted keys
  // Transposed layout [c][pool]: indexer_score reads one d per lane across 32 lanes, so a
  // pool-major layout made every load a 32B transaction for 2B of data.
  pk[(size_t)c * pk_stride + p] = __float2half_rn(out);
  if (pk_nt) pk_nt[(size_t)p * 128 + c] = __float2half_rn(out);   // pool-major mirror for the mma path
}

void kpool_write(const half* idx_k, const half* idx_g, half* raw_plane, half* pool_k,
                 const float* ape, int pos_start, int n_new, int pk_stride, Stream s,
                 half* pool_k_nt) {
  int blocks = (pos_start + n_new + POOL - 1) / POOL;
  kpool_write_k<<<blocks, 128, 0, s>>>(idx_k, idx_g, raw_plane, pool_k, pool_k_nt, ape, pos_start,
                                       n_new, pk_stride);
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// mla_sparse_decode: block per (m,h), 8 warps partition token list, flash merge.
__global__ void mla_sparse_dec_k(const half* __restrict__ ql, const half* __restrict__ ckv,
                                 const int* __restrict__ ridx, int kidx,
                                 half* __restrict__ lo, int m) {
  int mh = blockIdx.x % 64, mm = blockIdx.x / 64;
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;   // 8 warps, 256 threads
  __shared__ half q_sm[512];
  __shared__ float w_m[8], w_l[8], w_acc[8][512];
  for (int i = lane; i < 512; i += 32) q_sm[i] = ql[(size_t)mm * 64 * 512 + mh * 512 + i];
  __syncthreads();

  float m_i = -INFINITY, l_i = 0.f, acc[16] = {};
  // all lanes of a warp share one token (each lane owns 16 q/v channels).
  // Software-pipelined: the next token's row is fetched while the current one is reduced, because
  // the loop is latency-bound (K=2 head sharing halved the bytes and changed nothing).
  int ntok = wid < kidx ? ridx[(size_t)mm * kidx + wid] : -1;
  uint4 pa = {}, pb = {};
  if (ntok >= 0) {
    const half* crow = ckv + (size_t)ntok * 512;
    pa = *(const uint4*)(crow + lane * 16);
    pb = *(const uint4*)(crow + lane * 16 + 8);
  }
  for (int base = 0; base < kidx; base += 8) {
    bool valid = ntok >= 0;
    uint4 a = pa, b = pb;
    int nb = base + 8 + wid;
    int ntok2 = nb < kidx ? ridx[(size_t)mm * kidx + nb] : -1;
    pa = uint4{}; pb = uint4{};
    if (ntok2 >= 0) {
      const half* crow = ckv + (size_t)ntok2 * 512;
      pa = *(const uint4*)(crow + lane * 16);
      pb = *(const uint4*)(crow + lane * 16 + 8);
    }
    ntok = ntok2;
    half2 va[4] = {*(half2*)&a.x, *(half2*)&a.y, *(half2*)&a.z, *(half2*)&a.w};
    half2 vb[4] = {*(half2*)&b.x, *(half2*)&b.y, *(half2*)&b.z, *(half2*)&b.w};
    float dot = 0.f;
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      half2 qa1 = *(const half2*)(q_sm + lane * 16 + u * 2);
      half2 qa2 = *(const half2*)(q_sm + lane * 16 + 8 + u * 2);
      float2 d1 = __half22float2(__hmul2(qa1, va[u]));
      float2 d2 = __half22float2(__hmul2(qa2, vb[u]));
      dot += d1.x + d1.y + d2.x + d2.y;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
    if (!valid) continue;                       // shuffle done, warp converged here
    dot *= SCALE;
    if (dot > m_i) {
      float sc = __expf(m_i - dot);
      #pragma unroll
      for (int u = 0; u < 16; u++) acc[u] *= sc;
      l_i *= sc; m_i = dot;
    }
    float p = __expf(dot - m_i);
    l_i += p;
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      float2 f1 = __half22float2(va[u]), f2 = __half22float2(vb[u]);
      acc[u * 2] += p * f1.x; acc[u * 2 + 1] += p * f1.y;
      acc[8 + u * 2] += p * f2.x; acc[8 + u * 2 + 1] += p * f2.y;
    }
  }
  // publish warp state, merge across warps: smem channels [8][512]
  if (lane == 0) { w_m[wid] = m_i; w_l[wid] = l_i; }
  #pragma unroll
  for (int u = 0; u < 16; u++) w_acc[wid][lane * 16 + u] = acc[u];
  __syncthreads();
  if (threadIdx.x < 256) {
    int ch = threadIdx.x;                          // channels ch and ch+256
    float gm = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 8; w++) gm = fmaxf(gm, w_m[w]);
    float gl = 0.f, o0 = 0.f, o1 = 0.f;
    if (gm == -INFINITY) gl = 1.f;
    else {
      #pragma unroll
      for (int w = 0; w < 8; w++) {
        float sc = w_m[w] == -INFINITY ? 0.f : __expf(w_m[w] - gm);
        gl += sc * w_l[w];
        o0 += sc * w_acc[w][ch];
        o1 += sc * w_acc[w][ch + 256];
      }
    }
    float inv = gl > 0.f ? 1.f / gl : 0.f;
    lo[(size_t)mm * 64 * 512 + mh * 512 + ch] = __float2half_rn(o0 * inv);
    lo[(size_t)mm * 64 * 512 + mh * 512 + ch + 256] = __float2half_rn(o1 * inv);
  }
}

// K=2 heads per warp: the key row (1KB) is loaded once and feeds two heads' dots and v-accumulators.
// The K=1 kernel above is L2-bound (2.6 TB/s measured) because each of the 64 heads re-reads the same
// 2052 rows; sharing the load across a head pair halves that traffic.
__global__ void mla_sparse_dec2_k(const half* __restrict__ ql, const half* __restrict__ ckv,
                                  const int* __restrict__ ridx, int kidx,
                                  half* __restrict__ lo, int m) {
  int mh = (blockIdx.x % 32) * 2, mm = blockIdx.x / 32;
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;   // 8 warps, 256 threads
  __shared__ half q_sm[2][512];
  // Per-warp partials are staged as fp16: with fp32 the K=2 kernel needs 34KB of smem against the
  // 48KB static limit, i.e. one block per SM instead of two, which is why its 23-30% standalone
  // advantage vanished in-engine. The merged result is written as fp16 anyway.
  __shared__ float w_m[8][2], w_l[8][2];
  __shared__ half w_acc[8][2][512];
  for (int i = lane; i < 512; i += 32) {
    q_sm[0][i] = ql[((size_t)mm * 64 + mh) * 512 + i];
    q_sm[1][i] = ql[((size_t)mm * 64 + mh + 1) * 512 + i];
  }
  __syncthreads();

  float m_i[2] = {-INFINITY, -INFINITY}, l_i[2] = {0.f, 0.f}, acc0[16] = {}, acc1[16] = {};
  for (int base = 0; base < kidx; base += 8) {
    int i = base + wid;
    int tok = i < kidx ? ridx[(size_t)mm * kidx + i] : -1;
    bool valid = tok >= 0;
    uint4 a = {}, b = {};
    if (valid) {
      const half* crow = ckv + (size_t)tok * 512;
      a = *(const uint4*)(crow + lane * 16);
      b = *(const uint4*)(crow + lane * 16 + 8);
    }
    half2 va[4] = {*(half2*)&a.x, *(half2*)&a.y, *(half2*)&a.z, *(half2*)&a.w};
    half2 vb[4] = {*(half2*)&b.x, *(half2*)&b.y, *(half2*)&b.z, *(half2*)&b.w};
    // Accumulate the 16 products per lane in half2 with __hfma2 (4 ops per u) instead of converting
    // every product to float (hmul2 + half22float2 + 2 adds = ~6 ops per u). The partial sums are
    // O(1) over 16 terms before the fp32 cross-lane reduction, so fp16 accumulation here costs ~2e-3
    // relative - checked against the CPU reference by the parity test.
    half2 s0a = __float2half2_rn(0.f), s0b = __float2half2_rn(0.f);
    half2 s1a = __float2half2_rn(0.f), s1b = __float2half2_rn(0.f);
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      half2 qa = *(const half2*)(q_sm[0] + lane * 16 + u * 2);
      half2 qb = *(const half2*)(q_sm[0] + lane * 16 + 8 + u * 2);
      half2 ra = *(const half2*)(q_sm[1] + lane * 16 + u * 2);
      half2 rb = *(const half2*)(q_sm[1] + lane * 16 + 8 + u * 2);
      s0a = __hfma2(qa, va[u], s0a);
      s0b = __hfma2(qb, vb[u], s0b);
      s1a = __hfma2(ra, va[u], s1a);
      s1b = __hfma2(rb, vb[u], s1b);
    }
    float2 f0 = __half22float2(s0a), f1 = __half22float2(s0b);
    float2 g0 = __half22float2(s1a), g1 = __half22float2(s1b);
    float d0 = f0.x + f0.y + f1.x + f1.y;
    float d1 = g0.x + g0.y + g1.x + g1.y;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      d0 += __shfl_xor_sync(0xffffffffu, d0, o);
      d1 += __shfl_xor_sync(0xffffffffu, d1, o);
    }
    if (!valid) continue;
    d0 *= SCALE; d1 *= SCALE;
    if (d0 > m_i[0]) {
      float sc = __expf(m_i[0] - d0);
      #pragma unroll
      for (int u = 0; u < 16; u++) acc0[u] *= sc;
      l_i[0] *= sc; m_i[0] = d0;
    }
    if (d1 > m_i[1]) {
      float sc = __expf(m_i[1] - d1);
      #pragma unroll
      for (int u = 0; u < 16; u++) acc1[u] *= sc;
      l_i[1] *= sc; m_i[1] = d1;
    }
    float p0 = __expf(d0 - m_i[0]), p1 = __expf(d1 - m_i[1]);
    l_i[0] += p0; l_i[1] += p1;
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      float2 f1 = __half22float2(va[u]), f2 = __half22float2(vb[u]);
      acc0[u * 2] += p0 * f1.x; acc0[u * 2 + 1] += p0 * f1.y;
      acc0[8 + u * 2] += p0 * f2.x; acc0[8 + u * 2 + 1] += p0 * f2.y;
      acc1[u * 2] += p1 * f1.x; acc1[u * 2 + 1] += p1 * f1.y;
      acc1[8 + u * 2] += p1 * f2.x; acc1[8 + u * 2 + 1] += p1 * f2.y;
    }
  }
  if (lane == 0) {
    w_m[wid][0] = m_i[0]; w_l[wid][0] = l_i[0];
    w_m[wid][1] = m_i[1]; w_l[wid][1] = l_i[1];
  }
  #pragma unroll
  for (int u = 0; u < 16; u++) {
    w_acc[wid][0][lane * 16 + u] = __float2half_rn(acc0[u]);
    w_acc[wid][1][lane * 16 + u] = __float2half_rn(acc1[u]);
  }
  __syncthreads();
  #pragma unroll
  for (int h = 0; h < 2; h++) {
    int ch = threadIdx.x;                          // 256 threads -> 2 channels each = 512
    float gm = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 8; w++) gm = fmaxf(gm, w_m[w][h]);
    float gl = 0.f, o0 = 0.f, o1 = 0.f;
    if (gm == -INFINITY) gl = 1.f;
    else {
      int l0 = ch / 16, u0 = ch % 16, l1 = (ch + 256) / 16, u1 = (ch + 256) % 16;
      #pragma unroll
      for (int w = 0; w < 8; w++) {
        float sc = w_m[w][h] == -INFINITY ? 0.f : __expf(w_m[w][h] - gm);
        gl += sc * w_l[w][h];
        o0 += sc * __half2float(w_acc[w][h][l0 * 16 + u0]);
        o1 += sc * __half2float(w_acc[w][h][l1 * 16 + u1]);
      }
    }
    float inv = gl > 0.f ? 1.f / gl : 0.f;
    half* dst = lo + ((size_t)mm * 64 + mh + h) * 512;
    dst[ch] = __float2half_rn(o0 * inv);
    dst[ch + 256] = __float2half_rn(o1 * inv);
  }
}

void mla_sparse_decode(const half* q_lat, const half* ckv, const int* raw_idx, int kidx,
                       half* lat_out, int m, Stream s) {
  // Default 2: head-pair sharing halves the key-row bytes, and with the per-warp partials staged as
  // fp16 the kernel needs 18KB of smem instead of 34KB, so it still gets 2 blocks per SM. With fp32
  // staging (34KB against the 48KB static limit) it measured exactly neutral in-engine; as fp16 it is
  // 1.52x faster on the sparse stage. HELIOS_SPARSE_K=1 selects the older single-head kernel.
  int K = 2;
  if (const char* e = getenv("HELIOS_SPARSE_K")) K = atoi(e);
  if (K == 2 && (m % 1) == 0) {   // heads come in 64 = 32 pairs, no divisibility constraint on m
    mla_sparse_dec2_k<<<m * 32, 256, 0, s>>>(q_lat, ckv, raw_idx, kidx, lat_out, m);
  } else {
    mla_sparse_dec_k<<<m * 64, 256, 0, s>>>(q_lat, ckv, raw_idx, kidx, lat_out, m);
  }
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// mla_dense_prefill: block per (q_row, head), sequential causal flash over ckv[0..pos].
__global__ void mla_dense_pre_k(const half* __restrict__ ql, const half* __restrict__ ckv,
                                half* __restrict__ lo, int pos_start) {
  int q = blockIdx.x, mh = blockIdx.y;
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  __shared__ half q_sm[512];
  __shared__ float w_m[8], w_l[8], w_acc[8][512];
  for (int i = lane; i < 512; i += 32) q_sm[i] = ql[(size_t)q * 64 * 512 + mh * 512 + i];
  __syncthreads();
  int tend = pos_start + q;                       // inclusive causal limit
  float m_i = -INFINITY, l_i = 0.f, acc[16] = {};
  // all lanes of a warp share one token (each lane owns 16 q/v channels)
  for (int base = 0; base <= tend; base += 8) {
    int tok = base + wid;
    bool valid = tok <= tend;
    const half* crow = ckv + (size_t)(valid ? tok : 0) * 512;
    uint4 a = *(const uint4*)(crow + lane * 16);
    uint4 b = *(const uint4*)(crow + lane * 16 + 8);
    half2 va[4] = {*(half2*)&a.x, *(half2*)&a.y, *(half2*)&a.z, *(half2*)&a.w};
    half2 vb[4] = {*(half2*)&b.x, *(half2*)&b.y, *(half2*)&b.z, *(half2*)&b.w};
    float dot = 0.f;
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      half2 qa1 = *(const half2*)(q_sm + lane * 16 + u * 2);
      half2 qa2 = *(const half2*)(q_sm + lane * 16 + 8 + u * 2);
      float2 d1 = __half22float2(__hmul2(qa1, va[u]));
      float2 d2 = __half22float2(__hmul2(qa2, vb[u]));
      dot += d1.x + d1.y + d2.x + d2.y;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
    if (!valid) continue;                       // shuffle done, warp converged here
    dot *= SCALE;
    if (dot > m_i) {
      float sc = __expf(m_i - dot);
      #pragma unroll
      for (int u = 0; u < 16; u++) acc[u] *= sc;
      l_i *= sc; m_i = dot;
    }
    float p = __expf(dot - m_i);
    l_i += p;
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      float2 f1 = __half22float2(va[u]), f2 = __half22float2(vb[u]);
      acc[u * 2] += p * f1.x; acc[u * 2 + 1] += p * f1.y;
      acc[8 + u * 2] += p * f2.x; acc[8 + u * 2 + 1] += p * f2.y;
    }
  }
  if (lane == 0) { w_m[wid] = m_i; w_l[wid] = l_i; }
  for (int u = 0; u < 16; u++) w_acc[wid][lane * 16 + u] = acc[u];
  __syncthreads();
  if (threadIdx.x < 256) {
    int ch = threadIdx.x;                          // 256 threads -> 2 channels each = 512
    float gm = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 8; w++) gm = fmaxf(gm, w_m[w]);
    float gl = 0.f, o0 = 0.f, o1 = 0.f;
    if (gm == -INFINITY) gl = 1.f;
    else {
      int l0 = ch / 16, u0 = ch % 16, l1 = (ch + 256) / 16, u1 = (ch + 256) % 16;
      #pragma unroll
      for (int w = 0; w < 8; w++) {
        float sc = w_m[w] == -INFINITY ? 0.f : __expf(w_m[w] - gm);
        gl += sc * w_l[w];
        o0 += sc * w_acc[w][l0 * 16 + u0];
        o1 += sc * w_acc[w][l1 * 16 + u1];
      }
    }
    float inv = gl > 0.f ? 1.f / gl : 0.f;
    lo[(size_t)q * 64 * 512 + mh * 512 + ch] = __float2half_rn(o0 * inv);
    lo[(size_t)q * 64 * 512 + mh * 512 + ch + 256] = __float2half_rn(o1 * inv);
  }
}

void mla_dense_prefill(const half* q_lat, const half* ckv, half* lat_out,
                       int qlen, int pos_start, Stream s) {
  dim3 g(qlen, 64);
  mla_dense_pre_k<<<g, 256, 0, s>>>(q_lat, ckv, lat_out, pos_start);
  cuda_check(cudaPeekAtLastError());
}

}} // namespace helios::attn