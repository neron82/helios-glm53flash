#include "engine/glue2.cuh"
#include <cstdlib>
#include <cuda_fp16.h>

namespace helios { namespace glue {

// ---------------------------------------------------------------- fp16 GEMM (w row-major [N,K])
// Block computes a 16(M) x 16(N) output tile, smem-tiled over K in 16-wide chunks.
__global__ void gemm_nt_f16_k(const half* __restrict__ x, const half* __restrict__ w,
                              void* __restrict__ y, int M, int N, int K, bool y_fp32, bool add) {
  __shared__ half xs[16][17], ws[16][17];
  int tm = blockIdx.y * 16, tn = blockIdx.x * 16;
  int tx = threadIdx.x, ty = threadIdx.y;        // 16x16 threads
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += 16) {
    if (tm + ty < M && k0 + tx < K) xs[ty][tx] = x[(size_t)(tm + ty) * K + k0 + tx];
    else xs[ty][tx] = __float2half(0.f);
    // ws[n_local][k_local] = w[tn + n_local][k0 + k_local]
    if (tn + ty < N && k0 + tx < K) ws[ty][tx] = w[(size_t)(tn + ty) * K + k0 + tx];
    else ws[ty][tx] = __float2half(0.f);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 16; k++) acc += __half2float(xs[ty][k]) * __half2float(ws[tx][k]);
    __syncthreads();
  }
  int m = tm + ty, n = tn + tx;
  if (m >= M || n >= N) return;
  if (y_fp32) {
    float* yf = (float*)y;
    if (add) atomicAdd(&yf[(size_t)m * N + n], acc);
    else yf[(size_t)m * N + n] = acc;
  } else {
    ((half*)y)[(size_t)m * N + n] = __float2half_rn(acc);
  }
}

// ---------------------------------------------------------------------------
// Single-row GEMV: y[n] = dot(x[0..K), w[n][0..K)). One warp per output with the lanes splitting
// K, so the weight row is read coalesced (32 lanes x half2 = 64 halves per transaction group).
// The tiled kernel above is 100x slower at M=1: it computes a 16x16 output tile of which 15 rows
// are discarded, with a scalar half->float conversion per MAC. At 1 token per step the engine
// calls this ~8 times per layer, and it measured as 18.2 of the KDA layer's 23.1ms per token.
__global__ void gemv_nt_f16_k(const half* __restrict__ x, const half* __restrict__ w,
                              void* __restrict__ y, int N, int K, bool y_fp32) {
  extern __shared__ half xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (wid >= N) return;
  const half* wr = w + (size_t)wid * K;
  float acc = 0.f;
  for (int k = lane * 2; k + 1 < K; k += 64) {
    float2 xf = __half22float2(*(const half2*)(xs + k));
    float2 wf = __half22float2(*(const half2*)(wr + k));
    acc += xf.x * wf.x + xf.y * wf.y;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    if (y_fp32) ((float*)y)[wid] = acc;
    else ((half*)y)[wid] = __float2half_rn(acc);
  }
}


// ---------------------------------------------------------------- tiled f16 NN GEMM
// C[M,N] = A[M,K] * B[K,N], all fp16 in / fp16 out with fp32 accumulation. 64x64 output tile per
// block, 256 threads, 4x4 outputs per thread, K staged in 16-wide smem tiles with half2 loads.
// The scalar kernels this replaces (o_absorb_t, the KDA low-rank projections, the indexer
// projections) each run at 0.6-1.5 TFLOPS because they issue one convert+FMA per MAC; measured
// 8192-row prefill shapes make them ~4s of a 28s batch. B must be [K,N] row-major (coalesced in N):
// callers hold [N,K] weights, so the engine builds transposed copies once at init.
#define GNN_TM 64
#define GNN_TN 64
#define GNN_TK 16
__global__ void gemm_nn_f16_k(const half* __restrict__ A, const half* __restrict__ B,
                              void* __restrict__ Cv, int M, int N, int K, int lda, int ldc,
                              bool c_fp32) {
  half* C = nullptr; float* Cf = nullptr;
  if (c_fp32) Cf = (float*)Cv; else C = (half*)Cv;
  __shared__ half as[GNN_TM][GNN_TK + 8];
  __shared__ half bs[GNN_TK][GNN_TN + 8];
  int m0 = blockIdx.y * GNN_TM, n0 = blockIdx.x * GNN_TN;
  // 4 rows x 4 cols per thread (256 threads). An 8x4 variant benched 4-5x faster but was
  // non-deterministic (the same kernel reported all-wrong, 0.8%-wrong and 99%-wrong across runs),
  // and it bought nothing in-engine, so the 4x4 form stays until the race is found.
  int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;      // 16x16 threads
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += GNN_TK) {
    // stage A: 64 rows x 16 k   (2624 halves / 256 threads)
    for (int i = threadIdx.x; i < GNN_TM * GNN_TK; i += 256) {
      int r = i >> 4, c = i & 15;
      int gm = m0 + r, gk = k0 + c;
      as[r][c] = (gm < M && gk < K) ? A[(size_t)gm * lda + gk] : __float2half(0.f);
    }
    // stage B: 16 k x 64 n
    for (int i = threadIdx.x; i < GNN_TK * GNN_TN; i += 256) {
      int kk = i >> 6, n = i & 63;
      int gk = k0 + kk, gn = n0 + n;
      bs[kk][n] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : __float2half(0.f);
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < GNN_TK; kk++) {
      // A fragment: 4 ROWS at column kk (an earlier revision walked along k here and paired the wrong
      // rows with the weights - caught by the greedy-output comparison).
      const half* brow = &bs[kk][tx * 4];
      float a[4], b[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) a[i] = __half2float(as[ty * 4 + i][kk]);
      #pragma unroll
      for (int j = 0; j < 4; j++) b[j] = __half2float(brow[j]);
      #pragma unroll
      for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int gm = m0 + ty * 4 + i;
    if (gm >= M) continue;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int gn = n0 + tx * 4 + j;
      if (gn >= N) continue;
      if (c_fp32) Cf[(size_t)gm * ldc + gn] = acc[i][j];
      else C[(size_t)gm * ldc + gn] = __float2half_rn(acc[i][j]);
    }
  }
}

// src [rows][cols] f16 -> dst [cols][rows] f16. One thread per element: the smem-tiled version I
// first wrote had four separate indexing bugs (inverted fragment pairing, swapped load bounds,
// swapped grid axes) and the bench self-test below caught each one. This runs once at init on ~230MB,
// so the strided side costs a couple of seconds and correctness-by-construction is worth more.
__global__ void transpose_f16_k(const half* __restrict__ src, half* __restrict__ dst,
                                long long n, int rows, int cols) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int r = (int)(i / cols), c = (int)(i % cols);
  dst[(size_t)c * rows + r] = src[i];
}

void transpose_f16_half(void* dst, const half* src, int rows, int cols, Stream s) {
  long long n = (long long)rows * cols;
  if (n <= 0) return;
  long long blocks = (n + 255) / 256;
  transpose_f16_k<<<(unsigned)blocks, 256, 0, s>>>(src, (half*)dst, n, rows, cols);
  cuda_check(cudaPeekAtLastError());
}

void gemm_nn_f16(void* C, const half* A, const half* B, int M, int N, int K, Stream s, int ldc,
                 bool c_fp32, int lda) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (ldc <= 0) ldc = N;
  if (lda <= 0) lda = K;
  dim3 grid((N + GNN_TN - 1) / GNN_TN, (M + GNN_TM - 1) / GNN_TM, 1);
  gemm_nn_f16_k<<<grid, 256, 0, s>>>(A, B, C, M, N, K, lda, ldc, c_fp32);
  cuda_check(cudaPeekAtLastError());
}

void gemv_nt_f16(void* y, const half* x, const half* w, int N, int K, bool y_fp32, Stream s) {
  int block = 256;
  int grid = (N + block / 32 - 1) / (block / 32);
  gemv_nt_f16_k<<<grid, block, (size_t)K * 2, s>>>(x, w, y, N, K, y_fp32);
  cuda_check(cudaPeekAtLastError());
}

void gemm_nt_f16(void* y, const half* x, const half* w, int M, int N, int K,
                 bool y_fp32, bool add, Stream s) {
  // Route the decode case (one token) to the GEMV above; K must be even for the half2 path.
  static const bool rc = getenv("HELIOS_GEMM_RC") == nullptr || atoi(getenv("HELIOS_GEMM_RC")) != 0;
  if (rc && M == 1 && !add && (K % 2) == 0) { gemv_nt_f16(y, x, w, N, K, y_fp32, s); return; }
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  gemm_nt_f16_k<<<grid, block, 0, s>>>(x, w, y, M, N, K, y_fp32, add);
  cuda_check(cudaPeekAtLastError());
}

__global__ void transpose_f32_bf16_k(const float* __restrict__ src, unsigned short* __restrict__ dst,
                                     int M, int F) {
  int m = blockIdx.y, f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F || m >= M) return;
  dst[(size_t)f * M + m] = __bfloat16_as_ushort(__float2bfloat16(src[(size_t)m * F + f]));
}

void transpose_f32_bf16(const float* src, void* dst, int M, int F, Stream s) {
  dim3 g((F + 255) / 256, M);
  transpose_f32_bf16_k<<<g, 256, 0, s>>>(src, (unsigned short*)dst, M, F);
  cuda_check(cudaPeekAtLastError());
}

__global__ void cast_f16_bf16_k(const half* __restrict__ s, unsigned short* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) d[i] = __bfloat16_as_ushort(__float2bfloat16(__half2float(s[i])));
}
__global__ void cast_f32_bf16_k(const float* __restrict__ s, unsigned short* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) d[i] = __bfloat16_as_ushort(__float2bfloat16(s[i]));
}
void cast_f16_bf16(const half* src, void* dst, size_t n, Stream s) {
  cast_f16_bf16_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, (unsigned short*)dst, n);
  cuda_check(cudaPeekAtLastError());
}
void cast_f32_bf16(const float* src, void* dst, size_t n, Stream s) {
  cast_f32_bf16_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, (unsigned short*)dst, n);
  cuda_check(cudaPeekAtLastError());
}

__global__ void add_f32_inplace_k(float* __restrict__ a, const float* __restrict__ b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

__global__ void sigmoid_f32_k(float* __restrict__ x, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) x[i] = 1.0f / (1.0f + __expf(-x[i]));
}
void sigmoid_f32(float* x, size_t n, Stream s) {
  sigmoid_f32_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, n);
  cuda_check(cudaPeekAtLastError());
}

void add_f32_inplace(float* a, const float* b, size_t n, Stream s) {
  add_f32_inplace_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(a, b, n);
  cuda_check(cudaPeekAtLastError());
}

__global__ void scatter_row_k(half* __restrict__ dst, const half* __restrict__ src,
                              const int* __restrict__ row_idx, int width) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < width) dst[(size_t)(*row_idx) * width + c] = src[c];
}
void scatter_row(half* dst, const half* src, const int* row_idx, int width, Stream s) {
  scatter_row_k<<<(width + 255) / 256, 256, 0, s>>>(dst, src, row_idx, width);
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------- MoE permutation
__global__ void moe_count_k(const int64_t* __restrict__ ids, int total, int E,
                            int64_t* __restrict__ count) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  long long c = 0;
  for (int i = 0; i < total; i++) if (ids[i] == e) c++;
  count[e] = c;
}

__global__ void moe_offset_k(const int64_t* __restrict__ count, int E,
                             int64_t* __restrict__ offset, int64_t* __restrict__ expert_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int64_t acc = 0;
  for (int e = 0; e < E; e++) {
    offset[e] = acc;
    acc += count[e];
    expert_count[e] = count[e];
  }
  expert_count[E] = 0;   // ignored by the kernel
  offset[E] = acc;       // total, used as the atomic cursor base
}

__global__ void moe_scatter_k(const int64_t* __restrict__ ids, const half* __restrict__ weights,
                              int total, int topk, const int64_t* __restrict__ offset,
                              int64_t* __restrict__ cursor,
                              int64_t* __restrict__ token_sorted, half* __restrict__ weight_sorted) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int e = ids[i];
  int64_t pos = atomicAdd((unsigned long long*)&cursor[e], 1ull);
  token_sorted[pos] = i / topk;
  weight_sorted[pos] = weights[i];
}

void moe_permute(const int64_t* ids, const half* weights, int M, int topk, int E,
                 int64_t* expert_count, int64_t* token_sorted, half* weight_sorted,
                 int64_t* workspace, Stream s) {
  int64_t* cnt = workspace;
  int64_t* off = workspace + E + 2;
  int64_t* cur = workspace + 2 * (E + 2);
  int total = M * topk;
  moe_count_k<<<(E + 127) / 128, 128, 0, s>>>(ids, total, E, cnt);
  moe_offset_k<<<1, 1, 0, s>>>(cnt, E, off, expert_count);
  cuda_check(cudaMemcpyAsync(cur, off, (E + 1) * sizeof(int64_t), cudaMemcpyDeviceToDevice, s));
  moe_scatter_k<<<(total + 255) / 256, 256, 0, s>>>(ids, weights, total, topk, off, cur,
                                                    token_sorted, weight_sorted);
  cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------- KDA prepare
// qkv transpose+cast [S,F] fp32 -> [F,S] bf16, tiled through smem so both sides are coalesced.
// The elementwise form in kda_prepare_k writes with stride S (8192 floats = 32KB), i.e. one sector per
// element - it was 28ms per layer, 0.95s of a 8192-token prefill batch.
__global__ void kda_transpose_cast_k(const float* __restrict__ src, unsigned short* __restrict__ dst,
                                     int S, int F) {
  __shared__ float t[32][33];
  int s0 = blockIdx.y * 32, f0 = blockIdx.x * 32;
  int fs = f0 + threadIdx.x;                       // source column (fast axis of src)
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    int ss = s0 + threadIdx.y + j;
    t[threadIdx.y + j][threadIdx.x] = (ss < S && fs < F) ? src[(size_t)ss * F + fs] : 0.f;
  }
  __syncthreads();
  // dst[f][s]: consecutive threads (tx) must write consecutive s, so s = s0+tx and the smem entry
  // needed is t[tx][ty+j] (s relative = tx, f relative = ty+j).
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    int df = f0 + threadIdx.y + j;
    int ds = s0 + threadIdx.x;
    if (df < F && ds < S) dst[(size_t)df * S + ds] = __bfloat16_as_ushort(__float2bfloat16(t[threadIdx.x][threadIdx.y + j]));
  }
}

void kda_transpose_cast(void* dst, const float* src, int S, int F, Stream s) {
  dim3 grid((F + 31) / 32, (S + 31) / 32), block(32, 8);
  kda_transpose_cast_k<<<grid, block, 0, s>>>(src, (unsigned short*)dst, S, F);
  cuda_check(cudaPeekAtLastError());
}

// beta = sigmoid(b) and g = lb*sigmoid(exp(A_log[h])*(f + dt_bias[channel])), both elementwise and
// already coalesced (they are the remaining quarter of kda_prepare).
__global__ void kda_gate_k(const float* __restrict__ b, const float* __restrict__ f,
                           const float* __restrict__ dt_bias, const float* __restrict__ a_log,
                           float lb, unsigned short* __restrict__ beta, float* __restrict__ g,
                           int S, int H, int Dk) {
  int HD = H * Dk;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < S * HD; i += gridDim.x * blockDim.x) {
    int s = i / HD, rem = i % HD, hh = rem / Dk, dd = rem % Dk;
    float decay = __expf(a_log[hh]);
    g[i] = lb * (1.0f / (1.0f + __expf(-(decay * (f[i] + dt_bias[hh * Dk + dd])))));
    (void)s;
  }
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < S * H; j += gridDim.x * blockDim.x)
    beta[j] = __bfloat16_as_ushort(__float2bfloat16(1.0f / (1.0f + __expf(-b[j]))));
}

void kda_gate(const float* b, const float* f, const float* dt_bias, const float* a_log, float lb,
              void* beta_bf16, float* g, int S, int H, int Dk, Stream s) {
  int total = S * H * Dk;
  kda_gate_k<<<(total + 511) / 512, 512, 0, s>>>(b, f, dt_bias, a_log, lb, (unsigned short*)beta_bf16,
                                                g, S, H, Dk);
  cuda_check(cudaPeekAtLastError());
}

__global__ void kda_prepare_k(const float* __restrict__ qkv, const float* __restrict__ b,
                              const float* __restrict__ f, const float* __restrict__ dt_bias,
                              const float* __restrict__ a_log, float lb,
                              unsigned short* __restrict__ mqkv, unsigned short* __restrict__ beta,
                              float* __restrict__ g, int S, int F, int H, int Dk) {
  int total = S * F + S * H + S * H * Dk;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    if (i < S * F) {                       // qkv transpose+cast: [S,F] -> [F,S]
      int s = i / F, ff = i % F;
      mqkv[(size_t)ff * S + s] = __bfloat16_as_ushort(__float2bfloat16(qkv[i]));
    } else if (i < S * F + S * H) {        // beta
      int j = i - S * F;
      beta[j] = __bfloat16_as_ushort(__float2bfloat16(1.0f / (1.0f + __expf(-b[j]))));
    } else {                               // g
      int j = i - S * F - S * H;
      int h = j % H, d = j / H;            // j indexes [S, H, Dk] -> s = d
      (void)d;
      int s = j / (H * Dk), rem = j % (H * Dk);
      int hh = rem / Dk, dd = rem % Dk;
      float decay = __expf(a_log[hh]);
      g[j] = lb * (1.0f / (1.0f + __expf(-(decay * (f[j] + dt_bias[hh * Dk + dd])))));
      (void)h;
    }
  }
}

void kda_prepare(const float* qkv, const float* b, const float* f, const float* dt_bias,
                 const float* a_log, float lower_bound, void* mixed_qkv_bf16, void* beta_bf16,
                 float* g, int S, int F, int H, int Dk, Stream s) {
  int total = S * F + S * H + S * H * Dk;
  kda_prepare_k<<<(total + 511) / 512, 512, 0, s>>>(qkv, b, f, dt_bias, a_log, lower_bound,
                                                   (unsigned short*)mixed_qkv_bf16,
                                                   (unsigned short*)beta_bf16, g, S, F, H, Dk);
  cuda_check(cudaPeekAtLastError());
}

__global__ void swiglu_clamp_k(const float* __restrict__ gate, const float* __restrict__ up,
                               half* __restrict__ out, size_t n, float limit) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gv = fminf(gate[i], limit);
  float uv = fminf(fmaxf(up[i], -limit), limit);
  float silu = gv / (1.0f + __expf(-gv));
  out[i] = __float2half_rn(silu * uv);
}
void swiglu_clamp(const float* gate, const float* up, half* out, size_t n, float limit, Stream s) {
  swiglu_clamp_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(gate, up, out, n, limit);
  cuda_check(cudaPeekAtLastError());
}

__global__ void layernorm_f16_k(const half* __restrict__ x, const half* __restrict__ w,
                                const half* __restrict__ b, half* __restrict__ y, int rows, int dim,
                                float eps) {
  int r = blockIdx.x;
  const half* xr = x + (size_t)r * dim;
  float sum = 0, sq = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = __half2float(xr[i]);
    sum += v; sq += v * v;
  }
  __shared__ float ss[2];
  __shared__ float s1[32], s2[32];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, o); sq += __shfl_xor_sync(0xffffffffu, sq, o); }
  if (lane == 0) { s1[warp] = sum; s2[warp] = sq; }
  __syncthreads();
  if (warp == 0) {
    int nw = (blockDim.x + 31) / 32;
    float a = lane < nw ? s1[lane] : 0.f, c = lane < nw ? s2[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); c += __shfl_xor_sync(0xffffffffu, c, o); }
    if (lane == 0) { ss[0] = a / dim; ss[1] = c / dim; }
  }
  __syncthreads();
  float mean = ss[0], var = ss[1] - mean * mean;
  float inv = rsqrtf(var + eps);
  half* yr = y + (size_t)r * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = (__half2float(xr[i]) - mean) * inv * __half2float(w[i]) + __half2float(b[i]);
    yr[i] = __float2half_rn(v);
  }
}
void layernorm_f16(const half* x, const half* w, const half* b, half* y, int rows, int dim,
                   float eps, Stream s) {
  layernorm_f16_k<<<rows, 128, 0, s>>>(x, w, b, y, rows, dim, eps);
  cuda_check(cudaPeekAtLastError());
}

}}  // namespace helios::glue
