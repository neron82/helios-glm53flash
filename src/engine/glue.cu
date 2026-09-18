#include "engine/glue.cuh"
#include <cuda_fp16.h>

namespace helios { namespace glue {

__global__ void embed_gather_k(const unsigned short* __restrict__ emb, const int* __restrict__ ids,
                               half* __restrict__ out, int hidden) {
  int t = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= hidden) return;
  unsigned short v = emb[(size_t)ids[t] * hidden + c];
  out[(size_t)t * hidden + c] = __float2half_rn(__bfloat162float(__ushort_as_bfloat16(v)));
}

void embed_gather(const void* embed_bf16, const int* ids, int n, half* out, int hidden, Stream s) {
  dim3 g(n, (hidden + 255) / 256);
  embed_gather_k<<<g, 256, 0, s>>>((const unsigned short*)embed_bf16, ids, out, hidden);
  cuda_check(cudaPeekAtLastError());
}

// streams layout is (R, H, D): streams[r*H*D + h*D + d]
__global__ void stream_expand_k(const half* __restrict__ h, float* __restrict__ st, int R, int H, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;      // i over R*D
  if (i >= R * D) return;
  int r = i / D, d = i % D;
  float v = __half2float(h[(size_t)r * D + d]);
  for (int hh = 0; hh < H; hh++) st[((size_t)r * H + hh) * D + d] = v;
}

void stream_expand(const half* h, float* streams, int n, int hidden, Stream s) {
  int tot = n * hidden;
  stream_expand_k<<<(tot + 255) / 256, 256, 0, s>>>(h, streams, n, 4, hidden);
  cuda_check(cudaPeekAtLastError());
}

// mean over the H streams: out[r,d] = (1/H) * sum_h streams[r,h,d]
__global__ void stream_mean_k(const float* __restrict__ st, half* __restrict__ out, int R, int H, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R * D) return;
  int r = i / D, d = i % D;
  float acc = 0.f;
  for (int hh = 0; hh < H; hh++) acc += st[((size_t)r * H + hh) * D + d];
  out[i] = __float2half_rn(acc / H);
}

// ---------------------------------------------------------------- MoE row gather / scatter
// The dense-MoE path processes one routed expert at a time: gather that expert's rows out of the
// permuted hidden state, run ordinary gemms, then scatter the weighted result back into y.
__global__ void gather_rows_k(const half* __restrict__ src, const int64_t* __restrict__ idx,
                              half* __restrict__ dst, int n, int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;      // i over n*h
  if (i >= n * h) return;
  int r = i / h, d = i % h;
  dst[i] = src[(size_t)idx[r] * h + d];
}

void gather_rows(const half* src, const int64_t* idx, half* dst, int n, int h, Stream s) {
  size_t tot = (size_t)n * h;
  gather_rows_k<<<(tot + 255) / 256, 256, 0, s>>>(src, idx, dst, n, h);
  cuda_check(cudaPeekAtLastError());
}

__global__ void scatter_add_rows_k(float* __restrict__ y, const float* __restrict__ src,
                                   const int64_t* __restrict__ idx, const half* __restrict__ w,
                                   int n, int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;      // i over n*h
  if (i >= n * h) return;
  int r = i / h, d = i % h;
  float wv = __half2float(w[r]);
  atomicAdd(&y[(size_t)idx[r] * h + d], wv * src[i]);
}

void scatter_add_rows(float* y, const float* src, const int64_t* idx, const half* w, int n, int h,
                      Stream s) {
  size_t tot = (size_t)n * h;
  scatter_add_rows_k<<<(tot + 255) / 256, 256, 0, s>>>(y, src, idx, w, n, h);
  cuda_check(cudaPeekAtLastError());
}

void stream_mean(const float* streams, half* out, int R, int H, int D, Stream s) {
  int tot = R * D;
  stream_mean_k<<<(tot + 255) / 256, 256, 0, s>>>(streams, out, R, H, D);
  cuda_check(cudaPeekAtLastError());
}

__global__ void add_inplace_k(half* __restrict__ acc, const half* __restrict__ x, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) acc[i] = __float2half_rn(__half2float(acc[i]) + __half2float(x[i]));
}

void add_inplace(half* acc, const half* x, size_t n, Stream s) {
  add_inplace_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(acc, x, n);
  cuda_check(cudaPeekAtLastError());
}

__global__ void moe_combine_k(half* __restrict__ out, const half* __restrict__ eo,
                              const float* __restrict__ w, int k, int hidden) {
  int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= hidden) return;
  float acc = 0.f;
  for (int j = 0; j < k; j++)
    acc += w[t * k + j] * __half2float(eo[((size_t)t * k + j) * hidden + c]);
  out[(size_t)t * hidden + c] = __float2half_rn(acc);
}

void moe_combine(half* out, const half* exp_out, const float* w, int n, int k, int hidden, Stream s) {
  dim3 g((hidden + 255) / 256, n);
  moe_combine_k<<<g, 256, 0, s>>>(out, exp_out, w, k, hidden);
  cuda_check(cudaPeekAtLastError());
}

__global__ void cast_f16_f32_k(const half* __restrict__ s, float* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) d[i] = __half2float(s[i]);
}
__global__ void cast_f32_f16_k(const float* __restrict__ s, half* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) d[i] = __float2half_rn(s[i]);
}
__global__ void cast_bf16_f16_k(const unsigned short* __restrict__ s, half* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) d[i] = __float2half_rn(__bfloat162float(__ushort_as_bfloat16(s[i])));
}
void cast_f16_f32(const half* src, float* dst, size_t n, Stream s) {
  cast_f16_f32_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, dst, n); cuda_check(cudaPeekAtLastError());
}
void cast_f32_f16(const float* src, half* dst, size_t n, Stream s) {
  cast_f32_f16_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, dst, n); cuda_check(cudaPeekAtLastError());
}
void cast_bf16_f16(const void* src, half* dst, size_t n, Stream s) {
  cast_bf16_f16_k<<<(unsigned)((n + 255) / 256), 256, 0, s>>>((const unsigned short*)src, dst, n);
  cuda_check(cudaPeekAtLastError());
}

__global__ void copy_row_k(const half* __restrict__ src, half* __restrict__ dst, int row, int width) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < width) dst[c] = src[(size_t)row * width + c];
}
void copy_row(const half* src, half* dst, int row, int width, Stream s) {
  copy_row_k<<<(width + 255) / 256, 256, 0, s>>>(src, dst, row, width);
  cuda_check(cudaPeekAtLastError());
}

__global__ void argmax_k(const half* __restrict__ x, int n, int* __restrict__ out) {
  float best = -INFINITY; int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = __half2float(x[i]);
    if (v > best) { best = v; bi = i; }
  }
  __shared__ float sb[256]; __shared__ int si[256];
  sb[threadIdx.x] = best; si[threadIdx.x] = bi;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (threadIdx.x < o && (sb[threadIdx.x + o] > sb[threadIdx.x] ||
        (sb[threadIdx.x + o] == sb[threadIdx.x] && si[threadIdx.x + o] < si[threadIdx.x]))) {
      sb[threadIdx.x] = sb[threadIdx.x + o]; si[threadIdx.x] = si[threadIdx.x + o];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = si[0];
}
void argmax_f16(const half* x, int n, int* out_idx, Stream s) {
  argmax_k<<<1, 256, 0, s>>>(x, n, out_idx);
  cuda_check(cudaPeekAtLastError());
}

}}  // namespace helios::glue