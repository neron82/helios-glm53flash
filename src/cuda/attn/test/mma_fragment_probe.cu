// Probe: verify the m16n8k16 fragment layouts intended for a tensor-core mla_sparse_decode.
// Build/run: nvcc -O2 -arch=sm_86 -o /tmp/mma_probe src/cuda/attn/test/mma_fragment_probe.cu && /tmp/mma_probe
// Measured 2026-09-18: "0/128 wrong, max|d|=2.86e-06 OK".
//
// Layouts established by it (they are the part that is easy to get wrong and slow to debug):
//   scores  S[16 keys x 8 heads] = K[16 keys x 16 dims] * Q[8 heads x 16 dims]^T   (row.col)
//     A = K read as [keys][dims] row-major;  B = Q read as [heads][dims] row-major, which is exactly
//     the column-major (k,n) placement mma wants. So the indexer/ckv cache layout needs no change.
//     C: c0=(row g, col tig*2), c1=(row g, col tig*2+1), c2=(row g+8, col tig*2), c3=(row g+8, +1).
//   output  O[heads][dims] = P[heads][keys] * K[keys][dims]   (row.col)
//     A = P (scores epilogue already in that exact fragment orientation, no transpose needed);
//     B needs the keys as [dims][keys] row-major, i.e. a per-tile transpose of ckv through smem.
// Computes S[16 keys x 8 heads] = K[16 keys x 16 dims] * Q[16 dims x 8 heads] (row.col) on the
// tensor cores and compares against a CPU reference, including the exp() epilogue and the
// key-sum reduction across lanes.
#include <cstdio>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <random>

__global__ void probe_k(const half* K, const half* Q, float* S, int keys, int heads, int dims) {
  int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  for (int dt = 0; dt < dims / 16; dt++) {
    const int o = dt * 16 + tig * 2;
    // A = K (16 keys x 16 dims, row-major): a0=(g,o), a1=(g+8,o), a2=(g,o+8), a3=(g+8,o+8)
    uint32_t a0 = *(const uint32_t*)(&K[(size_t)g * dims + o]);
    uint32_t a1 = *(const uint32_t*)(&K[(size_t)(g + 8) * dims + o]);
    uint32_t a2 = *(const uint32_t*)(&K[(size_t)g * dims + o + 8]);
    uint32_t a3 = *(const uint32_t*)(&K[(size_t)(g + 8) * dims + o + 8]);
    // B = Q (16 dims x 8 heads, col-major): b0=(o, g), b1=(o+8, g) with the [head][dim] layout
    uint32_t b0 = *(const uint32_t*)(&Q[(size_t)g * dims + o]);
    uint32_t b1 = *(const uint32_t*)(&Q[(size_t)g * dims + o + 8]);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  // C layout: c0=(row g, col tig*2), c1=(row g, col tig*2+1), c2=(row g+8, ...), c3=(row g+8, ...)
  S[((size_t)g * heads) + tig * 2] = c0;
  S[((size_t)g * heads) + tig * 2 + 1] = c1;
  S[((size_t)(g + 8) * heads) + tig * 2] = c2;
  S[((size_t)(g + 8) * heads) + tig * 2 + 1] = c3;
}

int main() {
  const int keys = 16, heads = 8, dims = 32;
  std::mt19937 rng(7);
  auto fr = [&]{ return (float)((int)(rng() % 2000) - 1000) / 500.f; };
  std::vector<float> kf((size_t)keys * dims), qf((size_t)heads * dims), ref((size_t)keys * heads);
  for (auto& x : kf) x = fr();
  for (auto& x : qf) x = fr();
  std::vector<half> kh(kf.size()), qh(qf.size());
  for (size_t i = 0; i < kf.size(); i++) kh[i] = __float2half_rn(kf[i]);
  for (size_t i = 0; i < qf.size(); i++) qh[i] = __float2half_rn(qf[i]);
  for (int k = 0; k < keys; k++)
    for (int h = 0; h < heads; h++) {
      double s = 0;
      for (int d = 0; d < dims; d++) s += __half2float(kh[(size_t)k * dims + d]) * __half2float(qh[(size_t)h * dims + d]);
      ref[(size_t)k * heads + h] = (float)s;
    }
  half *dk, *dq; float* ds;
  cudaMalloc(&dk, kh.size() * 2); cudaMalloc(&dq, qh.size() * 2); cudaMalloc(&ds, ref.size() * 4);
  cudaMemcpy(dk, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dq, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice);
  probe_k<<<1, 32>>>(dk, dq, ds, keys, heads, dims);
  auto e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
  std::vector<float> got(ref.size());
  cudaMemcpy(got.data(), ds, got.size() * 4, cudaMemcpyDeviceToHost);
  int bad = 0; float mx = 0;
  for (size_t i = 0; i < ref.size(); i++) { float d = fabsf(got[i] - ref[i]); mx = fmaxf(mx, d); if (d > 1e-2f) bad++; }
  printf("mma m16n8k16 scores: %d/%zu wrong, max|d|=%.2e %s\n", bad, ref.size(), mx, bad ? "FAIL" : "OK");
  return bad != 0;
}
