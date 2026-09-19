#include "engine/cache.hpp"
#include "../cuda/cuda_shim.hpp"
#include <cstdio>

namespace helios {

// The raw indexer ring must hold every row a single call writes or reads: a chunk of `chunk` rows,
// plus four so that decode's one-token calls still see the three rows before them. Resume positions
// are pool-aligned (see prefix_plan), so a chunk never starts mid-group.
int Cache::raw_ring_rows(int cap, int max_chunk) {
  int rows = max_chunk > 0 ? max_chunk + POOL : 512 + POOL;
  if (rows < 16) rows = 16;
  return rows;
}

CachePlan Cache::plan(const Config& cfg, int cap, int max_chunk) {
  CachePlan p;
  p.cap = cap;
  for (int l = 0; l < cfg.n_layers; l++) (cfg.attn[l] == MLA ? p.n_mla : p.n_kda)++;
  size_t b = 0;
  const int ring = raw_ring_rows(cap, max_chunk);
  b += (size_t)p.n_mla * cap * 512 * 2;                  // ckv fp16
  b += (size_t)p.n_mla * (size_t)ring * 256 * 2;         // idx ring k||gate (not per-token)
  b += (size_t)p.n_mla * ((size_t)cap / 4) * 128 * 2;    // pool keys (pool-major; the scalar
                                                         // kernel's [c][p] mirror is not allocated)
  b += (size_t)p.n_kda * 24576 * 4 * 2;                  // conv ring bf16, K=4
  b += (size_t)p.n_kda * 64 * 128 * 128 * 4;             // rec state fp32
  p.bytes = b;
  return p;
}

bool Cache::init(const Model& m, int cap, int max_chunk) {
  Device& g0 = Engine::instance().gpu(0);
  HELIOS_CUDA_CHECK(cudaSetDevice(g0.phys_idx()));
  CachePlan p = plan(m.cfg, cap, max_chunk);
  raw_rows_ = raw_ring_rows(cap, max_chunk);
  cap_ = cap;
  n_mla_ = p.n_mla; n_kda_ = p.n_kda;

  auto A = [&](size_t bytes, int align = 256) {
    void* q = g0.alloc(bytes, align);
    if (!q) { fprintf(stderr, "[cache] OOM allocating %.1f MB\n", bytes / 1048576.0); abort(); }
    return q;
  };
  ckv_ = (half*)A((size_t)p.n_mla * cap * 512 * 2);
  idx_ring_ = (half*)A((size_t)p.n_mla * (size_t)raw_rows_ * 256 * 2);
  pool_k_nt_ = (half*)A((size_t)p.n_mla * ((size_t)cap / 4) * 128 * 2);
  kda_conv_ = (char*)A((size_t)p.n_kda * 24576 * 4 * 2);
  kda_rec_ = (float*)A((size_t)p.n_kda * 64 * 128 * 128 * 4, 1024);

  maxM = max_chunk > 0 ? max_chunk : 512;
  xh = (half*)A((size_t)maxM * 4096 * 2);
  xa = (half*)A((size_t)maxM * 4096 * 2);
  xf = (half*)A((size_t)maxM * 4096 * 2);
  streams = (float*)A((size_t)4 * maxM * 4096 * 4);
  tmp_a = (half*)A((size_t)maxM * 24576 * 2);
  tmp_b = (half*)A((size_t)maxM * 16384 * 2);
  tmp_c = (half*)A((size_t)maxM * 8192 * 2);
  tmp_f = (float*)A((size_t)maxM * 8192 * 4);
  logits = (half*)A((size_t)m.cfg.vocab * 2);
  kda_g = (float*)A((size_t)maxM * 64 * 128 * 4);
  kda_beta = (half*)A((size_t)maxM * 64 * 2);
  kda_mqkv = (half*)A((size_t)24576 * maxM * 2);
  kda_out = (half*)A((size_t)maxM * 64 * 128 * 2);

  printf("[cache] cap=%d tokens mla=%d kda=%d kv=%.2fGB total=%.2fGB maxM=%d idx_ring=%d rows\n",
         cap, p.n_mla, p.n_kda, (double)p.bytes / GiB,
         (double)(p.bytes + (size_t)maxM * (4096 * 2 * 2 + 4 * 4096 * 4 + 24576 * 2 + 16384 * 2 + 8192 * 2 + 8192 * 4 + 64 * 128 * 4 + 64 * 2 + 24576 * 2 + 64 * 128 * 2) + (size_t)m.cfg.vocab * 2) / GiB,
         maxM, raw_rows_);
  return true;
}

}  // namespace helios