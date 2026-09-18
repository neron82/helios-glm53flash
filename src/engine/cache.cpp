#include "engine/cache.hpp"
#include "../cuda/cuda_shim.hpp"
#include <cstdio>

namespace helios {

CachePlan Cache::plan(const Config& cfg, int cap, bool mtp_draft) {
  CachePlan p;
  p.cap = cap;
  for (int l = 0; l < cfg.n_layers; l++) (cfg.attn[l] == MLA ? p.n_mla : p.n_kda)++;
  if (mtp_draft && cfg.has_mtp) p.n_mla++;     // +1 slot for the MTP draft layer
  size_t b = 0;
  b += (size_t)p.n_mla * cap * 512 * 2;                  // ckv fp16
  b += (size_t)p.n_mla * cap * 256 * 2;                  // idx plane k||gate
  b += (size_t)p.n_mla * ((size_t)cap / 4) * 128 * 2;    // pool keys
  b += (size_t)p.n_kda * 24576 * 4 * 2;                  // conv ring bf16, K=4
  b += (size_t)p.n_kda * 64 * 128 * 128 * 4;             // rec state fp32
  p.bytes = b;
  return p;
}

bool Cache::init(const Model& m, int cap, int max_chunk, bool mtp_draft) {
  Device& g0 = Engine::instance().gpu(0);
  HELIOS_CUDA_CHECK(cudaSetDevice(g0.phys_idx()));
  CachePlan p = plan(m.cfg, cap, mtp_draft);
  mtp_ord_ = (mtp_draft && m.cfg.has_mtp) ? p.n_mla - 1 : -1;
  cap_ = cap;
  n_mla_ = p.n_mla; n_kda_ = p.n_kda;

  auto A = [&](size_t bytes, int align = 256) {
    void* q = g0.alloc(bytes, align);
    if (!q) { fprintf(stderr, "[cache] OOM allocating %.1f MB\n", bytes / 1048576.0); abort(); }
    return q;
  };
  ckv_ = (half*)A((size_t)p.n_mla * cap * 512 * 2);
  idx_plane_ = (half*)A((size_t)p.n_mla * cap * 256 * 2);
  pool_k_ = (half*)A((size_t)p.n_mla * ((size_t)cap / 4) * 128 * 2);
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

  printf("[cache] cap=%d tokens mla=%d kda=%d kv=%.2fGB total=%.2fGB maxM=%d\n",
         cap, p.n_mla, p.n_kda, (double)p.bytes / GiB,
         (double)(p.bytes + (size_t)maxM * (4096 * 2 * 2 + 4 * 4096 * 4 + 24576 * 2 + 16384 * 2 + 8192 * 2 + 8192 * 4 + 64 * 128 * 4 + 64 * 2 + 24576 * 2 + 64 * 128 * 2) + (size_t)m.cfg.vocab * 2) / GiB,
         maxM);
  return true;
}

}  // namespace helios