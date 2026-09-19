#pragma once
// GPU0 KV/state cache: MLA latent cache (ckv) + indexer planes (raw [k||gate] and pool keys),
// KDA conv ring + recurrent state. Single sequence (batch 1) v1; capacity in tokens.
#include "core/model.hpp"
#include "core/device.hpp"

namespace helios {

struct CachePlan {
  int cap = 0;                 // token capacity
  int n_mla = 0, n_kda = 0;    // layer counts
  size_t bytes = 0;
};

class Cache {
public:
  // Computes layout and allocates from gpu0's bump pool. Layers index by their ordinal
  // among MLA / KDA layers (Model::cfg.attn order).
  CachePlan plan(const Config& cfg, int cap, int max_chunk = 512);
  static int raw_ring_rows(int cap, int max_chunk);
  bool init(const Model& m, int cap, int max_chunk = 512);

  // ---- MLA ----
  half* ckv(int mla_ord) const { return ckv_ + (size_t)mla_ord * cap_ * 512; }
  // Raw [k||gate] rows exist only so a pool key can be completed from the four members of its group,
  // and only rows still in flight can be needed: the current call's, plus (during decode, one token
  // per call) the three before them. They live in a ring of `raw_rows` rows, indexed by position
  // modulo that size, instead of one row per token of context.
  half* idx_ring(int mla_ord) const { return idx_ring_ + (size_t)mla_ord * raw_rows_ * 256; }
  int raw_rows() const { return raw_rows_; }
  // Pool-major mirror [p][128]: the tensor-core indexer feeds the pool keys to mma as the B operand
  // in column-major form, which is exactly [p][d] contiguous in d. Storing it at write time avoids a
  // transposing (and cache-line-thrashing) load in every one of the ~500k indexer blocks per layer.
  half* pool_k_nt(int mla_ord) const { return pool_k_nt_ + (size_t)mla_ord * (cap_ / 4) * 128; }
  static constexpr int POOL = 4;   // tokens per indexer pool (mirrors attn::POOL)

  // ---- KDA ----  (conv state is bf16 [24576, K=4] per KDA layer)
  void* kda_conv(int kda_ord) const { return kda_conv_ + (size_t)kda_ord * 24576 * 4 * 2; }
  float* kda_rec(int kda_ord) const { return kda_rec_ + (size_t)kda_ord * 64 * 128 * 128; }

  int cap() const { return cap_; }
  int n_mla() const { return n_mla_; }
  int n_kda() const { return n_kda_; }
  int len() const { return len_; }
  void set_len(int n) { len_ = n; }

  // Working buffers (all on GPU0, sized for the model's max chunk of m rows)
  half* xh = nullptr;        // [M, 4096] fp16 current hidden
  half* xa = nullptr;        // [M, 4096] fp16 attn-sublayer input
  half* xf = nullptr;        // [M, 4096] fp16 ffn-sublayer input
  float* streams = nullptr;  // [4, M, 4096] fp32 stream stack
  half* tmp_a = nullptr;     // [M, 24576] fp16
  half* tmp_b = nullptr;     // [M, 16384] fp16
  half* tmp_c = nullptr;     // [M, 8192]  fp16
  float* tmp_f = nullptr;    // [M, 8192]  fp32
  half* logits = nullptr;    // [vocab] fp16 (last row only)
  int maxM = 0;               // rows the per-batch workspace holds (>= the runner's max_chunk)

  // KDA scratch
  float* kda_g = nullptr;    // [M, 64, 128] fp32 channelwise decay
  half* kda_beta = nullptr;  // [M, 64] fp16
  half* kda_mqkv = nullptr;  // [24576, M] fp16 channel-major (GDN layout)
  half* kda_out = nullptr;   // [M, 64, 128] fp16

private:
  size_t cap_ = 0;
  int len_ = 0;
  int n_mla_ = 0, n_kda_ = 0;
  half *ckv_ = nullptr, *idx_ring_ = nullptr, *pool_k_nt_ = nullptr;
  int raw_rows_ = 0;
  char* kda_conv_ = nullptr;   // bf16 conv ring, [kda][24576][4]
  float* kda_rec_ = nullptr;
};

}  // namespace helios