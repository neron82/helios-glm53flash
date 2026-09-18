#pragma once
// Model registry: config parsing, tensor placement plan, EXL3 group structs,
// expert RAM arena layout, and the full loader (trunk -> GPU0, routers/pools -> GPU1,
// experts -> RAM arena slabs + MTP preload to GPU1).
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/safetensors.hpp"

namespace helios {

// Runtime switch for MTP speculative decoding. Opt-in: it costs ~512MB of GPU0 cache and ~1.74GB of
// GPU1 (the draft layer's 288 experts), the latter taken from the expert slot pool.
inline bool mtp_enabled() {
  static const bool on = [] { const char* e = getenv("HELIOS_MTP"); return e && *e && *e != '0'; }();
  return on;
}

enum AttnKind : uint8_t { KDA = 0, MLA = 1 };

struct Config {
  int n_layers = 45;              // transformer layers 0..44 (+MTP at 45)
  int hidden = 4096;
  int heads = 64;                 // MLA heads and KDA channels-per-head count
  int q_lora = 1536, kv_lora = 512;
  int qk_nope = 256, v_dim = 256; // per-head dims (NoPE)
  int idx_heads = 32, idx_dim = 128, index_topk = 2048, kpool = 4;
  int n_expert = 288, topk = 8, n_shared = 1;
  int moe_inter = 2048, dense_inter = 12288;
  int hc_mult = 4, sinkhorn_iters = 20;
  float hc_eps = 1e-6f, rms_eps = 1e-5f;
  float routed_scale = 2.5f, swiglu_limit = 10.0f;
  float decay_lb = -5.0f;         // KDA gate lower bound
  int kda_conv_k = 4;
  int vocab = 154880;
  bool mhc = true, kpool_on = true, kpool_tail = true, index_share_mtp = true;
  std::vector<AttnKind> attn;     // per layer 0..44
  std::vector<bool> moe;          // per layer: sparse MoE vs dense MLP
  bool has_mtp = true;
  int mtp_layer = 45;             // tensor namespace index for the MTP draft layer
};

// One EXL3 quantized matrix: W[out,in] at K bits (K = trellis.d2*16/... see dims()).
struct Group {
  void* trellis = nullptr;        // device i16 [d0,d1,K/16]
  const half* suh = nullptr;      // device f16 [in]
  const half* svh = nullptr;      // device f16 [out]
  int mul1 = 0;                   // codebook multiplier word (host value)
  int K = 0, out = 0, in = 0;
  void set_dims(const std::vector<int64_t>& trellis_shape);
};

struct KDAWeights {
  // Transposed copies of the f16 low-rank weights (b [64,4096], f_a/g_a [128,4096], f_b/g_b [8192,128])
  // built at init on GPU0 so the tiled fp16 NN gemm can replace the scalar nt kernel below.
  const half *b_t = nullptr, *fa_t = nullptr, *fb_t = nullptr, *ga_t = nullptr, *gb_t = nullptr;
  Group qkv, o_proj;     // qkv: 4096 -> 24576 (q 8192 | k 8192 | v 8192 interleaved heads)
  const half* conv = nullptr;    // [24576,4] fp16 (depthwise k=4, channels first)
  const float* A_log = nullptr;  // [64]
  const float* dt_bias = nullptr;// [8192] = heads*dim channelwise
  const half* f_a = nullptr;     // [128,4096]
  const half* f_b = nullptr;     // [8192,128]
  const half* g_a = nullptr;     // [128,4096]
  const half* g_b = nullptr;     // [8192,128]
  const half* b = nullptr;       // [64,4096] beta proj
  const half* o_norm = nullptr;  // [128] per-head gated RMS norm weight
};

struct MLAWeights {
  Group q_a, q_b, kv_a, o_proj;
  const half* q_a_ln = nullptr;   // [1536]
  const half* kv_a_ln = nullptr;  // [512]
  const half* kv_b = nullptr;     // raw [64*512,512] fp16: head h rows h*512..h*512+255 = w_uk[h] (256x512), +256..+511 = w_uv[h]
  half* wuv_t = nullptr;          // w_uv transposed as [h][c][j] for coalesced o_absorb
};

struct IndexerWeights {
  Group wq_b;                     // 1536(latent q) -> 4096 (32*128)
  const half* wk = nullptr;       // [128,4096]
  const half* wproj = nullptr;    // [32,4096] score head weights
  const half* knorm_w = nullptr, *knorm_b = nullptr; // [128]
  const half* kpool_gate = nullptr; // [128,4096] compress gate
  const float* kpool_ape = nullptr; // [4,128] additive pos embeddings
};

struct MoeWeights {
  const half* router_gate = nullptr;   // [288,4096] GPU1
  const half* router_bias = nullptr;   // [288] fp16, mean-centered, GPU1
  Group shared[3];                     // gate,up,down GPU0 (shared expert)
  int arena_layer = -1;                // index into Model::slab_base (sparse layers only)
};

struct DenseMLP { Group gate, up, down; };  // GPU0

struct HCWeights {
  const float* fn_attn = nullptr, *fn_ffn = nullptr;   // f32 [24,16384] GPU0
  const half* fn_attn16 = nullptr, *fn_ffn16 = nullptr;// fp16 copies for decode path
  const float* base_attn = nullptr, *base_ffn = nullptr; // [24]
  const float* scale_attn = nullptr, *scale_ffn = nullptr;// [3]
};

struct Layer {
  int index = 0;
  int mla_ord = -1, kda_ord = -1;   // ordinal among MLA / KDA layers (cache indexing)
  AttnKind kind = KDA;
  bool moe = false;
  const void* embed_row_unused = nullptr; // placeholder
  KDAWeights kda; MLAWeights mla; IndexerWeights idx;
  MoeWeights moe_w; DenseMLP dense;
  HCWeights hc;
  const half* input_ln = nullptr, *post_ln = nullptr;
};

struct MTPWeights {
  MLAWeights mla; IndexerWeights idx; MoeWeights moe_w;
  Group eh_proj;                    // [4096, 8192] 4-bit: cat(hnorm(embed), enorm(trunk_h)) -> 4096
  const half* enorm = nullptr, *hnorm = nullptr, *shared_head_norm = nullptr;
  // The MTP block is a PLAIN residual block (no mHC), unlike trunk layers 0..44, so it carries the
  // usual pre-attn / pre-ffn RMSNorms.
  const half* input_ln = nullptr, *post_ln = nullptr;
  void* experts_gpu1 = nullptr;     // 288 slabs fully resident (288*stride)
};

struct Model {
  Config cfg;
  ShardSet shards;

  // ---- GPU0 (trunk) ----
  const void* embed = nullptr;          // bf16 [vocab,4096] (gather kernel converts)
  const half* final_norm = nullptr;     // [4096]
  Group lm_head;                        // 5-bit [vocab,4096]
  std::vector<Layer> layers;            // 0..44
  MTPWeights mtp;

  // ---- GPU1 (streaming) ----
  void* slot_pool = nullptr;            // n_slots * stride expert slot region
  int n_slots = 0;
  size_t slot_stride = 0;               // bytes per expert slab (fixed by dims)

  // ---- RAM arena ----
  char* arena = nullptr;                // sparse-layers (incl MTP) expert slabs, contiguous
  int arena_layers = 0;                 // 43 = layers 3..44 + MTP
  // slab(layer, e) = arena + (arena_slot_base[layer] + e) * slot_stride
  std::vector<int> arena_slot_base;     // per tensor-namespace layer -> slot base (or -1)
  std::vector<size_t> slab_off;         // 12 piece byte offsets within one expert slab:
                                        // [0..3] gate t/suh/svh/mul1, [4..7] up, [8..11] down

  // Stats
  size_t gpu0_bytes = 0, gpu1_bytes = 0, ram_bytes = 0;

  bool load(const std::string& dir, bool ram_only, bool verbose);
  const char* slab(int layer, int expert) const {
    return arena + (size_t)(arena_slot_base[layer] + expert) * slot_stride;
  }
};

// bf16 -> fp16 in place-safe convert (dst may differ)
void cvt_bf16_f16(const void* src, half* dst, size_t n);

}  // namespace helios