#pragma once
// Model registry: config parsing, tensor placement plan, EXL3 group structs,
// expert RAM arena layout, and the full loader (trunk -> GPU0, routers/pools -> GPU1,
// experts -> RAM arena slabs).
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/safetensors.hpp"
#include "core/expert_layout.hpp"

namespace helios {

enum AttnKind : uint8_t { KDA = 0, MLA = 1 };

struct Config {
  int n_layers = 45;              // transformer layers 0..44
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
  bool mhc = true, kpool_on = true, kpool_tail = true;
  std::vector<AttnKind> attn;     // per layer 0..44
  std::vector<bool> moe;          // per layer: sparse MoE vs dense MLP
};

// One EXL3 quantized matrix: W[out,in] at K bits (trellis third dimension / 16).
struct Group {
  void* trellis = nullptr;        // device i16 [in/16,out/16,16*K]
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
  int arena_layer = -1;                // logical expert base for this sparse layer
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

struct Model {
  std::string directory;
  Config cfg;
  ShardSet shards;

  // ---- GPU0 (trunk) ----
  const void* embed = nullptr;          // bf16 [vocab,4096] (gather kernel converts)
  const half* final_norm = nullptr;     // [4096]
  Group lm_head;                        // quantized [vocab,4096], bits read from checkpoint
  std::vector<Layer> layers;            // 0..44

  // ---- GPU1 (streaming) ----
  void* slot_pool = nullptr;            // n_slots * stride expert slot region
  int n_slots = 0;
  size_t slot_stride = 0;               // maximum trunk expert slab size

  // ---- RAM arena ----
  char* arena = nullptr;                // trunk sparse-layer expert slabs, contiguous
  int arena_layers = 0;                 // 42 = layers 3..44
  std::vector<int> arena_slot_base;     // logical expert index, or -1 for dense layers
  std::vector<size_t> arena_byte_base;  // compact RAM byte offset per layer
  std::vector<ExpertLayout> expert_layouts; // per-layer offsets, sizes and projection bits

  // Stats
  size_t gpu0_bytes = 0, gpu1_bytes = 0, ram_bytes = 0;

  bool load(const std::string& dir, bool ram_only, bool verbose);
  const char* slab(int layer, int expert) const {
    return arena + arena_byte_base.at(layer) + (size_t)expert * expert_layouts.at(layer).stride;
  }
};

// bf16 -> fp16 in place-safe convert (dst may differ)
void cvt_bf16_f16(const void* src, half* dst, size_t n);

}  // namespace helios
