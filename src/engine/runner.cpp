#include "engine/runner.hpp"
#include "engine/glue.cuh"
#include "engine/glue2.cuh"
#include "cuda/attn/attn.cuh"
#include "cuda/aux/norm.cuh"
#include "cuda/aux/activation.cuh"
#include "cuda/aux/hc_mix.cuh"
#include "cuda/aux/routing.cuh"
#include "cuda/aux/dsa_topk.cuh"
#include "cuda/aux/gdn.cuh"
#include "cuda/aux/add.cuh"
#include "cuda/quant/exl3_gemm.cuh"
#include "cuda/quant/exl3_moe.cuh"
#include "cuda/cuda_shim.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>

namespace helios {

namespace { constexpr int kRoutedBits = 2; }   // routed expert matrices in this checkpoint

using namespace helios::aux;
using namespace helios::attn;

static bool dbg_sync() { static int v = -1; if (v < 0) v = getenv("HELIOS_DEBUG_SYNC") ? 1 : 0; return v != 0; }
#define DBGSYNC(st, tag) do { if (dbg_sync()) { cudaError_t e_ = cudaStreamSynchronize(st); \
    if (e_ != cudaSuccess) { fprintf(stderr, "[runner] FAILED at %s: %s\n", tag, cudaGetErrorString(e_)); abort(); } } } while (0)

// GPU0 work runs on the legacy stream: some exllamav3 kernels refuse launches on the engine's
// non-blocking streams in-process (driver returns invalid argument), and the engine is fully
// serial on GPU0 so there is no overlap to lose.
static inline cudaStream_t s0_stream() { return Engine::instance().gpu(0).stream(0); }

// Scoped current-device guard: kernel launchers (exl3 DevCtx, cudaGetDevice-based helpers) pick
// per-device resources from the *current* device, so it must match the stream's device.
struct DevGuard {
  int prev;
  explicit DevGuard(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
  ~DevGuard() { if (prev >= 0) cudaSetDevice(prev); }
};

static inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- init
bool Runner::init(Model& m, Cache& c, SlotMgr& sm, Tokenizer* tk, int max_chunk) {
  m_ = &m; c_ = &c; sm_ = &sm; tk_ = tk;
  max_chunk_ = max_chunk;
  HELIOS_CUDA_CHECK(cudaSetDevice(Engine::instance().gpu(0).phys_idx()));
  Device& g0 = Engine::instance().gpu(0);
  Device& g1 = Engine::instance().gpu(1);
  int M = max_chunk;
  auto A0 = [&](size_t bytes, int align = 256) {
    void* p = g0.alloc(bytes, align);
    if (!p) { fprintf(stderr, "[runner] gpu0 OOM %.1f MB\n", bytes / 1048576.0); abort(); }
    return p;
  };
  auto A1 = [&](size_t bytes, int align = 256) {
    void* p = g1.alloc(bytes, align);
    if (!p) { fprintf(stderr, "[runner] gpu1 OOM %.1f MB\n", bytes / 1048576.0); abort(); }
    return p;
  };

  int n_kda = 0;
  for (int l = 0; l < m.cfg.n_layers; l++) if (m.cfg.attn[l] == KDA) n_kda++;

  w0.tokens = (int*)A0((size_t)M * 4);
  w0.logits16 = (half*)A0((size_t)m.cfg.vocab * 2);
  w0.logits32 = (float*)A0((size_t)m.cfg.vocab * 4);
  w0.qkv = (float*)A0((size_t)M * 24576 * 4);
  w0.mqkv_bf16 = A0((size_t)24576 * M * 2);
  w0.conv_out_bf16 = A0((size_t)M * 24576 * 2);
  w0.kda_rec_bf16 = A0((size_t)M * 8192 * 2);
  w0.kda_norm = (half*)A0((size_t)M * 8192 * 2);
  w0.f_mid = (half*)A0((size_t)M * 128 * 2);
  w0.f_out = (float*)A0((size_t)M * 8192 * 4);
  w0.b_out = (float*)A0((size_t)M * 64 * 4);
  w0.g_mid = (half*)A0((size_t)M * 128 * 2);
  w0.g_out = (float*)A0((size_t)M * 8192 * 4);
  w0.kda_g = (float*)A0((size_t)M * 8192 * 4);
  w0.beta_bf16 = A0((size_t)M * 64 * 2);
  w0.qlat_a = (half*)A0((size_t)M * 1536 * 2);
  w0.q_nope = (half*)A0((size_t)M * 64 * 256 * 2);
  w0.q_lat = (half*)A0((size_t)M * 64 * 512 * 2);
  w0.lat_out = (half*)A0((size_t)M * 64 * 512 * 2);
  w0.o_abs = (half*)A0((size_t)M * 16384 * 2);
  w0.ckv_new = (half*)A0((size_t)M * 512 * 2);
  w0.idx_q = (half*)A0((size_t)M * 32 * 128 * 2);
  w0.idx_k = (half*)A0((size_t)M * 128 * 2);
  w0.idx_gate = (half*)A0((size_t)M * 128 * 2);
  w0.idx_w = (half*)A0((size_t)M * 32 * 2);
  int maxpools = (c_->cap() / 4) + 2;
  w0.scores = (half*)A0((size_t)M * maxpools * 2);
  w0.topk_idx = (int*)A0((size_t)M * 512 * 4);
  w0.raw_idx = (int*)A0((size_t)M * (512 * 4 + 4) * 4);
  w0.qpos = (int*)A0((size_t)M * 4);
  w0.attn_out = (float*)A0((size_t)M * 4096 * 4);
  w0.attn_out16 = (half*)A0((size_t)M * 4096 * 2);
  w0.ffn_out = (float*)A0((size_t)M * 4096 * 4);
  w0.ffn_out16 = (half*)A0((size_t)M * 4096 * 2);
  w0.mlp_gate = (float*)A0((size_t)M * 12288 * 4);
  w0.mlp_up = (float*)A0((size_t)M * 12288 * 4);
  w0.mlp_act = (half*)A0((size_t)M * 12288 * 2);
  w0.mlp_down = (float*)A0((size_t)M * 4096 * 4);
  w0.shared_out = (float*)A0((size_t)M * 4096 * 4);
  w0.a_had = (half*)A0((size_t)M * 24576 * 2);
  w0.a_had2 = (half*)A0((size_t)M * 24576 * 2);
  int chunks = hc_mix_num_chunks(M, 4 * 4096);
  w0.hc_partials = (float*)A0((size_t)M * chunks * 32 * 4);
  w0.hc_post = (float*)A0((size_t)M * 4 * 4);
  w0.hc_comb = (float*)A0((size_t)M * 16 * 4);
  w0.hc_collapsed = (half*)A0((size_t)M * 4096 * 2);
  w0.norm_out = (half*)A0((size_t)M * 4096 * 2);
  w0.mtp_in = (half*)A0((size_t)8192 * 2);
  if (mtp_enabled()) {
    const int rows = 8;                                  // max verify batch (k+1) when drafting
    mtp_hid_ = (half*)A0((size_t)4096 * 2);
    logits_multi_ = (half*)A0((size_t)rows * m.cfg.vocab * 2);
    HELIOS_CUDA_CHECK(cudaMallocHost((void**)&host_logits_multi_, (size_t)rows * m.cfg.vocab * 2));
    logits32_multi_ = (float*)A0((size_t)rows * m.cfg.vocab * 4);
    // KDA conv + recurrent state for every KDA layer: the draft round verifies k+1 tokens at once,
    // so the state must be restorable to the round's start before re-running only the accepted prefix.
    size_t per = (size_t)24576 * 4 * 2 + (size_t)64 * 128 * 128 * 4;
    mtp_ckpt_ = (half*)A0((size_t)c.n_kda() * per, 1024);
    kda_in_stride_ = (size_t)8 * 4096 * 2;             // up to 8 verified rows per layer
    kda_in_ = (half*)A0((size_t)c.n_kda() * kda_in_stride_);
  }
  w0.conv_w_bf16 = A0((size_t)n_kda * 24576 * 4 * 2);
  w0.dt_bias_bf16 = A0((size_t)n_kda * 8192 * 2);
  w0.onorm_bf16 = A0((size_t)n_kda * 128 * 2);

  w1.x = (half*)A1((size_t)M * 4096 * 2);
  w1.router_scores = (half*)A1((size_t)M * 288 * 2);
  w1.topk_i64 = (int64_t*)A1((size_t)M * 8 * 8);
  w1.topk_w = (half*)A1((size_t)M * 8 * 2);
  w1.y = (float*)A1((size_t)M * 4096 * 4);
  MoEPlan mp;
  mp.concurrency = std::min(8, exl3::moe_max_concurrency(g1.phys_idx()));
  if (mp.concurrency < 1) mp.concurrency = 1;
  mp.max_tokens_per_expert = M;
  w1.moe_tg = A1((size_t)mp.concurrency * M * 4096 * 2);
  w1.moe_tu = A1((size_t)mp.concurrency * M * 4096 * 2);
  w1.moe_ig = A1((size_t)mp.concurrency * M * 2048 * 2);
  w1.moe_iu = A1((size_t)mp.concurrency * M * 2048 * 2);
  w1.ec = (int64_t*)A1((size_t)(m.cfg.n_expert + 2) * 8);
  w1.tsorted = (int64_t*)A1((size_t)M * 8 * 8);
  w1.wsorted = (half*)A1((size_t)M * 8 * 2);
  w1.perm_ws = (int64_t*)A1(3 * (m.cfg.n_expert + 2) * 8);
  w1.tables = (void**)A1(9 * m.cfg.n_expert * sizeof(void*));
  {   // zeroed fallback slab: if an expert can never be made resident, the kernel reads zeros from
      // here instead of dereferencing a null table entry.
    void* zs = A1(m.slot_stride);
    HELIOS_CUDA_CHECK(cudaMemset(zs, 0, m.slot_stride));
    sm_->set_zero_slab(zs);
  }
  w1.dense_x = (half*)A1((size_t)M * 4096 * 2);
  w1.dense_g = (float*)A1((size_t)M * 2048 * 4);
  w1.dense_u = (float*)A1((size_t)M * 2048 * 4);
  w1.dense_a = (half*)A1((size_t)M * 2048 * 2);
  w1.dense_d = (float*)A1((size_t)M * 4096 * 4);
  w1.dense_had = (half*)A1((size_t)M * 4096 * 2);
  {   // routed experts share one mul1 word (verified across the checkpoint)
    const char* base = m.slab(3, 0);
    memcpy(&routed_mul1_, base + moe_off_[3], 4);
  }
  moe_plan_ = mp;
  for (int i = 0; i < 12; i++) moe_off_[i] = m.slab_off.empty() ? 0 : m.slab_off[i];

  // pinned host staging for GPU0 <-> GPU1 exchanges (P2P unavailable)
  HELIOS_CUDA_CHECK(cudaHostAlloc((void**)&w0.pin_h, (size_t)M * 4096 * 2, cudaHostAllocDefault));
  HELIOS_CUDA_CHECK(cudaMallocHost((void**)&w1.host_y, (size_t)M * 4096 * 4));
  HELIOS_CUDA_CHECK(cudaMallocHost((void**)&w0.host_ids, (size_t)M * 8 * 8));
  HELIOS_CUDA_CHECK(cudaMallocHost((void**)&w0.host_logits, (size_t)m.cfg.vocab * 2));

  // per-KDA-layer bf16 copies of conv weight / dt_bias / o_norm
  {
    int k = 0;
    for (int l = 0; l < m.cfg.n_layers; l++) {
      Layer& L = m.layers[l];
      if (L.kind != KDA) continue;
      glue::cast_f16_bf16(L.kda.conv, (char*)w0.conv_w_bf16 + (size_t)k * 24576 * 4 * 2,
                          24576 * 4, g0.stream(0));
      glue::cast_f32_bf16(L.kda.dt_bias, (char*)w0.dt_bias_bf16 + (size_t)k * 8192 * 2,
                          8192, g0.stream(0));
      glue::cast_f16_bf16(L.kda.o_norm, (char*)w0.onorm_bf16 + (size_t)k * 128 * 2,
                          128, g0.stream(0));
      HELIOS_CUDA_CHECK(cudaGetLastError());
      k++;
    }
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(g0.stream(0)));
  }
  {
    int a = 0, k = 0;
    for (int l = 0; l < m.cfg.n_layers; l++) {
      m.layers[l].mla_ord = m.cfg.attn[l] == MLA ? a++ : -1;
      m.layers[l].kda_ord = m.cfg.attn[l] == KDA ? k++ : -1;
    }
  }
  {   // transposed w_uv for every MLA layer (also the MTP layer if it has one)
    int made = 0;
    for (int l = 0; l < m.cfg.n_layers; l++) {
      if (m.cfg.attn[l] != MLA || !m.layers[l].mla.kv_b) continue;
      half* wt = (half*)A0((size_t)64 * 512 * 256 * 2);
      attn::wuv_transpose(m.layers[l].mla.kv_b, wt, s0_stream());
      m.layers[l].mla.wuv_t = wt;
      made++;
    }
    if (made) { HELIOS_CUDA_CHECK(cudaStreamSynchronize(s0_stream())); printf("[runner] wuv^T built for %d MLA layers\n", made); }
  }
  if (!getenv("HELIOS_KDA_GEMM") || atoi(getenv("HELIOS_KDA_GEMM")) != 0)
  {   // transposed KDA low-rank f16 weights: [N,K] -> [K,N] so the tiled NN gemm can be used
    size_t bytes = 0;
    for (int l = 0; l < m.cfg.n_layers; l++) {
      if (m.cfg.attn[l] != KDA) continue;
      KDAWeights& k = m.layers[l].kda;
      if (!k.b || !k.f_a || !k.f_b || !k.g_a || !k.g_b) continue;
      auto mk = [&](const half* src, int N, int K) -> const half* {
        half* dst = (half*)A0((size_t)K * N * 2);
        glue::transpose_f16_half(dst, src, N, K, s0_stream());
        bytes += (size_t)K * N * 2;
        return dst;
      };
      k.b_t = mk(k.b, 64, 4096);
      k.fa_t = mk(k.f_a, 128, 4096);
      k.fb_t = mk(k.f_b, 8192, 128);
      k.ga_t = mk(k.g_a, 128, 4096);
      k.gb_t = mk(k.g_b, 8192, 128);
    }
    if (bytes) {
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s0_stream()));
      printf("[runner] KDA low-rank transposes built: %.1f MB\n", bytes / 1048576.0);
    }
  }
  printf("[runner] max_chunk=%d moe_concurrency=%d gpu1_moe_scratch=%.1f MB\n", M, mp.concurrency,
         (2.0 * mp.concurrency * M * (4096 + 2048) * 2) / 1048576.0);
  reset();
  mtp_build();
  return true;
}

void Runner::reset() {
  pos_ = 0;
  Device& g0 = Engine::instance().gpu(0);
  // zero KDA states + ckv/idx planes are position-addressed, no clear needed beyond states
  HELIOS_CUDA_CHECK(cudaMemsetAsync(c_->kda_conv(0), 0, (size_t)c_->n_kda() * 24576 * 4 * 2, g0.stream(0)));
  HELIOS_CUDA_CHECK(cudaMemsetAsync(c_->kda_rec(0), 0, (size_t)c_->n_kda() * 64 * 128 * 128 * 4, g0.stream(0)));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(g0.stream(0)));
  tm_ = Timings{};
}


// ---------------------------------------------------------------- KDA layer
void Runner::kda_layer(Layer& L, int n, int pos) {
  Device& g0 = Engine::instance().gpu(0);
  DevGuard guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  // Per-batch stage timing (HELIOS_PROF): KDA is 34 of the 45 layers and was never instrumented,
  // so the attention profile did not account for a large part of each batch.
  static cudaEvent_t kev[8] = {};
  static float kacc[7] = {};
  static long kcalls = 0;
  const bool kprof = getenv("HELIOS_PROF") != nullptr;
  if (kprof && !kev[0]) for (int i = 0; i < 8; i++) cudaEventCreate(&kev[i]);
  auto kmark = [&](int i) { if (kprof) cudaEventRecord(kev[i], s); };
  kmark(0);
  int kda_ord = L.kda_ord;
  if (capture_) {
    // Snapshot the pre-state and this layer's input rows. dump(reason) is *not* used here: this runs
    // only for the k+1-token verification batch, and the snapshot must precede any state update.
    const size_t per = kda_state_bytes();
    char* ck = (char*)mtp_ckpt_ + (size_t)kda_ord * per;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(ck, c_->kda_conv(kda_ord), (size_t)24576 * 4 * 2,
                                      cudaMemcpyDeviceToDevice, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(ck + (size_t)24576 * 4 * 2, c_->kda_rec(kda_ord),
                                      (size_t)64 * 128 * 128 * 4, cudaMemcpyDeviceToDevice, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync((char*)kda_in_ + (size_t)kda_ord * kda_in_stride_, c_->xa,
                                      (size_t)n_capture_ * 4096 * 2, cudaMemcpyDeviceToDevice, s));
  }
  // 1) qkv projection (EXL3, 4-bit) with input/output Hadamard
  exl3::GroupWords qw{(const uint16_t*)L.kda.qkv.trellis, L.kda.qkv.suh, L.kda.qkv.svh, L.kda.qkv.mul1};
  int shape = exl3::gemm(w0.qkv, c_->xa, qw, n, 24576, 4096, L.kda.qkv.K, true, s, w0.a_had, false, 0, 0);
  if (dbg_sync()) fprintf(stderr, "[gemm] kda.qkv L%d n=%d bits=%d shape=%d\n", L.index, n, L.kda.qkv.K, shape);
  kmark(1);
  // 2) b / f / g low-rank paths (fp16 weights, fp32 out)
  // These five projections are 0.9s of a 28s prefill batch through the scalar nt kernel (a 16x16 tile
  // with one convert+FMA per MAC); with the init-time transposes they run on the tiled fp16 NN gemm.
  static const bool kda_gemm = getenv("HELIOS_KDA_GEMM") == nullptr ||
                               atoi(getenv("HELIOS_KDA_GEMM")) != 0;   // default on, HELIOS_KDA_GEMM=0 to disable
  if (kda_gemm && L.kda.b_t && n >= 8) {
    glue::gemm_nn_f16(w0.b_out, c_->xa, L.kda.b_t, n, 64, 4096, s, 0, true);
    glue::gemm_nn_f16(w0.f_mid, c_->xa, L.kda.fa_t, n, 128, 4096, s, 0, false);
    glue::gemm_nn_f16(w0.f_out, w0.f_mid, L.kda.fb_t, n, 8192, 128, s, 0, true);
    glue::gemm_nn_f16(w0.g_mid, c_->xa, L.kda.ga_t, n, 128, 4096, s, 0, false);
    glue::gemm_nn_f16(w0.g_out, w0.g_mid, L.kda.gb_t, n, 8192, 128, s, 0, true);
  } else {
    glue::gemm_nt_f16(w0.b_out, c_->xa, L.kda.b, n, 64, 4096, true, false, s);
    glue::gemm_nt_f16(w0.f_mid, c_->xa, L.kda.f_a, n, 128, 4096, false, false, s);
    glue::gemm_nt_f16(w0.f_out, w0.f_mid, L.kda.f_b, n, 8192, 128, true, false, s);
    glue::gemm_nt_f16(w0.g_mid, c_->xa, L.kda.g_a, n, 128, 4096, false, false, s);
    glue::gemm_nt_f16(w0.g_out, w0.g_mid, L.kda.g_b, n, 8192, 128, true, false, s);
  }
  kmark(2);
  // 3) gate/beta prep + channel-major cast
  static const bool kda_prep2 = getenv("HELIOS_KDA_PREP2") == nullptr ||
                                atoi(getenv("HELIOS_KDA_PREP2")) != 0;
  if (kda_prep2) {
    // Split so the transpose part is smem-tiled (the fused version wrote with stride S = one sector
    // per element: 28ms per layer, 0.95s of an 8192-token batch).
    glue::kda_transpose_cast(w0.mqkv_bf16, w0.qkv, n, 24576, s);
    glue::kda_gate(w0.b_out, w0.f_out, L.kda.dt_bias, L.kda.A_log, m_->cfg.decay_lb, w0.beta_bf16,
                   w0.kda_g, n, 64, 128, s);
  } else
  glue::kda_prepare(w0.qkv, w0.b_out, w0.f_out, L.kda.dt_bias, L.kda.A_log, m_->cfg.decay_lb,
                    w0.mqkv_bf16, w0.beta_bf16, w0.kda_g, n, 24576, 64, 128, s);
  kmark(3);
  // 4) depthwise conv (k=4) with streaming state
  const void* conv_w = (const char*)w0.conv_w_bf16 + (size_t)kda_ord * 24576 * 4 * 2;
  // activation=true: the KDA short conv is followed by SiLU (see colibri delta_attention.h)
  aux::cuda_causal_conv1d_update((const bfloat16*)w0.mqkv_bf16, (bfloat16*)c_->kda_conv(kda_ord),
                                 nullptr, (const bfloat16*)conv_w, nullptr,
                                 (bfloat16*)w0.conv_out_bf16, 1, 24576, n, 4, 4, true, false, s);
  kmark(4);
  // 5) delta rule recurrence (bf16 out, fp32 state)
  // L2 persistence for the recurrent state. The state is 4MB per layer and is read *and* written for
  // every token, i.e. 8MB/token/layer = 8.5us at DRAM bandwidth - which is what the scan costs
  // (measured 0.256ms per token across 34 layers, strictly linear in the token count). The device
  // set-aside is configured at init (cudaLimitPersistingL2CacheSize) but nothing ever set a stream
  // access policy, so it went unused. Window is this layer's state only, well under the 64MB aside.
  static const bool kda_l2 = getenv("HELIOS_KDA_L2") == nullptr ||
                             atoi(getenv("HELIOS_KDA_L2")) != 0;
  if (kda_l2) {
    cudaStreamAttrValue av{};
    av.accessPolicyWindow.base_ptr = (void*)c_->kda_rec(kda_ord);
    av.accessPolicyWindow.num_bytes = (size_t)64 * 128 * 128 * 4;
    av.accessPolicyWindow.hitRatio = 1.0f;
    av.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &av);
  }
  aux::cuda_recurrent_gated_delta_rule((const bfloat16*)w0.conv_out_bf16, w0.kda_g,
                                       (const bfloat16*)w0.beta_bf16, c_->kda_rec(kda_ord),
                                       (bfloat16*)w0.kda_rec_bf16, 1, n, 64, 64, 128, 128,
                                       1, nullptr, true, false, s);
  if (const char* dd = getenv("HELIOS_DUMP_KDA")) {   // debug: dump stage tensors
    auto w = [&](const char* n, const void* p, size_t bytes) {
      FILE* f = fopen((std::string(dd) + "." + n).c_str(), "wb");
      if (f) { std::vector<char> b(bytes); cudaMemcpy(b.data(), p, bytes, cudaMemcpyDeviceToHost);
               fwrite(b.data(), 1, bytes, f); fclose(f); }
    };
    w("conv", w0.conv_out_bf16, (size_t)n * 24576 * 2);
    w("gate", w0.kda_g, (size_t)n * 8192 * 4);
    w("z", w0.g_out, (size_t)n * 8192 * 4);
    w("rec", w0.kda_rec_bf16, (size_t)n * 8192 * 2);
    cudaStreamSynchronize(s);
  }
  kmark(5);
  // 6) gated RMS norm (bf16 in, fp32 gate, fp16 out)
  const void* onorm = (const char*)w0.onorm_bf16 + (size_t)kda_ord * 128 * 2;
  aux::gated_rms_norm(w0.kda_rec_bf16, onorm, aux::kBFloat16, w0.kda_norm, aux::kHalf,
                      w0.g_out, aux::kFloat, n * 64, 128, m_->cfg.rms_eps, 0.0f, 1, false, 1, s);
  // 7) output projection
  exl3::GroupWords ow{(const uint16_t*)L.kda.o_proj.trellis, L.kda.o_proj.suh, L.kda.o_proj.svh, L.kda.o_proj.mul1};
  exl3::gemm(w0.attn_out, w0.kda_norm, ow, n, 4096, 8192, L.kda.o_proj.K, true, s, w0.a_had);
  glue::cast_f32_f16(w0.attn_out, w0.attn_out16, (size_t)n * 4096, s);
  if (kprof) {
    kmark(6); kmark(7);
    cudaEventSynchronize(kev[7]);
    float seg[7], tot = 0;
    for (int i = 0; i < 7; i++) {
      cudaEventElapsedTime(&seg[i], kev[i], kev[i + 1]);
      kacc[i] += seg[i]; tot += seg[i];
    }
    kacc[6] += 0;   // seg[6] folded into tot below
    if (++kcalls % 34 == 0) {   // 34 KDA layers per batch
      fprintf(stderr, "[kda] batch %ld n=%d: %.1f ms (qkv=%.1f lowrank=%.1f prep=%.1f conv=%.1f "
                      "scan=%.1f norm+oproj=%.1f, %.2f ms/layer)\n",
              kcalls / 34, n, tot, kacc[0], kacc[1], kacc[2], kacc[3], kacc[4], kacc[5], tot / 34.0);
      for (int i = 0; i < 7; i++) kacc[i] = 0;
    }
  }
}

// ---------------------------------------------------------------- MLA layer
void Runner::mla_layer(Layer& L, int n, int pos) {
  Device& g0 = Engine::instance().gpu(0);
  DevGuard guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  int ord = L.mla_ord;
  static cudaEvent_t mev[9] = {};
  static float macc[8] = {};
  static long mla_calls = 0;
  const bool mla_prof = getenv("HELIOS_PROF") != nullptr;
  if (mla_prof && !mev[0]) for (int i = 0; i < 9; i++) cudaEventCreate(&mev[i]);
  auto mark = [&](int i) { if (mla_prof) cudaEventRecord(mev[i], s); };
  mark(0);
  // q path: q_a -> norm -> q_b
  exl3::GroupWords qa{(const uint16_t*)L.mla.q_a.trellis, L.mla.q_a.suh, L.mla.q_a.svh, L.mla.q_a.mul1};
  exl3::gemm(w0.qkv, c_->xa, qa, n, 1536, 4096, L.mla.q_a.K, true, s, w0.a_had);
  aux::rms_norm(w0.qkv, aux::kFloat, L.mla.q_a_ln, aux::kHalf, w0.qlat_a, aux::kHalf,
                n, 1536, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  DBGSYNC(s, "mla q_a");
  exl3::GroupWords qb{(const uint16_t*)L.mla.q_b.trellis, L.mla.q_b.suh, L.mla.q_b.svh, L.mla.q_b.mul1};
  exl3::gemm(w0.q_nope, w0.qlat_a, qb, n, 16384, 1536, L.mla.q_b.K, false, s, w0.a_had);
  DBGSYNC(s, "mla q_b");
  mark(1);
  // kv path: kv_a -> norm -> cache
  exl3::GroupWords ka{(const uint16_t*)L.mla.kv_a.trellis, L.mla.kv_a.suh, L.mla.kv_a.svh, L.mla.kv_a.mul1};
  int msh = getenv("HELIOS_MLA_SHAPE") ? atoi(getenv("HELIOS_MLA_SHAPE")) : 0;
  int kvsh = exl3::gemm(w0.qkv, c_->xa, ka, n, 512, 4096, L.mla.kv_a.K, true, s, w0.a_had2, false, msh, 0);
  if (const char* dd = getenv("HELIOS_SELFTEST")) {   // in-process: dump the exact device bytes used
    cudaStreamSynchronize(s);
    auto wr = [&](const char* nm, const void* p, size_t bytes) {
      FILE* f = fopen((std::string(dd) + "." + nm).c_str(), "wb");
      if (f) { std::vector<char> b(bytes); cudaMemcpy(b.data(), p, bytes, cudaMemcpyDeviceToHost);
               fwrite(b.data(), 1, bytes, f); fclose(f); }
    };
    wr("x", c_->xa, (size_t)n * 4096 * 2);
    wr("y", w0.qkv, (size_t)n * 512 * 4);
    wr("tr", ka.trellis, (size_t)256 * 32 * 64 * 2);
    wr("suh", ka.suh, (size_t)4096 * 2);
    wr("svh", ka.svh, (size_t)512 * 2);
    cudaStreamSynchronize(s);
    fprintf(stderr, "[selftest] dumped device bytes for mla.kv_a L%d (shape=%d)\n", L.index, kvsh);
  }
  if (dbg_sync()) fprintf(stderr, "[gemm] mla.kv_a L%d M=%d N=512 K=4096 bits=%d shape=%d x=%p y=%p tr=%p suh=%p svh=%p ahad=%p\n",
                          L.index, n, L.mla.kv_a.K, kvsh, (void*)c_->xa, (void*)w0.qkv,
                          ka.trellis, (void*)ka.suh, (void*)ka.svh, (void*)w0.a_had);
  if (const char* dd = getenv("HELIOS_DUMP_MLA")) {
    cudaStreamSynchronize(s);   // dumps must not race the engine stream
    auto w = [&](const char* nm, const void* p, size_t bytes) {
      FILE* f = fopen((std::string(dd) + "." + nm).c_str(), "wb");
      if (f) { std::vector<char> b(bytes); cudaMemcpy(b.data(), p, bytes, cudaMemcpyDeviceToHost);
               fwrite(b.data(), 1, bytes, f); fclose(f); }
    };
    w("kv_pre", w0.qkv, (size_t)n * 512 * 4);
    w("qa_pre", w0.qkv, (size_t)n * 512 * 4);
    cudaStreamSynchronize(s);
  }
  aux::rms_norm(w0.qkv, aux::kFloat, L.mla.kv_a_ln, aux::kHalf, w0.ckv_new, aux::kHalf,
                n, 512, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(c_->ckv(ord) + (size_t)pos * 512, w0.ckv_new, (size_t)n * 512 * 2,
                                    cudaMemcpyDeviceToDevice, s));
  DBGSYNC(s, "mla kv_a");
  mark(2);
  // indexer projections
  exl3::GroupWords wq{(const uint16_t*)L.idx.wq_b.trellis, L.idx.wq_b.suh, L.idx.wq_b.svh, L.idx.wq_b.mul1};
  exl3::gemm(w0.idx_q, w0.qlat_a, wq, n, 4096, 1536, L.idx.wq_b.K, false, s, w0.a_had);
  glue::gemm_nt_f16(w0.idx_k, c_->xa, L.idx.wk, n, 128, 4096, false, false, s);
  glue::layernorm_f16(w0.idx_k, L.idx.knorm_w, L.idx.knorm_b, w0.idx_k, n, 128, 1e-5f, s);
  glue::gemm_nt_f16(w0.idx_gate, c_->xa, L.idx.kpool_gate, n, 128, 4096, false, false, s);
  glue::gemm_nt_f16(w0.idx_w, c_->xa, L.idx.wproj, n, 32, 4096, false, false, s);
  DBGSYNC(s, "mla idx proj");
  mark(3);
  attn::kpool_write(w0.idx_k, w0.idx_gate, c_->idx_plane(ord), c_->pool_k(ord), L.idx.kpool_ape, pos, n,
                    c_->cap() / 4, s,
                    c_->pool_k_nt(ord));
  DBGSYNC(s, "mla kpool");
  mark(8);            // separates kpool_write from the qpos copy + q_latent that shared this segment
  // per-row positions
  {
    std::vector<int> qp(n);
    for (int i = 0; i < n; i++) qp[i] = pos + i;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.qpos, qp.data(), n * 4, cudaMemcpyHostToDevice, s));
  }
  int npools = (pos + n + 3) / 4;
  // absorbed q_lat (m <= 8 per launch)
  // q_lat[m,h,:] = q_nope[m,h,:] x w_uk[h]^T, with w_uk[h] = kv_b rows h*512..h*512+255, i.e.
  // [K=256][N=512] row-major with N (c) contiguous - exactly the layout the tiled NN gemm wants, so
  // this needs no transpose and reuses the kernel that is bit-exact against its CPU reference.
  // The scalar kernel here measured 460ms per 8192-token batch (~1.6 TFLOPS).
  static const bool qlat_gemm = getenv("HELIOS_QLAT_GEMM") == nullptr ||
                                atoi(getenv("HELIOS_QLAT_GEMM")) != 0;
  if (qlat_gemm && n >= 16) {
    for (int h = 0; h < 64; h++)
      glue::gemm_nn_f16(w0.q_lat + (size_t)h * 512, w0.q_nope + (size_t)h * 256,
                        L.mla.kv_b + (size_t)h * 512 * 512, n, 512, 256, s, 64 * 512, false,
                        64 * 256);
  } else
  attn::q_latent_t(w0.q_nope, L.mla.kv_b, w0.q_lat, n, s);
  DBGSYNC(s, "mla q_latent");
  mark(4);
  // Score and select in row blocks. The score matrix is M x npools, which at 250k context is ~1GB
  // written and read back per layer for nothing but the top-512; blocking it to 1024 rows keeps the
  // working set ~an order of magnitude smaller and is numerically identical (same scores, same
  // ordered top-k). This is the pattern that removed 256-launch overhead from q_latent/o_absorb.
  {
    const int ktop = std::min(512, npools);
    const int rb = std::min(n, getenv("HELIOS_IDX_BLOCK") ? atoi(getenv("HELIOS_IDX_BLOCK")) : 1024);
    for (int off = 0; off < n; off += rb) {
      const int rows = std::min(rb, n - off);
      attn::indexer_score(w0.idx_q + (size_t)off * 32 * 128, c_->pool_k(ord),
                          w0.idx_w + (size_t)off * 32, w0.qpos + off, w0.scores, rows, npools,
                          c_->cap() / 4, s,
                          c_->pool_k_nt(ord));
      aux::dsa_topk(w0.scores, npools, w0.topk_idx + (size_t)off * 512, rows, npools, ktop, 512,
                    nullptr, npools, s);
      attn::pool_expand_row(w0.topk_idx + (size_t)off * 512, w0.qpos + off,
                            w0.raw_idx + (size_t)off * (512 * POOL + POOL), rows, s);
    }
  }
  DBGSYNC(s, "mla indexer+topk");
  mark(5);
  attn::mla_sparse_decode(w0.q_lat, c_->ckv(ord), w0.raw_idx, 512 * 4 + 4, w0.lat_out, n, s);
  DBGSYNC(s, "mla sparse_decode");
  if (const char* dd = getenv("HELIOS_DUMP_MLA")) {
    cudaStreamSynchronize(s);   // dumps must not race the engine stream
    auto w = [&](const char* nm, const void* p, size_t bytes) {
      FILE* f = fopen((std::string(dd) + "." + nm).c_str(), "wb");
      if (f) { std::vector<char> b(bytes); cudaMemcpy(b.data(), p, bytes, cudaMemcpyDeviceToHost);
               fwrite(b.data(), 1, bytes, f); fclose(f); }
    };
    w("qlat", w0.q_lat, (size_t)n * 64 * 512 * 2);
    w("lat", w0.lat_out, (size_t)n * 64 * 512 * 2);
    w("ckv", c_->ckv(ord), (size_t)(pos + n) * 512 * 2);
    w("q_nope", w0.q_nope, (size_t)n * 64 * 256 * 2);
    cudaStreamSynchronize(s);
  }
  mark(6);
  if (L.mla.wuv_t) {
    // The scalar kernel issues one convert+FMA per MAC (1.5 TFLOPS at the 8192-row prefill shape,
    // 1.02s of a 28s batch). wuv_t is [K,N] = [512,256] per head, i.e. exactly the layout the tiled
    // fp16 NN gemm wants, so run it as 64 strided-output GEMMs instead.
    static const bool absorb_gemm = getenv("HELIOS_ABSORB_GEMM") == nullptr ||
                                    atoi(getenv("HELIOS_ABSORB_GEMM")) != 0;
    if (absorb_gemm && n >= 16) {
      for (int h = 0; h < 64; h++)
        glue::gemm_nn_f16(w0.o_abs + (size_t)h * 256, w0.lat_out + (size_t)h * 512,
                          L.mla.wuv_t + (size_t)h * 512 * 256, n, 256, 512, s, 64 * 256, false,
                          64 * 512);
    } else
    attn::o_absorb_t(w0.lat_out, L.mla.wuv_t, w0.o_abs, n, s);
  } else
  for (int off = 0; off < n; off += 8) {
    int mm = std::min(8, n - off);
    attn::o_absorb(w0.lat_out + (size_t)off * 64 * 512, L.mla.kv_b,
                   w0.o_abs + (size_t)off * 64 * 256, mm, s);
  }
  DBGSYNC(s, "mla o_absorb");
  exl3::GroupWords op{(const uint16_t*)L.mla.o_proj.trellis, L.mla.o_proj.suh, L.mla.o_proj.svh, L.mla.o_proj.mul1};
  exl3::gemm(w0.attn_out, w0.o_abs, op, n, 4096, 16384, L.mla.o_proj.K, true, s, w0.a_had);
  DBGSYNC(s, "mla o_proj");
  glue::cast_f32_f16(w0.attn_out, w0.attn_out16, (size_t)n * 4096, s);

  if (mla_prof) {
    mark(7);
    cudaEventSynchronize(mev[7]);
    cudaEventSynchronize(mev[8]);
    float seg[8];
    for (int i = 0; i < 8; i++) {
      if (i == 3 || i == 7) continue;             // 3 spans the extra mark; 7 is computed below
      cudaEventElapsedTime(&seg[i], mev[i], mev[i + 1]);
      macc[i] += seg[i];
    }
    float kpool_seg = 0, qlat_seg = 0;
    cudaEventElapsedTime(&kpool_seg, mev[3], mev[8]);   // kpool_write + qpos copy
    cudaEventElapsedTime(&qlat_seg, mev[8], mev[4]);    // q_latent_t
    macc[3] += kpool_seg;
    macc[7] += qlat_seg;
    if (++mla_calls % 11 == 0) {   // every MLA layer of one batch: per-batch deltas
      fprintf(stderr, "[mla] batch %ld (pos~%d): q_proj=%.1f kv=%.1f idx_proj=%.1f kpool=%.1f "
                      "idx_score=%.1f sparse=%.1f o_absorb=%.1f [kpool_write+qpos=%.1f "
                      "q_latent=%.1f ms]\n",
              mla_calls / 11, pos, macc[0], macc[1], macc[2], macc[3], macc[4], macc[5], macc[6],
              macc[3], macc[7]);
      for (int i = 0; i < 8; i++) macc[i] = 0;
    }
  }
}

// ---------------------------------------------------------------- FFN
void Runner::dense_mlp(Layer& L, int n) {
  DevGuard guard(Engine::instance().gpu(0).phys_idx());
  cudaStream_t s = s0_stream();
  exl3::GroupWords g{(const uint16_t*)L.dense.gate.trellis, L.dense.gate.suh, L.dense.gate.svh, L.dense.gate.mul1};
  exl3::GroupWords u{(const uint16_t*)L.dense.up.trellis, L.dense.up.suh, L.dense.up.svh, L.dense.up.mul1};
  exl3::GroupWords d{(const uint16_t*)L.dense.down.trellis, L.dense.down.suh, L.dense.down.svh, L.dense.down.mul1};
  exl3::gemm(w0.mlp_gate, c_->xf, g, n, 12288, 4096, L.dense.gate.K, true, s, w0.a_had);
  exl3::gemm(w0.mlp_up, c_->xf, u, n, 12288, 4096, L.dense.up.K, true, s, w0.a_had);
  glue::swiglu_clamp(w0.mlp_gate, w0.mlp_up, w0.mlp_act, (size_t)n * 12288, m_->cfg.swiglu_limit, s);
  exl3::gemm(w0.mlp_down, w0.mlp_act, d, n, 4096, 12288, L.dense.down.K, true, s, w0.a_had);
  glue::cast_f32_f16(w0.mlp_down, w0.ffn_out16, (size_t)n * 4096, s);
}

void Runner::moe_ffn(Layer& L, int n) {
  Device& g0 = Engine::instance().gpu(0);
  Device& g1 = Engine::instance().gpu(1);
  cudaStream_t s0 = s0_stream(), s1 = g1.stream(0);
  {   // GPU0 part first (shared expert), under the GPU0 device
    DevGuard gpu0_guard(g0.phys_idx());
    int E __attribute__((unused)) = m_->cfg.n_expert;
  }
  int E = m_->cfg.n_expert, K = m_->cfg.topk;
  // shared expert on GPU0 (explicit guard: the rest of this function is GPU1 work)
  {
    DevGuard shared_guard(g0.phys_idx());
    exl3::GroupWords g{(const uint16_t*)L.moe_w.shared[0].trellis, L.moe_w.shared[0].suh, L.moe_w.shared[0].svh, L.moe_w.shared[0].mul1};
    exl3::GroupWords u{(const uint16_t*)L.moe_w.shared[1].trellis, L.moe_w.shared[1].suh, L.moe_w.shared[1].svh, L.moe_w.shared[1].mul1};
    exl3::GroupWords d{(const uint16_t*)L.moe_w.shared[2].trellis, L.moe_w.shared[2].suh, L.moe_w.shared[2].svh, L.moe_w.shared[2].mul1};
    int force = getenv("HELIOS_FORCE_SHAPE") ? atoi(getenv("HELIOS_FORCE_SHAPE")) : 0;
    int shp = exl3::gemm(w0.mlp_gate, c_->xf, g, n, 2048, 4096, L.moe_w.shared[0].K, true, s0, w0.a_had,
                         false, force, 0);
    if (dbg_sync()) fprintf(stderr, "[gemm] shared.gate L%d n=%d bits=%d shape=%d trellis=%p\n",
                            L.index, n, L.moe_w.shared[0].K, shp, L.moe_w.shared[0].trellis);
    exl3::gemm(w0.mlp_up, c_->xf, u, n, 2048, 4096, L.moe_w.shared[1].K, true, s0, w0.a_had,
               false, force, 0);
    glue::swiglu_clamp(w0.mlp_gate, w0.mlp_up, w0.mlp_act, (size_t)n * 2048, m_->cfg.swiglu_limit, s0);
    exl3::gemm(w0.shared_out, w0.mlp_act, d, n, 4096, 2048, L.moe_w.shared[2].K, true, s0, w0.a_had,
               false, force, 0);
  }
  DBGSYNC(s0, "shared expert");
  static double t_sync0 = 0, t_d2h = 0, t_moe = 0, t_xfer_back = 0, t_other = 0;
  static int prof_n = 0;
  auto now2 = [&]() { return now_ms(); };
  double tt0 = now2();
  auto chk1 = [&](const char* tag) {
    if (!dbg_sync()) return;
    cudaError_t e = cudaStreamSynchronize(s1);
    if (e != cudaSuccess) { fprintf(stderr, "[runner] MoE stage '%s' failed: %s\n", tag, cudaGetErrorString(e)); abort(); }
  };
  DevGuard gpu1_guard(g1.phys_idx());   // from here on: GPU1 work only, device restored on exit
  // GPU0 -> GPU1 transfer of the MoE input
  double tt1 = now2(); t_other += tt1 - tt0;
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.pin_h, c_->xf, (size_t)n * 4096 * 2, cudaMemcpyDeviceToHost, s0));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s0));
  double tt2 = now2(); t_d2h += tt2 - tt1;
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w1.x, w0.pin_h, (size_t)n * 4096 * 2, cudaMemcpyHostToDevice, s1));
  chk1("x upload");
  double tt3 = now2(); t_other += tt3 - tt2;
  // router (GPU1)
  glue::gemm_nt_f16(w1.router_scores, w1.x, L.moe_w.router_gate, n, E, 4096, false, false, s1);
  chk1("router gemm");
  aux::routing_ds3_nogroup(w1.router_scores, L.moe_w.router_bias, w1.topk_i64, w1.topk_w,
                           n, E, K, m_->cfg.routed_scale, true, aux::ROUTING_ACT_SIGMOID, s1);
  chk1("routing topk");
  glue::moe_permute(w1.topk_i64, w1.topk_w, n, K, E, w1.ec, w1.tsorted, w1.wsorted, w1.perm_ws, s1);
  chk1("moe_permute");
  static std::vector<int64_t> dev_cnt(288);
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.host_ids, w1.topk_i64, (size_t)n * K * 8,
                                    cudaMemcpyDeviceToHost, s1));
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(dev_cnt.data(), w1.ec, (size_t)E * 8, cudaMemcpyDeviceToHost,
                                    s1));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s1));
  if (dbg_sync()) {
    std::vector<half> wtmp((size_t)n * K);
    HELIOS_CUDA_CHECK(cudaMemcpy(wtmp.data(), w1.topk_w, wtmp.size() * 2, cudaMemcpyDeviceToHost));
    fprintf(stderr, "[moe] L%d experts:", L.index);
    for (int i = 0; i < n * K; i++) fprintf(stderr, " %lld", (long long)w0.host_ids[i]);
    fprintf(stderr, "\n[moe] L%d weights:", L.index);
    for (int i = 0; i < n * K; i++) fprintf(stderr, " %.4f", __half2float(wtmp[i]));
    fprintf(stderr, "\n");
  }
  // Acquire slots for the experts used. The authoritative selection is the device-side per-expert
  // token count written by moe_permute: the kernel reads expert_count to decide which experts it
  // will touch, and it dereferences the table entry for each of them unconditionally. Driving the
  // acquire loop from the same counts guarantees every expert the kernel uses is resident.
  chk1("pre-slot");
  // The MTP draft layer's experts are fully resident on GPU1 and must NOT touch the slot pool: no
  // acquire, and crucially no note_request (the census ranks the trunk's streaming policy, and the
  // draft layer's routing would pollute it).
  const bool is_mtp = (L.index == m_->cfg.mtp_layer) && m_->mtp.experts_gpu1 != nullptr;
  int acquired = 0;
  if (!is_mtp) {
    sm_->begin_step();
    for (int e = 0; e < E; e++) {
      if (dev_cnt[e] <= 0) continue;
      sm_->note_request(L.index, e);
      if (sm_->find(L.index, e) < 0) { sm_->acquire(L.index, e); acquired++; }
    }
    sm_->sync_copies();
  }
  // pointer tables
  {
    static std::vector<const void*> host(9 * 288);
    size_t stride = is_mtp ? m_->slot_stride : sm_->stride();
    for (int e = 0; e < E; e++) {
      const char* base;
      if (is_mtp) {
        base = (const char*)m_->mtp.experts_gpu1 + (size_t)e * stride;
      } else {
        int slot = sm_->find(L.index, e);
        base = slot >= 0 ? sm_->slot_ptr(slot)
                         : (dev_cnt[e] > 0 ? (const char*)sm_->zero_slab() : nullptr);
      }
      host[0 * E + e] = base ? base + moe_off_[0] : nullptr;
      host[1 * E + e] = base ? base + moe_off_[1] : nullptr;
      host[2 * E + e] = base ? base + moe_off_[2] : nullptr;
      host[3 * E + e] = base ? base + moe_off_[4] : nullptr;
      host[4 * E + e] = base ? base + moe_off_[5] : nullptr;
      host[5 * E + e] = base ? base + moe_off_[6] : nullptr;
      host[6 * E + e] = base ? base + moe_off_[8] : nullptr;
      host[7 * E + e] = base ? base + moe_off_[9] : nullptr;
      host[8 * E + e] = base ? base + moe_off_[10] : nullptr;
      (void)stride;
    }
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(w1.tables, host.data(), 9 * E * sizeof(void*),
                                      cudaMemcpyHostToDevice, s1));
  }
  // Fail fast if a needed expert has no resident slot: the kernel dereferences the table entries
  // unconditionally, so a null pointer there is an illegal access instead of a clear error.
  if (!is_mtp) {
    int missing = 0, first = -1;
    for (int e = 0; e < E; e++)
      if (dev_cnt[e] > 0 && sm_->find(L.index, e) < 0) { missing++; if (first < 0) first = e; }
    if (missing) {
      // Rare, pool-state dependent: force the experts resident rather than launching over null
      // pointers. If that still fails, point them at a zeroed slab so the kernel reads zeros
      // instead of faulting - a logged, graceful degradation instead of a crash.
      int forced = 0, zeroed = 0;
      for (int e = 0; e < E; e++) {
        if (dev_cnt[e] <= 0 || sm_->find(L.index, e) >= 0) continue;
        if (sm_->force_resident(L.index, e) >= 0) forced++;
        else zeroed++;
      }
      if (forced || zeroed) {
        fprintf(stderr, "[moe] L%d n=%d missing=%d forced=%d zeroed=%d\n", L.index, n, missing,
                forced, zeroed);
        sm_->diag_expert(L.index, first);
        missing = 0;
        for (int e = 0; e < E; e++)
          if (dev_cnt[e] > 0 && sm_->find(L.index, e) < 0) missing++;
      }
      if (missing) {
        fprintf(stderr, "[moe] L%d n=%d UNRESOLVABLE missing=%d first=%d\n", L.index, n, missing,
                first);
        abort();
      }
    }
  }
  HELIOS_CUDA_CHECK(cudaMemsetAsync(w1.y, 0, (size_t)n * 4096 * 4, s1));
  exl3::ExpertTables tb;
  tb.gate_trellis = (const uint16_t**)(w1.tables + 0 * E);
  tb.gate_suh = (const half**)(w1.tables + 1 * E);
  tb.gate_svh = (const half**)(w1.tables + 2 * E);
  tb.up_trellis = (const uint16_t**)(w1.tables + 3 * E);
  tb.up_suh = (const half**)(w1.tables + 4 * E);
  tb.up_svh = (const half**)(w1.tables + 5 * E);
  tb.down_trellis = (const uint16_t**)(w1.tables + 6 * E);
  tb.down_suh = (const half**)(w1.tables + 7 * E);
  tb.down_svh = (const half**)(w1.tables + 8 * E);
  int active = 0;
  { std::vector<int64_t> cnt(E);
    HELIOS_CUDA_CHECK(cudaMemcpy(cnt.data(), w1.ec, E * 8, cudaMemcpyDeviceToHost));
    for (auto v : cnt) if (v > 0) active++; }
  if (active > 0 && getenv("HELIOS_MOE_DENSE")) {
    // Per-expert dense gemms instead of the fused grouped kernel: measured 14.6 TFLOPS at 64 rows
    // per expert through the plain gemm path versus ~1.8 TFLOPS through moe_grouped at the same
    // shapes, so the grouped kernel's per-expert cost is what limits prefill.
    std::vector<int> off(E + 1, 0);
    for (int e = 0; e < E; e++) off[e + 1] = off[e] + (int)(dev_cnt[e] > 0 ? dev_cnt[e] : 0);
    int done_rows = 0;
    for (int e = 0; e < E; e++) {
      const int cnt = off[e + 1] - off[e];
      if (cnt <= 0) continue;
      int slot = sm_->find(L.index, e);
      if (slot < 0) { fprintf(stderr, "[moe-dense] expert %d not resident\n", e); abort(); }
      const char* base = sm_->slot_ptr(slot);
      exl3::GroupWords gw{(const uint16_t*)(base + moe_off_[0]), (const half*)(base + moe_off_[1]),
                          (const half*)(base + moe_off_[2]), routed_mul1_};
      exl3::GroupWords uw{(const uint16_t*)(base + moe_off_[4]), (const half*)(base + moe_off_[5]),
                          (const half*)(base + moe_off_[6]), routed_mul1_};
      exl3::GroupWords dw{(const uint16_t*)(base + moe_off_[8]), (const half*)(base + moe_off_[9]),
                          (const half*)(base + moe_off_[10]), routed_mul1_};
      const int64_t* idx = w1.tsorted + off[e];
      glue::gather_rows(w1.x, idx, w1.dense_x, cnt, 4096, s1);
      // Routed experts are 2-bit in this checkpoint (the fused call passes K_gate/K_up/K_down = 2);
      // K here is the top-k count, not the bit width.
      exl3::gemm(w1.dense_g, w1.dense_x, gw, cnt, 2048, 4096, kRoutedBits, true, s1, w1.dense_had);
      exl3::gemm(w1.dense_u, w1.dense_x, uw, cnt, 2048, 4096, kRoutedBits, true, s1, w1.dense_had);
      glue::swiglu_clamp(w1.dense_g, w1.dense_u, w1.dense_a, (size_t)cnt * 2048, m_->cfg.swiglu_limit,
                         s1);
      exl3::gemm(w1.dense_d, w1.dense_a, dw, cnt, 4096, 2048, kRoutedBits, true, s1, w1.dense_had);
      glue::scatter_add_rows(w1.y, w1.dense_d, idx, w1.wsorted + off[e], cnt, 4096, s1);
      done_rows += cnt;
    }
    (void)done_rows;
    HELIOS_CUDA_CHECK(cudaEventRecord(g1.event(3), s1));
    HELIOS_CUDA_CHECK(cudaStreamWaitEvent(s0, g1.event(3), 0));
  } else if (active > 0) {
    // NOTE: the fused MoE kernel currently rejects launches on the engine's non-blocking
    // streams in this process (driver returns invalid argument) while the legacy stream works,
    // so run it there and order the following work with an event.
    exl3::moe_grouped(w1.x, w1.y, w1.ec, w1.tsorted, w1.wsorted, w1.moe_tg, w1.moe_tu,
                      w1.moe_ig, w1.moe_iu, tb, n, 4096, 2048, E, K, moe_plan_.max_tokens_per_expert,
                      moe_plan_.concurrency, 2, 2, 2, false, true, m_->cfg.swiglu_limit,
                      MOE_ACT_SILU, active, s1);
    HELIOS_CUDA_CHECK(cudaGetLastError());
  }
  if (dbg_sync()) {
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s1));
    std::vector<float> yy((size_t)n * 4096);
    HELIOS_CUDA_CHECK(cudaMemcpy(yy.data(), w1.y, yy.size() * 4, cudaMemcpyDeviceToHost));
    double sabs = 0; for (float v : yy) sabs += fabs(v);
    if (const char* dy = getenv("HELIOS_DUMP_MOE")) {
      if (L.index == 3) { FILE* f = fopen(dy, "wb"); if (f) { fwrite(yy.data(), 4, yy.size(), f); fclose(f); } }
    }
    std::vector<int64_t> ec(E + 2, 0), ts((size_t)n * 8);
    std::vector<half> ws((size_t)n * 8);
    HELIOS_CUDA_CHECK(cudaMemcpy(ec.data(), w1.ec, (E + 2) * 8, cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaMemcpy(ts.data(), w1.tsorted, ts.size() * 8, cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaMemcpy(ws.data(), w1.wsorted, ws.size() * 2, cudaMemcpyDeviceToHost));
    int nz = 0; for (int64_t v : ec) if (v) nz++;
    fprintf(stderr, "[moe] L%d y |sum| %.4f | ec nonzero %d (ec[0..7]=%lld,%lld,%lld,%lld) | "
                    "tsorted[0..3]=%lld,%lld,%lld,%lld wsorted[0..3]=%.3f,%.3f,%.3f,%.3f\n",
            L.index, sabs, nz, (long long)ec[0], (long long)ec[1], (long long)ec[2], (long long)ec[3],
            (long long)ts[0], (long long)ts[1], (long long)ts[2], (long long)ts[3],
            __half2float(ws[0]), __half2float(ws[1]), __half2float(ws[2]), __half2float(ws[3]));
  }
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w1.host_y, w1.y, (size_t)n * 4096 * 4, cudaMemcpyDeviceToHost, s1));
  double tt4 = now2(); t_moe += tt4 - tt3;
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s1));
  double tt5 = now2(); t_moe += tt5 - tt4;
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.ffn_out, w1.host_y, (size_t)n * 4096 * 4, cudaMemcpyHostToDevice, s0));
  double tt6 = now2(); t_xfer_back += tt6 - tt5;
  if (++prof_n % (getenv("HELIOS_PROF_EVERY") ? atoi(getenv("HELIOS_PROF_EVERY")) : 200) == 0 &&
      getenv("HELIOS_PROF"))
    fprintf(stderr, "[prof] n=%d other=%.1f d2h+sync=%.1f moe=%.1f back=%.1f\n",
            prof_n, t_other, t_d2h, t_moe, t_xfer_back);
  sm_->end_step();
  // routed (w0.ffn_out) + shared (w0.shared_out) -> fp16
  glue::add_f32_inplace(w0.ffn_out, w0.shared_out, (size_t)n * 4096, s0);
  glue::cast_f32_f16(w0.ffn_out, w0.ffn_out16, (size_t)n * 4096, s0);
}


// ---------------------------------------------------------------- chunk driver

void Runner::run_chunk(int n, int pos, bool prefill) {
  last_rows_ = n;
  (void)prefill;
  Device& g0 = Engine::instance().gpu(0);
  DevGuard chunk_guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  fprintf(stderr, "[runner] chunk n=%d pos=%d\n", n, pos);
  glue::embed_gather(m_->embed, w0.tokens, n, c_->xh, 4096, s);
  glue::stream_expand(c_->xh, c_->streams, n, 4096, s);
  DBGSYNC(s, "embed/expand");
  for (int l = 0; l < m_->cfg.n_layers; l++) layer_step(m_->layers[l], n, pos, c_->streams);
  // GLM-5.3 has no hc_head weights: the final stream collapse is the mean over the hyper-streams
  glue::stream_mean(c_->streams, c_->xh, n, 4, 4096, s);
}

void Runner::layer_step(Layer& L, int n, int pos, float* streams) {
  if (getenv("HELIOS_TRACE")) fprintf(stderr, "[trace] layer_step L%d n=%d pos=%d\n", L.index, n, pos);
  // Per-stage timing for prefill diagnosis (HELIOS_PROF=1): accumulated, printed at exit.
  static double t_hc = 0, t_attn = 0, t_ffn = 0;
  static long calls = 0;
  auto now3 = []() { return now_ms(); };
  bool prof = getenv("HELIOS_PROF") != nullptr;
  double p0 = prof ? now3() : 0;
  Device& g0 = Engine::instance().gpu(0);
  DevGuard guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  {
    // attention site
    aux::hc_mix(streams, L.hc.fn_attn16, true, L.hc.base_attn, L.hc.scale_attn, n, 4, 4096,
                hc_mix_num_chunks(n, 4 * 4096), m_->cfg.rms_eps, m_->cfg.hc_eps, m_->cfg.sinkhorn_iters,
                w0.hc_partials, w0.hc_post, w0.hc_comb, c_->xa, true, s);
    DBGSYNC(s, "hc_mix attn");
    // per-layer input norm on the collapsed mix output (transformer.forward: hc.mix -> norm -> attn)
    if (L.input_ln) {
      aux::rms_norm(c_->xa, aux::kHalf, L.input_ln, aux::kHalf, c_->xa, aux::kHalf,
                    n, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
    }
    if (prof) { double a0 = now3(); t_hc += a0 - p0; p0 = a0; }
    if (!getenv("HELIOS_LM_SKIP_ATTN")) {
      if (L.kind == KDA) kda_layer(L, n, pos);
      else mla_layer(L, n, pos);
    }
    if (prof) { double a1 = now3(); t_attn += a1 - p0; p0 = a1; }
    DBGSYNC(s, "attn layer");
    aux::hc_apply(streams, w0.attn_out16, true, w0.hc_post, w0.hc_comb, n, 4, 4096, s);
    DBGSYNC(s, "hc_apply attn");
    // ffn site
    aux::hc_mix(streams, L.hc.fn_ffn16, true, L.hc.base_ffn, L.hc.scale_ffn, n, 4, 4096,
                hc_mix_num_chunks(n, 4 * 4096), m_->cfg.rms_eps, m_->cfg.hc_eps, m_->cfg.sinkhorn_iters,
                w0.hc_partials, w0.hc_post, w0.hc_comb, c_->xf, true, s);
    if (L.post_ln) {
      aux::rms_norm(c_->xf, aux::kHalf, L.post_ln, aux::kHalf, c_->xf, aux::kHalf,
                    n, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
    }
    if (prof) { double f0 = now3(); t_hc += f0 - p0; p0 = f0; }
    if (!getenv("HELIOS_LM_SKIP_MOE")) ffn_layer(L, n, pos);
    DBGSYNC(s, "ffn layer");
    if (prof) { double f1 = now3(); t_ffn += f1 - p0; p0 = f1; }
    if (prof) {
      double a2 = now3();
      t_hc += a2 - p0;
      calls++;
      if (calls % 45 == 0)
        fprintf(stderr, "[stage] %ld layers: hc+norm=%.1fms attn=%.1fms moe=%.1fms\n",
                calls / 45, t_hc, t_attn, t_ffn);
    }
    aux::hc_apply(streams, w0.ffn_out16, true, w0.hc_post, w0.hc_comb, n, 4, 4096, s);
    DBGSYNC(s, "hc_apply ffn");
  }
}


// ---------------------------------------------------------------- bench
// Dense EXL3 gemm throughput at the shapes the fused MoE processes. Uses a resident routed expert's
// own weights (layer 3) for gate/up/down so the numbers are directly comparable to moe_grouped.
bool Runner::bench_gemm(int max_m) {
  Device& g0 = Engine::instance().gpu(0);
  DevGuard guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  Layer& L = m_->layers[3];
  exl3::GroupWords g{(const uint16_t*)L.moe_w.shared[0].trellis, L.moe_w.shared[0].suh,
                     L.moe_w.shared[0].svh, L.moe_w.shared[0].mul1};
  exl3::GroupWords u{(const uint16_t*)L.moe_w.shared[1].trellis, L.moe_w.shared[1].suh,
                     L.moe_w.shared[1].svh, L.moe_w.shared[1].mul1};
  exl3::GroupWords d{(const uint16_t*)L.moe_w.shared[2].trellis, L.moe_w.shared[2].suh,
                     L.moe_w.shared[2].svh, L.moe_w.shared[2].mul1};
  {   // transpose_f16_half self-test (used for the KDA low-rank weights): GPU vs host
    const int R = 70, C = 133;           // deliberately not tile multiples
    std::vector<half> a((size_t)R * C), b((size_t)C * R), ref((size_t)C * R);
    for (size_t i = 0; i < a.size(); i++) a[i] = __float2half_rn((float)((int)(i % 501) - 250) * 0.01f);
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) ref[(size_t)c * R + r] = a[(size_t)r * C + c];
    half *da, *db;
    HELIOS_CUDA_CHECK(cudaMalloc(&da, a.size() * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&db, b.size() * 2));
    HELIOS_CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    glue::transpose_f16_half(db, da, R, C, s);
    HELIOS_CUDA_CHECK(cudaMemcpy(b.data(), db, b.size() * 2, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (size_t i = 0; i < b.size(); i++) if (b[i] != ref[i]) bad++;
    printf("[bench] transpose_f16_half %dx%d -> %dx%d: %d/%zu wrong %s\n", R, C, C, R, bad, b.size(),
           bad ? "FAIL" : "OK");
    cudaFree(da); cudaFree(db);
  }
  {   // kda_transpose_cast self-test: [S,F] fp32 -> [F,S] bf16
    const int S = 37, F = 70;            // deliberately not tile multiples
    std::vector<float> src((size_t)S * F);
    for (size_t i = 0; i < src.size(); i++) src[i] = (float)((int)(i % 97) - 48) * 0.25f;
    std::vector<unsigned short> got((size_t)F * S), ref((size_t)F * S);
    for (int sI = 0; sI < S; sI++)
      for (int f = 0; f < F; f++)
        ref[(size_t)f * S + sI] = __bfloat16_as_ushort(__float2bfloat16(src[(size_t)sI * F + f]));
    float* ds; unsigned short* dd;
    HELIOS_CUDA_CHECK(cudaMalloc(&ds, src.size() * 4));
    HELIOS_CUDA_CHECK(cudaMalloc(&dd, got.size() * 2));
    HELIOS_CUDA_CHECK(cudaMemcpy(ds, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    glue::kda_transpose_cast(dd, ds, S, F, s);
    HELIOS_CUDA_CHECK(cudaMemcpy(got.data(), dd, got.size() * 2, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (size_t i = 0; i < got.size(); i++) if (got[i] != ref[i]) bad++;
    printf("[bench] kda_transpose_cast %dx%d: %d/%zu wrong %s\n", S, F, bad, got.size(),
           bad ? "FAIL" : "OK");
    cudaFree(ds); cudaFree(dd);
  }
  {   // gemm_nn_f16 self-test + timing at the o_absorb / q_latent shapes
    const int M = 256, N = 128, K = 96;      // non-tile-multiple on purpose
    std::vector<half> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N), ref((size_t)M * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = __float2half_rn((float)((int)(i % 37) - 18) * 0.05f);
    for (size_t i = 0; i < B.size(); i++) B[i] = __float2half_rn((float)((int)(i % 53) - 26) * 0.05f);
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++) {
        float s = 0;
        for (int k = 0; k < K; k++) s += __half2float(A[(size_t)i * K + k]) * __half2float(B[(size_t)k * N + j]);
        ref[(size_t)i * N + j] = __float2half_rn(s);
      }
    half *dA, *dB, *dC;
    HELIOS_CUDA_CHECK(cudaMalloc(&dA, A.size() * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&dB, B.size() * 2));
    HELIOS_CUDA_CHECK(cudaMalloc(&dC, C.size() * 2));
    HELIOS_CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice));
    glue::gemm_nn_f16(dC, dA, dB, M, N, K, s);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));   // cudaMemcpy only orders the default stream
    HELIOS_CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 2, cudaMemcpyDeviceToHost));
    // Tolerance, not bit-exactness: the kernel accumulates fp32 in a different order than the CPU
    // reference, so a small fraction of values land on the other side of an fp16 rounding tie.
    int bad = 0; float mx = 0;
    for (size_t i = 0; i < C.size(); i++) {
      float d = fabsf(__half2float(C[i]) - __half2float(ref[i]));
      if (d > mx) mx = d;
      if (d > 2e-2f) bad++;
    }
    printf("[bench] gemm_nn_f16 %dx%dx%d vs CPU: %d/%zu beyond 2e-2 (max|d|=%.2e) %s\n", M, N, K, bad,
           C.size(), mx, bad ? "FAIL" : "OK");
    cudaFree(dC);
    // timing at the two kernels this serves: o_absorb (per head) and q_latent (per head)
    {
      const int Ms = 8192, Ns = 256, Ks = 512;
      half* A2; half* B2; half* C2;
      HELIOS_CUDA_CHECK(cudaMalloc(&A2, (size_t)Ms * Ks * 2));
      HELIOS_CUDA_CHECK(cudaMalloc(&B2, (size_t)Ks * Ns * 2));
      HELIOS_CUDA_CHECK(cudaMalloc(&C2, (size_t)Ms * Ns * 2));
      HELIOS_CUDA_CHECK(cudaMemset(A2, 0, (size_t)Ms * Ks * 2));
      HELIOS_CUDA_CHECK(cudaMemset(B2, 0, (size_t)Ks * Ns * 2));
      glue::gemm_nn_f16(C2, A2, B2, Ms, Ns, Ks, s);
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
      HELIOS_CUDA_CHECK(cudaEventRecord(t0, s));
      for (int r = 0; r < 3; r++) glue::gemm_nn_f16(C2, A2, B2, Ms, Ns, Ks, s);
      HELIOS_CUDA_CHECK(cudaEventRecord(t1, s));
      HELIOS_CUDA_CHECK(cudaEventSynchronize(t1));
      float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
      double fl = 2.0 * Ms * Ns * Ks * 3;
      printf("[bench] gemm_nn_f16 8192x256x128 x3: %.1f ms -> %.1f TFLOPS\n", ms, fl / (ms * 1e-3) / 1e12);
      cudaFree(A2); cudaFree(B2); cudaFree(C2);
    }
    cudaFree(dA); cudaFree(dB);
  }
  printf("[bench] dense EXL3 gemm, gate/up 4096->2048 (K=%d), down 2048->4096 (K=%d)\n",
         L.moe_w.shared[0].K, L.moe_w.shared[2].K);
  printf("%6s %12s %12s %10s %10s\n", "m", "ms/layer", "us/expert", "GFLOP", "TFLOPS");
  for (int m = 1; m <= max_m; m = (m < 8 ? m * 2 : (m < 64 ? m + 8 : m * 2))) {
    // one pass = the three routed matrices for `m` rows, i.e. what one expert costs
    exl3::gemm(w0.mlp_gate, c_->xf, g, m, 2048, 4096, L.moe_w.shared[0].K, true, s, w0.a_had,
               false, 0, 0);
    glue::swiglu_clamp(w0.mlp_gate, w0.mlp_up, w0.mlp_act, (size_t)m * 2048, m_->cfg.swiglu_limit, s);
    exl3::gemm(w0.mlp_up, c_->xf, u, m, 2048, 4096, L.moe_w.shared[1].K, true, s, w0.a_had,
               false, 0, 0);
    exl3::gemm(w0.shared_out, w0.mlp_act, d, m, 4096, 2048, L.moe_w.shared[2].K, true, s, w0.a_had,
               false, 0, 0);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t t0, t1;
    HELIOS_CUDA_CHECK(cudaEventCreate(&t0));
    HELIOS_CUDA_CHECK(cudaEventCreate(&t1));
    int reps = m <= 8 ? 200 : (m <= 64 ? 60 : 20);
    HELIOS_CUDA_CHECK(cudaEventRecord(t0, s));
    for (int r = 0; r < reps; r++) {
      exl3::gemm(w0.mlp_gate, c_->xf, g, m, 2048, 4096, L.moe_w.shared[0].K, true, s, w0.a_had,
                 false, 0, 0);
      glue::swiglu_clamp(w0.mlp_gate, w0.mlp_up, w0.mlp_act, (size_t)m * 2048, m_->cfg.swiglu_limit, s);
      exl3::gemm(w0.mlp_up, c_->xf, u, m, 2048, 4096, L.moe_w.shared[1].K, true, s, w0.a_had,
                 false, 0, 0);
      exl3::gemm(w0.shared_out, w0.mlp_act, d, m, 4096, 2048, L.moe_w.shared[2].K, true, s, w0.a_had,
                 false, 0, 0);
    }
    HELIOS_CUDA_CHECK(cudaEventRecord(t1, s));
    HELIOS_CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    HELIOS_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= reps;
    double flop = 3.0 * 2.0 * (double)m * 2048 * 4096;   // gate + up + down
    printf("%6d %12.3f %12.1f %10.2f %10.2f\n", m, ms, ms * 1000.0, flop / 1e9,
           flop / (ms * 1e-3) / 1e12);
    HELIOS_CUDA_CHECK(cudaEventDestroy(t0));
    HELIOS_CUDA_CHECK(cudaEventDestroy(t1));
  }
  // Host-side launch cost: how long the CPU spends inside one exl3::gemm call (no sync), which is
  // what starves the GPU if the prefill stage breakdown shows large "stream gaps".
  {
    Layer& M3 = m_->layers[3];
    exl3::GroupWords w{(const uint16_t*)M3.mla.q_b.trellis, M3.mla.q_b.suh, M3.mla.q_b.svh,
                       M3.mla.q_b.mul1};
    int m = 2048;
    exl3::gemm(w0.q_nope, w0.qlat_a, w, m, 16384, 1536, M3.mla.q_b.K, false, s, w0.a_had);
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    double t0 = now_ms();
    const int reps = 200;
    for (int r = 0; r < reps; r++)
      exl3::gemm(w0.q_nope, w0.qlat_a, w, m, 16384, 1536, M3.mla.q_b.K, false, s, w0.a_had);
    double host_us = (now_ms() - t0) / reps * 1000.0;
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    printf("[bench] host time inside exl3::gemm (q_b, m=2048): %.1f us/call\n", host_us);
  }
  // MLA projection shapes at prefill batch sizes (the stage that dominates prefill)
  {
    Layer& M3 = m_->layers[3];
    struct Job { const char* name; const Group* w; int N; int K; bool fp32; };
    Job jobs[] = {{"q_a  4096->1536", &M3.mla.q_a, 1536, 4096, true},
                  {"q_b  1536->16384", &M3.mla.q_b, 16384, 1536, false},
                  {"kv_a 4096->512", &M3.mla.kv_a, 512, 4096, true},
                  {"wq_b 1536->4096", &M3.idx.wq_b, 4096, 1536, false},
                  {"kda_qkv 4096->24576", &m_->layers[0].kda.qkv, 24576, 4096, true},
                  {"o_proj 16384->4096", &M3.mla.o_proj, 4096, 16384, true}};   // m<=2048 only
    printf("\n[bench] MLA projection gemms (layer 3)\n%6s %-18s %10s %8s %8s\n", "m", "projection",
           "ms", "GFLOP", "TFLOPS");
    for (const Job& j : jobs) {
      exl3::GroupWords w{(const uint16_t*)j.w->trellis, j.w->suh, j.w->svh, j.w->mul1};
      for (int m : {1, 8, 128, 512}) {   // small m = the decode case; >2048 needs more workspace
        int reps = m <= 128 ? 8 : 3;
        if (m == 1) {   // host-side enqueue cost of one call at the decode shape
          for (int warm = 0; warm < 5; warm++)
            exl3::gemm(w0.qkv, c_->xa, w, 1, j.N, j.K, j.w->K, j.fp32, s, w0.a_had2, false, 0, 0);
          cudaStreamSynchronize(s);
          double t_enq = now_ms();
          for (int it = 0; it < 200; it++)
            exl3::gemm(w0.qkv, c_->xa, w, 1, j.N, j.K, j.w->K, j.fp32, s, w0.a_had2, false, 0, 0);
          double enq = (now_ms() - t_enq) / (200.0 * (int)(sizeof(jobs) / sizeof(jobs[0])));
          cudaStreamSynchronize(s);
          fprintf(stderr, "  [enqueue-only] %.2f us per gemm call at m=1 (host, %d shapes)\n",
                  enq * 1000.0, (int)(sizeof(jobs) / sizeof(jobs[0])));
        }
        exl3::gemm(w0.qkv, c_->xa, w, m, j.N, j.K, j.w->K, j.fp32, s, w0.a_had2, false, 0, 0);
        HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
        cudaEvent_t t0, t1;
        HELIOS_CUDA_CHECK(cudaEventCreate(&t0));
        HELIOS_CUDA_CHECK(cudaEventCreate(&t1));
        HELIOS_CUDA_CHECK(cudaEventRecord(t0, s));
        for (int r = 0; r < reps; r++)
          exl3::gemm(w0.qkv, c_->xa, w, m, j.N, j.K, j.w->K, j.fp32, s, w0.a_had2, false, 0, 0);
        HELIOS_CUDA_CHECK(cudaEventRecord(t1, s));
        HELIOS_CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        HELIOS_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= reps;
        double flop = 2.0 * (double)m * j.N * j.K;
        printf("%6d %-18s %10.2f %8.1f %8.2f\n", m, j.name, ms, flop / 1e9,
               flop / (ms * 1e-3) / 1e12);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
      }
    }
  }
  fflush(stdout);
  return true;
}

// Copy rate out of the RAM arena into GPU1 slots - isolates "how fast can we actually stream
// experts" from everything else in the MoE path.
bool Runner::bench_arena_copy(int n_slabs) {
  Device& g1 = Engine::instance().gpu(1);
  DevGuard guard(g1.phys_idx());
  cudaStream_t dma = g1.stream(1);
  void* dst = nullptr;
  HELIOS_CUDA_CHECK(cudaMalloc(&dst, m_->slot_stride));
  cudaEvent_t e0, e1;
  HELIOS_CUDA_CHECK(cudaEventCreate(&e0));
  HELIOS_CUDA_CHECK(cudaEventCreate(&e1));
  const int E = m_->cfg.n_expert;
  // warmup
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(dst, m_->slab(3, 0), m_->slot_stride, cudaMemcpyHostToDevice, dma));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(dma));
  HELIOS_CUDA_CHECK(cudaEventRecord(e0, dma));
  for (int i = 0; i < n_slabs; i++) {
    int e = i % E;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(dst, m_->slab(3, e), m_->slot_stride, cudaMemcpyHostToDevice, dma));
  }
  HELIOS_CUDA_CHECK(cudaEventRecord(e1, dma));
  HELIOS_CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0;
  HELIOS_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  double gb = (double)n_slabs * m_->slot_stride / 1e9;
  printf("[arena] %d slabs x %.2f MB from the RAM arena in %.1f ms = %.2f GB/s (%.3f ms/slab)\n",
         n_slabs, m_->slot_stride / 1048576.0, ms, gb / (ms / 1000.0), ms / n_slabs);
  fflush(stdout);
  cudaFree(dst);
  return true;
}

bool Runner::dump_layers(const std::vector<int>& ids, int upto, const std::string& path,
                         const std::string& sub_path) {
  reset();
  Device& g0 = Engine::instance().gpu(0);
  DevGuard dbg_guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  int n = (int)ids.size();
  glue::embed_gather(m_->embed, w0.tokens, n, c_->xh, 4096, s);
  DBGSYNC(s, "dbg embed");
  glue::stream_expand(c_->xh, c_->streams, n, 4096, s);
  DBGSYNC(s, "dbg expand");
  int attn_ord = 0, kda_ord = 0;
  fprintf(stderr, "[dbg] running layers, n=%d\n", n);
  for (int l = 0; l < upto && l < m_->cfg.n_layers; l++) {
    Layer& L = m_->layers[l];
    L.mla_ord = L.kind == MLA ? attn_ord++ : -1;
    L.kda_ord = L.kind == KDA ? kda_ord++ : -1;
    aux::hc_mix(c_->streams, L.hc.fn_attn16, true, L.hc.base_attn, L.hc.scale_attn, n, 4, 4096,
                hc_mix_num_chunks(n, 4 * 4096), m_->cfg.rms_eps, m_->cfg.hc_eps, m_->cfg.sinkhorn_iters,
                w0.hc_partials, w0.hc_post, w0.hc_comb, c_->xa, true, s);
    DBGSYNC(s, "dbg hc_mix attn");
    if (!sub_path.empty()) {   // pre-norm collapsed mix
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      FILE* fp = fopen((sub_path + ".pre").c_str(), "wb");
      if (fp) {
        std::vector<half> buf((size_t)n * 4096);
        HELIOS_CUDA_CHECK(cudaMemcpy(buf.data(), c_->xa, buf.size() * 2, cudaMemcpyDeviceToHost));
        fwrite(buf.data(), 2, buf.size(), fp);
        fclose(fp);
      }
    }
    if (L.input_ln)
      aux::rms_norm(c_->xa, aux::kHalf, L.input_ln, aux::kHalf, c_->xa, aux::kHalf, n, 4096,
                    m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
    if (!sub_path.empty()) {   // dump the sublayer input (normed mix) for the last layer  (see .pre)
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      FILE* fx = fopen((sub_path + ".in").c_str(), "wb");
      if (fx) {
        std::vector<half> buf((size_t)n * 4096);
        HELIOS_CUDA_CHECK(cudaMemcpy(buf.data(), c_->xa, buf.size() * 2, cudaMemcpyDeviceToHost));
        fwrite(buf.data(), 2, buf.size(), fx);
        fclose(fx);
      }
    }
    if (L.kind == KDA) kda_layer(L, n, 0); else mla_layer(L, n, 0);
    aux::hc_apply(c_->streams, w0.attn_out16, true, w0.hc_post, w0.hc_comb, n, 4, 4096, s);
    aux::hc_mix(c_->streams, L.hc.fn_ffn16, true, L.hc.base_ffn, L.hc.scale_ffn, n, 4, 4096,
                hc_mix_num_chunks(n, 4 * 4096), m_->cfg.rms_eps, m_->cfg.hc_eps, m_->cfg.sinkhorn_iters,
                w0.hc_partials, w0.hc_post, w0.hc_comb, c_->xf, true, s);
    if (!sub_path.empty()) {   // ffn input (post-norm) + pre-norm for the last layer
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      FILE* fx = fopen((sub_path + ".xffn_pre").c_str(), "wb");
      if (fx) { std::vector<half> b((size_t)n * 4096);
                HELIOS_CUDA_CHECK(cudaMemcpy(b.data(), c_->xf, b.size() * 2, cudaMemcpyDeviceToHost));
                fwrite(b.data(), 2, b.size(), fx); fclose(fx); }
    }
    if (L.post_ln)
      aux::rms_norm(c_->xf, aux::kHalf, L.post_ln, aux::kHalf, c_->xf, aux::kHalf, n, 4096,
                    m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
    if (!sub_path.empty()) {
      HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
      FILE* fx = fopen((sub_path + ".xf").c_str(), "wb");
      if (fx) { std::vector<half> b((size_t)n * 4096);
                HELIOS_CUDA_CHECK(cudaMemcpy(b.data(), c_->xf, b.size() * 2, cudaMemcpyDeviceToHost));
                fwrite(b.data(), 2, b.size(), fx); fclose(fx); }
    }
    ffn_layer(L, n, 0);
    aux::hc_apply(c_->streams, w0.ffn_out16, true, w0.hc_post, w0.hc_comb, n, 4, 4096, s);
  }
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  {
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[dbg] cannot write %s\n", path.c_str()); return false; }
    std::vector<float> buf((size_t)4 * n * 4096);
    HELIOS_CUDA_CHECK(cudaMemcpy(buf.data(), c_->streams, buf.size() * 4, cudaMemcpyDeviceToHost));
    fwrite(buf.data(), 4, buf.size(), f);
    fclose(f);
  }
  if (!sub_path.empty()) {
    FILE* f = fopen(sub_path.c_str(), "wb");
    std::vector<half> a((size_t)n * 4096), b((size_t)n * 4096);
    HELIOS_CUDA_CHECK(cudaMemcpy(a.data(), w0.attn_out16, a.size() * 2, cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaMemcpy(b.data(), w0.ffn_out16, b.size() * 2, cudaMemcpyDeviceToHost));
    fwrite(a.data(), 2, a.size(), f);
    fwrite(b.data(), 2, b.size(), f);
    fclose(f);
  }
  printf("[dbg] dumped %d layers to %s\n", upto, path.c_str());
  return true;
}

void Runner::final_head(int n) {
  DevGuard guard(Engine::instance().gpu(0).phys_idx());
  cudaStream_t s = s0_stream();
  DBGSYNC(s, "pre-final");
  // logits must come from the LAST token of the chunk, not the first
  const half* last = c_->xh + (size_t)(n > 0 ? n - 1 : 0) * 4096;
  aux::rms_norm(last, aux::kHalf, m_->final_norm, aux::kHalf, w0.norm_out, aux::kHalf,
                1, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  exl3::GroupWords hw{(const uint16_t*)m_->lm_head.trellis, m_->lm_head.suh, m_->lm_head.svh, m_->lm_head.mul1};
  exl3::gemm(w0.logits32, w0.norm_out, hw, 1, m_->cfg.vocab, 4096, m_->lm_head.K, true, s, w0.a_had);
  glue::cast_f32_f16(w0.logits32, w0.logits16, (size_t)m_->cfg.vocab, s);
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.host_logits, w0.logits16, (size_t)m_->cfg.vocab * 2,
                                    cudaMemcpyDeviceToHost, s));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
}

const half* Runner::logits_dev() const { return (const half*)w0.logits16; }

void Runner::prefill(const std::vector<int>& ids) {
  double t0 = now_ms();
  int n = (int)ids.size();
  // The KV/pool planes are position-addressed and bounded by the configured capacity: never let the
  // prefill run past it, or every cache write would index past its allocation.
  const int cap = c_->cap();
  if (pos_ >= cap) {
    fprintf(stderr, "[runner] context full (%d/%d): dropping %d prompt tokens\n", pos_, cap, n);
    return;
  }
  if (pos_ + n > cap) {
    fprintf(stderr, "[runner] prompt %d tokens exceeds remaining context %d of %d: truncating\n",
            n, cap - pos_, cap);
    n = cap - pos_;
  }
  // Chunk-major prefill (default, verified). The layer-major variant below streams each layer's
  // experts once per super-chunk instead of once per chunk (much less PCIe traffic for long
  // prompts) but still trips an illegal access inside the MoE path, so it stays opt-in.
  if (getenv("HELIOS_LAYER_MAJOR") == nullptr) {
    int done0 = 0, last0 = 0, chunk0 = std::min(max_chunk_, n);
    while (done0 < n) {
      chunk0 = std::min(max_chunk_, n - done0);
      last0 = chunk0;
      HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.tokens, ids.data() + done0, chunk0 * 4,
                                        cudaMemcpyHostToDevice, s0_stream()));
      double c0 = getenv("HELIOS_PROF") ? now_ms() : 0;
      run_chunk(chunk0, pos_, true);
      if (getenv("HELIOS_PROF"))
        fprintf(stderr, "[chunk] n=%d pos=%d wall=%.0fms (%.1f tok/s this chunk)\n", chunk0, pos_,
                now_ms() - c0, chunk0 * 1000.0 / (now_ms() - c0));
      pos_ += chunk0;
      done0 += chunk0;
    }
    final_head(last0);
    tm_.prefill_ms += now_ms() - t0;
    tm_.prefill_tokens += n;
    return;
  }
  // Layer-major prefill: a super-chunk of up to max_chunk_ tokens is pushed through the model
  // layer by layer, with an inner chunk loop. Keeping the layer outermost means each sparse
  // layer's experts are streamed from RAM once per super-chunk instead of once per chunk.
  const int S = max_chunk_;
  const int inner = std::min(128, S);
  int done = 0, last_rows = 0;
  while (done < n) {
    const int sc = std::min(S, n - done);
    last_rows = sc;
    cudaStream_t s = s0_stream();
    DevGuard guard(Engine::instance().gpu(0).phys_idx());
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.tokens, ids.data() + done, sc * 4, cudaMemcpyHostToDevice, s));
    glue::embed_gather(m_->embed, w0.tokens, sc, c_->xh, 4096, s);
    glue::stream_expand(c_->xh, c_->streams, sc, 4096, s);
    for (int l = 0; l < m_->cfg.n_layers; l++) {
      Layer& L = m_->layers[l];
      for (int c = 0; c < sc; c += inner) {
        int cn = std::min(inner, sc - c);
        layer_step(L, cn, pos_ + done + c, c_->streams + (size_t)c * 4 * 4096);
      }
    }
    pos_ += sc;
    done += sc;
    HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  }
  // final head from the last token of the last super-chunk
  {
    cudaStream_t s = s0_stream();
    glue::stream_mean(c_->streams + (size_t)(last_rows - 1) * 4 * 4096, c_->xh, 1, 4, 4096, s);
    final_head(1);
  }
  tm_.prefill_ms += now_ms() - t0;
  tm_.prefill_tokens += n;
}

void Runner::decode(const std::vector<int>& ids) {
  double t0 = now_ms();
  int n = (int)ids.size();
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.tokens, ids.data(), n * 4, cudaMemcpyHostToDevice, s0_stream()));
  run_chunk(n, pos_, false);
  pos_ += n;
  final_head(n);
  tm_.decode_ms += now_ms() - t0;
  tm_.decode_tokens += n;
}

size_t Runner::kda_state_bytes() const {
  return (size_t)24576 * 4 * 2 + (size_t)64 * 128 * 128 * 4;
}

void Runner::kda_capture_begin(int n) {
  if (!mtp_ready_) return;
  n_capture_ = n > 8 ? 8 : n;
  capture_ = true;
}

// Restore every KDA layer to the snapshot taken at the round's start, then advance it over exactly
// the accepted rows by replaying their captured inputs. Layer order matters: each layer's replay is
// self-contained (it only touches its own state), but the inputs were captured per layer.
void Runner::kda_rollback(int n, int pos) {
  capture_ = false;
  if (!mtp_ready_ || n <= 0) return;
  Device& g0 = Engine::instance().gpu(0);
  DevGuard guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  int ord = 0;
  for (int l = 0; l < m_->cfg.n_layers; l++) {
    Layer& L = m_->layers[l];
    if (L.kind != KDA || L.kda_ord < 0) continue;
    const int o = L.kda_ord;
    const size_t per = kda_state_bytes();
    char* ck = (char*)mtp_ckpt_ + (size_t)o * per;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(c_->kda_conv(o), ck, (size_t)24576 * 4 * 2,
                                      cudaMemcpyDeviceToDevice, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(c_->kda_rec(o), ck + (size_t)24576 * 4 * 2,
                                      (size_t)64 * 128 * 128 * 4, cudaMemcpyDeviceToDevice, s));
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(c_->xa, (char*)kda_in_ + (size_t)o * kda_in_stride_,
                                      (size_t)n * 4096 * 2, cudaMemcpyDeviceToDevice, s));
    kda_layer(L, n, pos);
    ord++;
  }
  (void)ord;
}

// Greedy MTP speculative decoding. Verified exactly against the non-drafting path: with temperature
// 0 the sampler is argmax, the drafts are verified by the trunk's own argmax, and only trunk-verified
// tokens are ever emitted - so HELIOS_MTP=1 must produce token-identical output to HELIOS_MTP=0.
std::vector<int> Runner::generate_mtp(const std::vector<int>& prompt, const GenParams& p,
                                      const std::function<bool(int)>& on_token) {
  // k drafts per round (k+1 rows verified). HELIOS_MTP_K=0 disables drafting: the loop then runs
  // exactly like the baseline (verify one token, take the trunk's own next token), which isolates a
  // loop/emission bug from a draft/rollback bug. Default 3.
  const int k = getenv("HELIOS_MTP_K") ? atoi(getenv("HELIOS_MTP_K")) : 3;
  const int vocab = m_->cfg.vocab;
  reset();
  prefill(prompt);
  std::vector<int> out;
  auto argmax_of = [&](const half* lg) {
    int best = 0; float bv = -1e30f;
    for (int i = 0; i < vocab; i++) {
      float v = __half2float(lg[i]);
      if (v > bv) { bv = v; best = i; }
    }
    return best;
  };
  auto seed_draft_hidden = [&](int row) {
    // the MTP input pairs the trunk's collapsed state at `row` with the embedding of the last token
    const half* src = c_->xh + (size_t)row * 4096;
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(mtp_hid_, src, (size_t)4096 * 2,
                                      cudaMemcpyDeviceToDevice, s0_stream()));
  };
  seed_draft_hidden(last_rows_ - 1);                // state that produced the post-prompt logits
  int t0 = argmax_of(w0.host_logits);
  int seed_row = last_rows_ - 1;
  long long drafted = 0, accepted_total = 0, rounds = 0;
  // `t0` is the pending token: produced by the trunk (from the prefill, then from each round's own
  // bonus) and emitted HERE, exactly once. The round below emits only the accepted drafts plus the
  // new bonus, so every round consumes `a+1` verified positions and emits `a+1` new tokens.
  bool stop = false;
  auto emit = [&](int tok) {
    if (stop || (int)out.size() >= p.max_tokens) return;
    out.push_back(tok);
    if (on_token && !on_token(tok)) stop = true;
  };
  if (t0 != tk_->eos_id()) emit(t0);
  while (!stop && (int)out.size() < p.max_tokens && t0 != tk_->eos_id()) {
    double t_round = now_ms();
    // ---- draft k tokens from the trunk state + the token just emitted
    int prev = t0;
    std::vector<int> drafts;
    for (int i = 0; i < k && (int)out.size() + (int)drafts.size() < p.max_tokens; i++) {
      int d = mtp_step(prev, pos_ + i);
      drafts.push_back(d);
      prev = d;
    }
    drafted += (long long)drafts.size();
    // ---- verify all of them in ONE trunk forward (this is where the win is: M = k+1, not k x M=1)
    std::vector<int> vbatch;
    vbatch.push_back(t0);
    for (int d : drafts) vbatch.push_back(d);
    const int vlen = (int)vbatch.size();
    const int start_pos = pos_;
    kda_capture_begin(vlen);
    HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.tokens, vbatch.data(), vlen * 4, cudaMemcpyHostToDevice,
                                      s0_stream()));
    run_chunk(vlen, start_pos, false);
    pos_ += vlen;
    final_head_multi(vlen);
    tm_.decode_tokens += vlen;
    // ---- accept the longest prefix the trunk agrees with (greedy)
    int a = 0;
    while (a < (int)drafts.size() && argmax_of(host_logits_multi_ + (size_t)a * vocab) == drafts[a]) a++;
    const int bonus = argmax_of(host_logits_multi_ + (size_t)a * vocab);
    rounds++;
    accepted_total += a;
    // ---- emit only the newly verified tokens: the accepted drafts and the trunk's own next token
    for (int i = 0; i < a; i++) emit(drafts[i]);
    emit(bonus);
    if (stop) break;
    // ---- roll the state back to the accepted prefix; the accepted rows' KV is already correct
    //      (the recurrence is causal), only the recurrent state ran past the end.
    const int accepted = a + 1;
    if (accepted < vlen) {
      c_->set_len(start_pos + accepted);
      pos_ = start_pos + accepted;
      kda_rollback(accepted, start_pos);
      seed_row = a;                                  // row a produced `bonus`
    } else {
      seed_row = accepted - 1;
    }
    seed_draft_hidden(seed_row);
    t0 = bonus;
    tm_.decode_ms += now_ms() - t_round;
  }
  if (getenv("HELIOS_MTP_TOKENS")) {
    fprintf(stderr, "[toks]");
    for (int t : out) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");
  }
  if (getenv("HELIOS_MTP_STATS"))
    fprintf(stderr, "[mtp] rounds=%lld drafts=%lld accepted=%lld (%.1f%% ) avg emitted/round=%.2f\n",
            rounds, drafted, accepted_total, drafted ? 100.0 * (double)accepted_total / (double)drafted : 0.0,
            rounds ? (double)out.size() / (double)rounds : 0.0);
  return out;
}

void Runner::mtp_build() {
  if (!mtp_enabled() || !m_->cfg.has_mtp || c_->mtp_ord() < 0 || !m_->mtp.experts_gpu1) return;
  mtp_layer_ = Layer{};
  mtp_layer_.index = m_->cfg.mtp_layer;
  mtp_layer_.kind = MLA;
  mtp_layer_.moe = true;
  mtp_layer_.mla_ord = c_->mtp_ord();
  mtp_layer_.mla = m_->mtp.mla;
  mtp_layer_.idx = m_->mtp.idx;
  mtp_layer_.moe_w = m_->mtp.moe_w;
  mtp_layer_.input_ln = m_->mtp.input_ln;
  mtp_layer_.post_ln = m_->mtp.post_ln;
  mtp_ready_ = true;
  printf("[mtp] draft layer ready: mla_ord=%d (own KV slot), 288 experts resident on GPU1\n",
         c_->mtp_ord());
}

// One MTP draft step. The block is a plain residual layer: it reads the previous hidden (the trunk's
// collapsed state `c_->xh` on the first step of a round, the previous block output afterwards) plus
// the previously drafted token, and returns the next drafted token. Everything it touches is a
// dedicated buffer or the MTP cache slot, so it cannot disturb trunk state.
int Runner::mtp_step(int tok, int pos) {
  Device& g0 = Engine::instance().gpu(0);
  DevGuard guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  // x = hnorm(embed(tok))   (embedding FIRST in the concat - qwen3_5_mtp.py: cat((x, y), dim=-1))
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.tokens, &tok, 4, cudaMemcpyHostToDevice, s));
  glue::embed_gather(m_->embed, w0.tokens, 1, w0.norm_out, 4096, s);
  aux::rms_norm(w0.norm_out, aux::kHalf, m_->mtp.hnorm, aux::kHalf, w0.mtp_in, aux::kHalf,
                1, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  // y = enorm(target_hidden)
  aux::rms_norm(mtp_hid_, aux::kHalf, m_->mtp.enorm, aux::kHalf, w0.mtp_in + 4096, aux::kHalf,
                1, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  // h = eh_proj(cat(x, y))  [8192] -> [4096]
  exl3::GroupWords ep{(const uint16_t*)m_->mtp.eh_proj.trellis, m_->mtp.eh_proj.suh,
                      m_->mtp.eh_proj.svh, m_->mtp.eh_proj.mul1};
  exl3::gemm(w0.qkv, w0.mtp_in, ep, 1, 4096, 8192, m_->mtp.eh_proj.K, true, s, w0.a_had);
  glue::cast_f32_f16(w0.qkv, mtp_hid_, 4096, s);
  // attn: norm -> MLA (draft cache slot, visibility bounded by `pos`) -> plain residual add
  aux::rms_norm(mtp_hid_, aux::kHalf, m_->mtp.input_ln, aux::kHalf, c_->xa, aux::kHalf,
                1, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  mla_layer(mtp_layer_, 1, pos);
  aux::add(mtp_hid_, aux::kHalf, w0.attn_out16, aux::kHalf, mtp_hid_, aux::kHalf, 4096, 4096, s);
  // ffn: norm -> MoE (resident experts, no slot manager, no census traffic) -> plain residual add
  aux::rms_norm(mtp_hid_, aux::kHalf, m_->mtp.post_ln, aux::kHalf, c_->xf, aux::kHalf,
                1, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  ffn_layer(mtp_layer_, 1, 0);
  aux::add(mtp_hid_, aux::kHalf, w0.ffn_out16, aux::kHalf, mtp_hid_, aux::kHalf, 4096, 4096, s);
  // head: shared_head.norm -> the SHARED lm_head
  aux::rms_norm(mtp_hid_, aux::kHalf, m_->mtp.shared_head_norm, aux::kHalf, w0.norm_out, aux::kHalf,
                1, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  exl3::GroupWords hw{(const uint16_t*)m_->lm_head.trellis, m_->lm_head.suh, m_->lm_head.svh,
                      m_->lm_head.mul1};
  exl3::gemm(w0.logits32, w0.norm_out, hw, 1, m_->cfg.vocab, 4096, m_->lm_head.K, true, s, w0.a_had);
  glue::cast_f32_f16(w0.logits32, w0.logits16, (size_t)m_->cfg.vocab, s);
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(w0.host_logits, w0.logits16, (size_t)m_->cfg.vocab * 2,
                                    cudaMemcpyDeviceToHost, s));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
  int best = 0; float bv = -1e30f;
  for (int i = 0; i < m_->cfg.vocab; i++) {
    float v = __half2float(w0.host_logits[i]);
    if (v > bv) { bv = v; best = i; }
  }
  return best;
}

// Verification needs the trunk's prediction at EVERY verified position, not just the last one.
void Runner::final_head_multi(int n) {
  Device& g0 = Engine::instance().gpu(0);
  DevGuard guard(g0.phys_idx());
  cudaStream_t s = s0_stream();
  aux::rms_norm(c_->xh, aux::kHalf, m_->final_norm, aux::kHalf, w0.norm_out, aux::kHalf,
                n, 4096, m_->cfg.rms_eps, 0.0f, 1.0f, false, 1, s);
  exl3::GroupWords hw{(const uint16_t*)m_->lm_head.trellis, m_->lm_head.suh, m_->lm_head.svh,
                      m_->lm_head.mul1};
  exl3::gemm(logits32_multi_, w0.norm_out, hw, n, m_->cfg.vocab, 4096, m_->lm_head.K, true, s,
             w0.a_had);
  glue::cast_f32_f16(logits32_multi_, logits_multi_, (size_t)n * m_->cfg.vocab, s);
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(host_logits_multi_, logits_multi_,
                                    (size_t)n * m_->cfg.vocab * 2, cudaMemcpyDeviceToHost, s));
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(s));
}

std::vector<int> Runner::generate(const std::vector<int>& prompt, const GenParams& p,
                                  const std::function<bool(int)>& on_token) {
  // Drafting is exact only when the sampler is plain argmax: any penalty or min-p filtering would
  // make the sampler disagree with the draft comparison.
  if (mtp_ready_ && p.temperature <= 0.0f && p.rep_penalty == 1.0f && p.min_p <= 0.0f)
    return generate_mtp(prompt, p, on_token);
  reset();
  prefill(prompt);
  std::vector<int> out;
  Sampler smp;
  smp.reset(p.seed ? p.seed : 1234);
  std::vector<int> recent = prompt;
  for (int i = 0; i < p.max_tokens; i++) {
    int tok = smp.sample(w0.host_logits, m_->cfg.vocab, p, recent);
    if (tok == tk_->eos_id()) break;
    out.push_back(tok);
    if (getenv("HELIOS_MTP_TOKENS")) fprintf(stderr, "[tok] %d\n", tok);
    recent.push_back(tok);
    if (on_token && !on_token(tok)) break;
    if ((int)out.size() >= p.max_tokens) break;
    decode({tok});
  }
  return out;
}

void Runner::ffn_layer(Layer& L, int n, int pos) {
  (void)pos;
  if (L.moe) moe_ffn(L, n);
  else dense_mlp(L, n);
}

}  // namespace helios