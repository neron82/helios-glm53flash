#include "core/model.hpp"
#include "core/device.hpp"
#include "cuda/cuda_shim.hpp"
#include "../third_party/json.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <algorithm>
#include <fcntl.h>
#include <chrono>
#include <sys/mman.h>
#include <random>

namespace helios {
using nlohmann::json;

static std::string L(int l) { return "model.language_model.layers." + std::to_string(l); }

void cvt_bf16_f16(const void* src, half* dst, size_t n) {
  const uint16_t* s = (const uint16_t*)src;
  for (size_t i = 0; i < n; i++) {
    uint32_t bits = ((uint32_t)s[i]) << 16;
    dst[i] = __float2half_rn(*(float*)&bits);
  }
}
static void cvt_f32_f16(const void* src, half* dst, size_t n) {
  const float* s = (const float*)src;
  for (size_t i = 0; i < n; i++) dst[i] = __float2half_rn(s[i]);
}

void Group::set_dims(const std::vector<int64_t>& ts) { K = (int)(ts[2] / 16); }

// ---------- expert slab layout ----------
static size_t align64(size_t x) { return (x + 63) & ~(size_t)63; }
struct SlabLayout {
  size_t off[12];    // order: gate t,suh,svh,mul1, up..., down...
  size_t stride;
  size_t piece[12];  // byte sizes
};
static SlabLayout make_slab_layout(int hidden, int moe_inter, int K) {
  // trellis bytes = out*in*K/8 ; suh = in*2 ; svh = out*2 ; mul1 = 4
  size_t t = (size_t)hidden * moe_inter * K / 8;
  SlabLayout sl{};
  size_t o = 0;
  // gate: out=moe_inter in=hidden -> suh hidden, svh moe_inter
  sl.piece[0]=t;                 sl.off[0]=o; o+=t;
  sl.piece[1]=(size_t)hidden*2;  sl.off[1]=o; o+=hidden*2;
  sl.piece[2]=(size_t)moe_inter*2; sl.off[2]=o; o+=moe_inter*2;
  sl.piece[3]=4;                 sl.off[3]=o; o=align64(o+4);
  sl.piece[4]=t;                 sl.off[4]=o; o+=t;
  sl.piece[5]=(size_t)hidden*2;  sl.off[5]=o; o+=hidden*2;
  sl.piece[6]=(size_t)moe_inter*2; sl.off[6]=o; o+=moe_inter*2;
  sl.piece[7]=4;                 sl.off[7]=o; o=align64(o+4);
  // down: out=hidden in=moe_inter -> suh moe_inter, svh hidden
  sl.piece[8]=t;                 sl.off[8]=o; o+=t;
  sl.piece[9]=(size_t)moe_inter*2; sl.off[9]=o; o+=moe_inter*2;
  sl.piece[10]=(size_t)hidden*2;   sl.off[10]=o; o+=hidden*2;
  sl.piece[11]=4;                sl.off[11]=o; o=align64(o+4);
  sl.stride = o;
  return sl;
}

// ---------- loading jobs ----------
struct Job {
  const TensorInfo* ti = nullptr;
  void* dst = nullptr;
  const void* src_ram = nullptr;   // ARENA_TO_GPU source
  size_t bytes = 0, elems = 0;
  int device = -1;                 // -1 = RAM dst (no CUDA)
  enum Kind : uint8_t { RAW, BF16_F16, F32_F16, ROUTER_BIAS, MUL1, A2G } kind = RAW;
  std::string name;
};

struct Loader : Model {
  bool ram_only = false;
  std::vector<Job> jobs;
  SlabLayout slay;
  int missing = 0;
  bool verbose_alloc = true;

  const TensorInfo* T(const std::string& name) {
    auto* t = shards.find(name);
    if (!t) { fprintf(stderr, "[model] MISSING tensor %s\n", name.c_str()); missing++; }
    return t;
  }
  void* A(size_t bytes, int dev, int align = 256) {
    if (ram_only) return nullptr;
    if (verbose_alloc && bytes > (16u << 20))
      printf("[model] alloc dev%d %.1f MB (gpu%d total %.2f GB)\n", dev, bytes / 1048576.0, dev,
             (dev == 0 ? gpu0_bytes : gpu1_bytes) / 1073741824.0);
    void* p = Engine::instance().gpu(dev).alloc(bytes, align);
    if (dev == 0) gpu0_bytes += bytes; else gpu1_bytes += bytes;
    return p;
  }
  void j(const TensorInfo* t, void* dst, int dev, Job::Kind k, size_t elems = 0) {
    if (!t) return;
    if (dev >= 0 && !dst) return;   // ram_only: skip GPU-side jobs
    Job jb; jb.ti = t; jb.dst = dst; jb.device = dev; jb.kind = k;
    jb.bytes = t->bytes; jb.elems = elems ? elems : t->elems; jb.name = t->name;
    jobs.push_back(jb);
  }

  // load one EXL3 group: 4 tensors. dst_dev = GPU rank; mul1 lands in host struct.
  void load_group(const std::string& base, Group& g, int dst_dev) {
    auto* tr = T(base + ".trellis");
    auto* su = T(base + ".suh");
    auto* sv = T(base + ".svh");
    auto* mu = T(base + ".mul1");
    if (!tr || !su || !sv || !mu) return;
    g.set_dims(tr->shape);
    g.out = (int)sv->elems; g.in = (int)su->elems;
    g.trellis = ram_only ? nullptr : A(tr->bytes, dst_dev);
    g.suh = (const half*)(ram_only ? nullptr : A(su->bytes, dst_dev));
    g.svh = (const half*)(ram_only ? nullptr : A(sv->bytes, dst_dev));
    j(tr, g.trellis, dst_dev, Job::RAW);
    j(su, (void*)g.suh, dst_dev, Job::RAW);
    j(sv, (void*)g.svh, dst_dev, Job::RAW);
    j(mu, &g.mul1, -1, Job::MUL1);
  }

  void load_f16_from(const std::string& name, const half*& dst, int dev, bool bf16src) {
    auto* t = T(name);
    if (!t) return;
    dst = (const half*)(ram_only ? nullptr : A(t->bytes, dev));
    j(t, (void*)dst, dev, bf16src ? Job::BF16_F16 : Job::RAW);
  }
  void load_f32_from(const std::string& name, const float*& dst, int dev) {
    auto* t = T(name);
    if (!t) return;
    dst = (const float*)(ram_only ? nullptr : A(t->bytes, dev));
    j(t, (void*)dst, dev, Job::RAW);
  }

  bool parse_config(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f) { fprintf(stderr, "[model] no config.json\n"); return false; }
    json c; f >> c;
    json& tc = c.contains("text_config") ? c["text_config"] : c;
    cfg.hidden = tc.value("hidden_size", 4096);
    cfg.n_layers = tc.value("num_hidden_layers", 45);
    cfg.heads = tc.value("num_attention_heads", 64);
    cfg.q_lora = tc.value("q_lora_rank", 1536);
    cfg.kv_lora = tc.value("kv_lora_rank", 512);
    cfg.idx_heads = tc.value("index_n_heads", 32);
    cfg.idx_dim = tc.value("index_head_dim", 128);
    cfg.index_topk = tc.value("index_topk", 2048);
    cfg.kpool = tc.value("index_kpool", 4);
    cfg.kpool_on = tc.value("index_kpool_compress", true);
    cfg.kpool_tail = tc.value("index_kpool_always_select_tail", true);
    cfg.index_share_mtp = tc.value("index_share_for_mtp_iteration", true);
    cfg.n_expert = tc.value("n_routed_experts", 288);
    cfg.topk = tc.value("num_experts_per_tok", 8);
    cfg.moe_inter = tc.value("moe_intermediate_size", 2048);
    cfg.dense_inter = tc.value("intermediate_size", 12288);
    cfg.hc_mult = tc.value("hc_mult", 4);
    cfg.sinkhorn_iters = tc.value("hc_sinkhorn_iters", 20);
    cfg.hc_eps = tc.value("hc_eps", 1e-6f);
    cfg.rms_eps = tc.value("rms_norm_eps", 1e-5f);
    cfg.vocab = tc.value("vocab_size", 154880);
    cfg.mhc = tc.value("mhc", true);
    if (tc.contains("linear_attn_config")) {
      auto& la = tc["linear_attn_config"];
      cfg.decay_lb = la.value("gate_lower_bound", -5.0f);
      cfg.kda_conv_k = la.value("short_conv_kernel_size", 4);
    }
    cfg.qk_nope = cfg.v_dim = tc.value("head_dim", 0) ? tc["head_dim"].get<int>() : 256;
    if (tc.value("mla_use_nope", true)) cfg.qk_nope = 256;
    // per-layer kinds
    cfg.attn.assign(cfg.n_layers, KDA);
    cfg.moe.assign(cfg.n_layers, false);
    if (tc.contains("layer_types"))
      for (int i = 0; i < cfg.n_layers; i++)
        cfg.attn[i] = tc["layer_types"][i].get<std::string>() == "deepseek_sparse_attention" ? MLA : KDA;
    if (tc.contains("mlp_layer_types"))
      for (int i = 0; i < cfg.n_layers; i++)
        cfg.moe[i] = tc["mlp_layer_types"][i].get<std::string>() == "sparse";
    cfg.has_mtp = tc.value("num_nextn_predict_layers", 1) > 0;
    cfg.mtp_layer = cfg.n_layers;
    return true;
  }

  void load_attn(int l, Layer& ly) {
    std::string p = L(l) + ".self_attn";
    if (ly.kind == KDA) {
      load_group(p + ".qkv_proj", ly.kda.qkv, 0);
      load_group(p + ".o_proj", ly.kda.o_proj, 0);
      load_f16_from(p + ".conv1d.weight", ly.kda.conv, 0, true);
      load_f32_from(p + ".A_log", ly.kda.A_log, 0);
      load_f32_from(p + ".dt_bias", ly.kda.dt_bias, 0);
      load_f16_from(p + ".f_a_proj.weight", ly.kda.f_a, 0, false);
      load_f16_from(p + ".f_b_proj.weight", ly.kda.f_b, 0, false);
      load_f16_from(p + ".g_a_proj.weight", ly.kda.g_a, 0, false);
      load_f16_from(p + ".g_b_proj.weight", ly.kda.g_b, 0, false);
      load_f16_from(p + ".b_proj.weight", ly.kda.b, 0, false);
      load_f16_from(p + ".o_norm.weight", ly.kda.o_norm, 0, true);
    } else {
      load_group(p + ".q_a_proj", ly.mla.q_a, 0);
      load_group(p + ".q_b_proj", ly.mla.q_b, 0);
      load_group(p + ".kv_a_proj_with_mqa", ly.mla.kv_a, 0);
      load_group(p + ".o_proj", ly.mla.o_proj, 0);
      load_f16_from(p + ".q_a_layernorm.weight", ly.mla.q_a_ln, 0, true);
      load_f16_from(p + ".kv_a_layernorm.weight", ly.mla.kv_a_ln, 0, true);
      load_f16_from(p + ".kv_b_proj.weight", ly.mla.kv_b, 0, false);
    }
    if (ly.kind == MLA) {  // indexer exists only on DSA (MLA) layers
      std::string ip = p + ".indexer";
      load_group(ip + ".wq_b", ly.idx.wq_b, 0);
      load_f16_from(ip + ".wk.weight", ly.idx.wk, 0, false);
      load_f16_from(ip + ".weights_proj.weight", ly.idx.wproj, 0, false);
      load_f16_from(ip + ".k_norm.weight", ly.idx.knorm_w, 0, false);
      load_f16_from(ip + ".k_norm.bias", ly.idx.knorm_b, 0, false);
      load_f16_from(ip + ".index_kpool_compress_gate", ly.idx.kpool_gate, 0, false);
      load_f32_from(ip + ".index_kpool_compress_ape", ly.idx.kpool_ape, 0);
    }
  }

  void load_mlp(int l, Layer& ly) {
    std::string p = L(l) + ".mlp";
    if (ly.moe) {
      auto* gw = T(p + ".gate.weight");
      auto* gb = T(p + ".gate.e_score_correction_bias");
      if (gw) {
        ly.moe_w.router_gate = (const half*)(ram_only ? nullptr : A(gw->bytes, 1));
        j(gw, (void*)ly.moe_w.router_gate, 1, Job::RAW);
      }
      if (gb) {
        ly.moe_w.router_bias = (const half*)(ram_only ? nullptr : A(gb->elems * 2, 1));
        j(gb, (void*)ly.moe_w.router_bias, 1, Job::ROUTER_BIAS);
      }
      load_group(p + ".shared_experts.gate_proj", ly.moe_w.shared[0], 0);
      load_group(p + ".shared_experts.up_proj",  ly.moe_w.shared[1], 0);
      load_group(p + ".shared_experts.down_proj", ly.moe_w.shared[2], 0);
      ly.moe_w.arena_layer = arena_slot_base[l];
    } else {
      load_group(p + ".gate_proj", ly.dense.gate, 0);
      load_group(p + ".up_proj",  ly.dense.up, 0);
      load_group(p + ".down_proj", ly.dense.down, 0);
    }
  }

  void load_hc(int l, Layer& ly) {
    std::string p = L(l);
    auto add = [&](const std::string& nm, const float*& f32dst, const half*& f16dst) {
      auto* t = T(p + "." + nm);
      if (!t) return;
      f32dst = (const float*)(ram_only ? nullptr : A(t->bytes, 0));
      f16dst = (const half*)(ram_only ? nullptr : A(t->elems * 2, 0));
      j(t, (void*)f32dst, 0, Job::RAW);
      // f32 -> f16 copy: reuse source job via second pass (tmp holds f32)
      Job jb; jb.ti = t; jb.dst = (void*)f16dst; jb.device = 0; jb.kind = Job::F32_F16;
      jb.bytes = t->bytes; jb.elems = t->elems; jb.name = t->name; jobs.push_back(jb);
    };
    if (cfg.mhc) {
      add("hc_attn_fn", ly.hc.fn_attn, ly.hc.fn_attn16);
      add("hc_ffn_fn",  ly.hc.fn_ffn,  ly.hc.fn_ffn16);
      load_f32_from(p + ".hc_attn_base", ly.hc.base_attn, 0);
      load_f32_from(p + ".hc_ffn_base",  ly.hc.base_ffn, 0);
      load_f32_from(p + ".hc_attn_scale", ly.hc.scale_attn, 0);
      load_f32_from(p + ".hc_ffn_scale",  ly.hc.scale_ffn, 0);
    }
    load_f16_from(p + ".input_layernorm.weight", ly.input_ln, 0, true);
    load_f16_from(p + ".post_attention_layernorm.weight", ly.post_ln, 0, true);
  }

  void load_experts(int l, int slot_base) {
    std::string p = L(l) + ".mlp.experts.";
    for (int e = 0; e < cfg.n_expert; e++) {
      char* slab = arena + (size_t)(slot_base + e) * slay.stride;
      static const char* tn[3] = {"gate_proj", "up_proj", "down_proj"};
      static const char* pn[4] = {".trellis", ".suh", ".svh", ".mul1"};
      for (int t = 0; t < 3; t++) for (int q = 0; q < 4; q++) {
        int idx = t * 4 + q;
        auto* ti = shards.find(p + std::to_string(e) + "." + tn[t] + pn[q]);
        if (!ti) { fprintf(stderr, "[model] MISSING %s%d.%s%s\n", p.c_str(), e, tn[t], pn[q]); missing++; continue; }
        Job jb; jb.ti = ti; jb.dst = slab + slay.off[idx]; jb.device = -1;
        jb.kind = q == 3 ? Job::MUL1 : Job::RAW;
        // mul1 for experts: store 4B into slab piece directly (device code reads it from RAM/slot)
        jb.kind = Job::RAW;
        jb.bytes = ti->bytes; jb.name = ti->name;
        jobs.push_back(jb);
      }
    }
  }

  void load_mtp() {
    if (!cfg.has_mtp) return;
    int l = cfg.mtp_layer;
    std::string p = L(l);
    // attn (MLA-type) + indexer
    load_group(p + ".self_attn.q_a_proj", mtp.mla.q_a, 0);
    load_group(p + ".self_attn.q_b_proj", mtp.mla.q_b, 0);
    load_group(p + ".self_attn.kv_a_proj_with_mqa", mtp.mla.kv_a, 0);
    load_group(p + ".self_attn.o_proj", mtp.mla.o_proj, 0);
    load_f16_from(p + ".self_attn.q_a_layernorm.weight", mtp.mla.q_a_ln, 0, true);
    load_f16_from(p + ".self_attn.kv_a_layernorm.weight", mtp.mla.kv_a_ln, 0, true);
    load_f16_from(p + ".self_attn.kv_b_proj.weight", mtp.mla.kv_b, 0, false);
    load_group(p + ".self_attn.indexer.wq_b", mtp.idx.wq_b, 0);
    load_f16_from(p + ".self_attn.indexer.wk.weight", mtp.idx.wk, 0, false);
    load_f16_from(p + ".self_attn.indexer.weights_proj.weight", mtp.idx.wproj, 0, false);
    load_f16_from(p + ".self_attn.indexer.k_norm.weight", mtp.idx.knorm_w, 0, false);
    load_f16_from(p + ".self_attn.indexer.k_norm.bias", mtp.idx.knorm_b, 0, false);
    load_f16_from(p + ".self_attn.indexer.index_kpool_compress_gate", mtp.idx.kpool_gate, 0, false);
    load_f32_from(p + ".self_attn.indexer.index_kpool_compress_ape", mtp.idx.kpool_ape, 0);
    // router + shared
    auto* gw = T(p + ".mlp.gate.weight");
    auto* gb = T(p + ".mlp.gate.e_score_correction_bias");
    if (gw) { mtp.moe_w.router_gate = (const half*)(ram_only ? nullptr : A(gw->bytes, 1)); j(gw, (void*)mtp.moe_w.router_gate, 1, Job::RAW); }
    if (gb) { mtp.moe_w.router_bias = (const half*)(ram_only ? nullptr : A(gb->elems * 2, 1)); j(gb, (void*)mtp.moe_w.router_bias, 1, Job::ROUTER_BIAS); }
    load_group(p + ".mlp.shared_experts.gate_proj", mtp.moe_w.shared[0], 0);
    load_group(p + ".mlp.shared_experts.up_proj",  mtp.moe_w.shared[1], 0);
    load_group(p + ".mlp.shared_experts.down_proj", mtp.moe_w.shared[2], 0);
    mtp.moe_w.arena_layer = arena_slot_base[l];
    load_experts(l, arena_slot_base[l]);
    if (!ram_only && (mtp_enabled() || getenv("HELIOS_MTP_EXPERTS"))) {
      // Full preload of the MTP draft layer's experts to GPU1. Off by default: the draft layer is
      // not wired into the decode loop, and this 1.74GB otherwise comes straight out of the expert
      // slot pool (~288 more slots, ~10% more residency, and residency is what decode pays for).
      size_t bytes = (size_t)cfg.n_expert * slay.stride;
      mtp.experts_gpu1 = A(bytes, 1);
      Job jb; jb.dst = mtp.experts_gpu1; jb.src_ram = slab(l, 0); jb.bytes = bytes;
      jb.device = 1; jb.kind = Job::A2G; jb.name = "mtp_experts_preload";
      jobs.push_back(jb);
    }
    load_f16_from(p + ".input_layernorm.weight", mtp.input_ln, 0, true);
    load_f16_from(p + ".post_attention_layernorm.weight", mtp.post_ln, 0, true);
    load_group(p + ".eh_proj", mtp.eh_proj, 0);
    load_f16_from(p + ".enorm.weight", mtp.enorm, 0, true);
    load_f16_from(p + ".hnorm.weight", mtp.hnorm, 0, true);
    load_f16_from(p + ".shared_head.norm.weight", mtp.shared_head_norm, 0, true);
  }

  bool load(const std::string& dir, bool ro, bool verbose) {
    ram_only = ro;
    try { shards.load_dir(dir); }
    catch (const std::exception& e) { fprintf(stderr, "[model] shard load failed: %s\n", e.what()); return false; }
    if (!parse_config(dir)) return false;

    // expert K bits from a sample expert trellis (2-bit expected)
    auto* probe = shards.find(L(3) + ".mlp.experts.0.gate_proj.trellis");
    int K = probe ? (int)(probe->shape[2] / 16) : 2;
    slay = make_slab_layout(cfg.hidden, cfg.moe_inter, K);
    slot_stride = slay.stride;
    slab_off.assign(slay.off, slay.off + 12);

    // arena slot bases: sparse layers (3..44) + MTP
    arena_slot_base.assign(cfg.n_layers + 1, -1);
    int seq = 0;
    for (int l = 0; l < cfg.n_layers; l++) if (cfg.moe[l]) arena_slot_base[l] = 288 * seq++;
    if (cfg.has_mtp) arena_slot_base[cfg.mtp_layer] = 288 * seq++;
    arena_layers = seq;
    size_t arena_bytes = (size_t)arena_layers * cfg.n_expert * slay.stride;
    arena = (char*)mmap(nullptr, arena_bytes, PROT_READ|PROT_WRITE,
                        MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (arena == MAP_FAILED) { fprintf(stderr, "[model] arena mmap %zu GB failed\n", arena_bytes >> 30); return false; }
    ram_bytes = arena_bytes;
    // Pin the arena: pageable host memory caps expert H2D at ~8 GB/s, pinned runs at link speed.
    // Registration locks the (already resident) pages; failure is non-fatal (we keep pageable).
    if (getenv("HELIOS_NO_PIN") == nullptr) {
      auto t_pin = std::chrono::steady_clock::now();
      cudaError_t pe = cudaHostRegister(arena, arena_bytes, cudaHostRegisterDefault);
      double pin_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pin).count();
      if (pe != cudaSuccess)
        fprintf(stderr, "[model] arena pin failed (%s) - continuing pageable\n", cudaGetErrorString(pe));
      else
        printf("[model] arena pinned in %.1f s\n", pin_ms / 1000.0);
    }

    if (!ram_only) {
      embed = A((size_t)cfg.vocab * cfg.hidden * 2, 0);           // bf16 kept
      {
        auto* te = T("model.language_model.embed_tokens.weight");
        if (te) j(te, (void*)embed, 0, Job::RAW);
      }
      load_f16_from("model.language_model.norm.weight", final_norm, 0, true);
      load_group("lm_head", lm_head, 0);
    }

    layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; l++) {
      Layer& ly = layers[l];
      ly.index = l; ly.kind = cfg.attn[l]; ly.moe = cfg.moe[l];
      load_attn(l, ly);
      load_mlp(l, ly);
      load_hc(l, ly);
      if (ly.moe) load_experts(l, arena_slot_base[l]);
    }
    load_mtp();

    if (!ram_only) {
      // GPU1 slot pool
      { int pi = Engine::instance().gpu(1).phys_idx(); cudaSetDevice(pi); }
      size_t free_v = 0, tot_v = 0;
      cudaMemGetInfo(&free_v, &tot_v);
      cudaSetDevice(Engine::instance().gpu(0).phys_idx());   // restore: allocations must not leak
                                                             // the current device to later stages
      free_v -= (size_t)Engine::instance().gpu(1).pool_used();
      free_v = free_v > (size_t)Engine::instance().gpu(1).pool_used() ? free_v : 0;
      size_t avail = free_v > (4ull<<30) ? free_v - (4ull<<30) : 0;
      int slots = (int)(avail / slay.stride);
      if (const char* ev = getenv("HELIOS_SLOTS")) slots = atoi(ev);
      n_slots = std::max(slots, 0);
      slot_pool = A((size_t)n_slots * slay.stride, 1, 4096);
    }

    // sort jobs: big GPU transfers first, then arena reads
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
      int ka = a.device == -1 ? 1 : (a.bytes > (16u<<20) ? 0 : 2);
      int kb = b.device == -1 ? 1 : (b.bytes > (16u<<20) ? 0 : 2);
      return std::make_tuple(ka, -a.bytes) < std::make_tuple(kb, -b.bytes);
    });

    if (verbose) printf("[model] jobs=%zu arena=%.1fGB gpu0=%.2fGB gpu1=%.2fGB slots=%d\n",
                        jobs.size(), (double)ram_bytes/(1<<30), (double)gpu0_bytes/(1<<30),
                        (double)gpu1_bytes/(1<<30), n_slots);

    run_jobs(verbose);
    verify_slabs(verbose);
    if (!ram_only) verify_gpu_tensors(verbose);

    if (missing) fprintf(stderr, "[model] WARNING: %d tensors missing\n", missing);
    return true;
  }

  // spot-check arena contents against shard bytes
  void verify_slabs(bool verbose) {
    std::mt19937 rng(1234);
    std::atomic<int> checked{0}, bad{0};
    auto worker = [&]() {
      void* tmp = malloc(4u << 20);
      for (int it = 0; it < 16; it++) {
        int l = 3 + (int)(rng() % (cfg.n_layers - 3 + (cfg.has_mtp ? 2 : 1)));
        if (l >= cfg.n_layers) l = cfg.mtp_layer;
        if (arena_slot_base[l] < 0) continue;
        int e = (int)(rng() % cfg.n_expert);
        int q = (int)(rng() % 4);       // piece within gate/up/down triple
        int t = (int)(rng() % 3);
        static const char* tn[3] = {"gate_proj", "up_proj", "down_proj"};
        static const char* pn[4] = {".trellis", ".suh", ".svh", ".mul1"};
        std::string nm = L(l) + ".mlp.experts." + std::to_string(e) + "." + tn[t] + pn[q];
        const TensorInfo* ti = shards.find(nm);
        if (!ti) continue;
        shards.read(*ti, tmp);
        const char* sl = slab(l, e) + slay.off[t * 4 + q];
        if (memcmp(tmp, sl, ti->bytes) != 0) { fprintf(stderr, "[verify] MISMATCH %s\n", nm.c_str()); bad++; }
        checked++;
      }
      free(tmp);
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 16; i++) th.emplace_back(worker);
    for (auto& x : th) x.join();
    printf("[verify] %d slab pieces checked, %d mismatches\n", checked.load(), bad.load());
    if (bad) fprintf(stderr, "[model] ARENA VERIFICATION FAILED\n");
  }

  // Sample the GPU-resident trunk tensors and compare with the shard bytes (mirrors verify_slabs
  // for the device side). Catches placement/transfer bugs before any kernel runs.
  void verify_gpu_tensors(bool verbose) {
    struct Item { const std::string name; void* dev; size_t bytes; };
    std::vector<Item> items;
    auto addg = [&](const std::string& n, const Group& g) {
      if (g.trellis) items.push_back({n + ".trellis", g.trellis, (size_t)g.out * g.in * g.K / 8 * 2});
      if (g.suh) items.push_back({n + ".suh", (void*)g.suh, (size_t)g.in * 2});
      if (g.svh) items.push_back({n + ".svh", (void*)g.svh, (size_t)g.out * 2});
    };
    for (int l = 0; l < cfg.n_layers; l++) {
      Layer& L = layers[l];
      std::string p2 = ("model.language_model.layers." + std::to_string(l));
      if (L.kind == KDA) {
        addg(p2 + ".linear_attn.qkv_proj", L.kda.qkv);
        addg(p2 + ".linear_attn.o_proj", L.kda.o_proj);
      } else {
        addg(p2 + ".self_attn.q_a_proj", L.mla.q_a);
        addg(p2 + ".self_attn.q_b_proj", L.mla.q_b);
        addg(p2 + ".self_attn.kv_a_proj_with_mqa", L.mla.kv_a);
        addg(p2 + ".self_attn.o_proj", L.mla.o_proj);
        addg(p2 + ".indexer.wq_b", L.idx.wq_b);
      }
      if (L.moe) { addg(p2 + ".mlp.shared_experts.gate_proj", L.moe_w.shared[0]); }
      else { addg(p2 + ".mlp.gate_proj", L.dense.gate); }
    }
    addg("lm_head", lm_head);
    int bad = 0, n = 0;
    std::vector<char> buf;
    for (auto& it : items) {
      const TensorInfo* ti = shards.find(it.name);
      if (!ti) continue;
      size_t bytes = std::min(it.bytes, ti->bytes);
      if (bytes == 0) continue;
      buf.resize(bytes);
      int dev = -1;
      for (int d = 0; d < N_GPU; d++) {
        cudaPointerAttributes at{};
        if (cudaPointerGetAttributes(&at, it.dev) == cudaSuccess && at.type == cudaMemoryTypeDevice) {
          dev = Engine::instance().gpu(d).phys_idx();
          if (at.device == dev) break;
        }
      }
      if (dev < 0) { fprintf(stderr, "[verify-gpu] unresolved ptr for %s\n", it.name.c_str()); bad++; continue; }
      cudaSetDevice(dev);
      cudaMemcpy(buf.data(), it.dev, bytes, cudaMemcpyDeviceToHost);
      // compare against the shard bytes
      std::vector<char> ref(bytes);
      shards.read(*ti, ref.data());
      if (memcmp(buf.data(), ref.data(), bytes) != 0) {
        fprintf(stderr, "[verify-gpu] MISMATCH %s (%zu bytes)\n", it.name.c_str(), bytes);
        bad++;
      }
      n++;
    }
    printf("[verify-gpu] %d device tensors checked, %d mismatches\n", n, bad);
  }

  void run_jobs(bool verbose);
};

// ---------- parallel runner ----------
struct Ctx {
  char* pin[2] = {nullptr, nullptr};
  cudaEvent_t ev[2] = {nullptr, nullptr};
  void* tmp = nullptr;
  size_t pin_cap = 64u << 20, tmp_cap = 64u << 20;
  bool used[2] = {false, false};
};

void Loader::run_jobs(bool verbose) {
  std::atomic<size_t> next{0};
  int nthreads = 16;
  auto t0 = std::chrono::steady_clock::now();
  auto worker = [&]() {
    Ctx cx;
    cx.tmp = malloc(cx.tmp_cap);
    if (!ram_only) {
      cudaSetDevice(Engine::instance().gpu(0).phys_idx());
      cudaMallocHost(&cx.pin[0], cx.pin_cap); cudaMallocHost(&cx.pin[1], cx.pin_cap);
      cudaEventCreateWithFlags(&cx.ev[0], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&cx.ev[1], cudaEventDisableTiming);
    }
    for (;;) {
      size_t i = next.fetch_add(1);
      if (i >= jobs.size()) break;
      const Job& jb = jobs[i];
      if (jb.device >= 0) cudaSetDevice(Engine::instance().gpu(jb.device).phys_idx());
      switch (jb.kind) {
        case Job::MUL1: shards.read(*jb.ti, jb.dst); break;
        case Job::A2G:
          cudaMemcpy(jb.dst, jb.src_ram, jb.bytes, cudaMemcpyHostToDevice);
          break;
        case Job::RAW:
          if (jb.device < 0) { shards.read(*jb.ti, jb.dst); break; }
          if (jb.bytes <= (16u << 20)) {
            shards.read(*jb.ti, cx.tmp);
            cudaMemcpyAsync(jb.dst, cx.tmp, jb.bytes, cudaMemcpyHostToDevice,
                            Engine::instance().gpu(jb.device).stream(2));
            cudaStreamSynchronize(Engine::instance().gpu(jb.device).stream(2));
          } else {
            size_t done = 0, slot = 0;
            while (done < jb.bytes) {
              size_t chunk = std::min(cx.pin_cap, jb.bytes - done);
              if (cx.used[slot]) while (cudaEventQuery(cx.ev[slot]) == cudaErrorNotReady) {}
              int fd = shards.shard_fd(jb.ti->shard);
              uint64_t pos = shards.data_start(jb.ti->shard) + jb.ti->offset + done;
              size_t rd = 0;
              while (rd < chunk) {
                ssize_t n = ::pread(fd, cx.pin[slot] + rd, chunk - rd, pos + rd);
                if (n <= 0) throw std::runtime_error("pread failed " + jb.name);
                rd += (size_t)n;
              }
              cudaStream_t st = Engine::instance().gpu(jb.device).stream(2 + slot);
              HELIOS_CUDA_CHECK(cudaMemcpyAsync((char*)jb.dst + done, cx.pin[slot], chunk, cudaMemcpyHostToDevice, st));
              HELIOS_CUDA_CHECK(cudaEventRecord(cx.ev[slot], st));
              cx.used[slot] = true;
              done += chunk; slot ^= 1;
            }
          }
          break;
        case Job::BF16_F16: case Job::F32_F16: {
          shards.read(*jb.ti, cx.tmp);
          if (jb.kind == Job::BF16_F16) cvt_bf16_f16(cx.tmp, (half*)cx.tmp, jb.elems);
          else cvt_f32_f16(cx.tmp, (half*)cx.tmp, jb.elems);
          cudaMemcpyAsync(jb.dst, cx.tmp, jb.elems * 2, cudaMemcpyHostToDevice,
                          Engine::instance().gpu(jb.device).stream(2));
          cudaStreamSynchronize(Engine::instance().gpu(jb.device).stream(2));
          break;
        }
        case Job::ROUTER_BIAS: {
          float* f = (float*)cx.tmp;
          shards.read(*jb.ti, f);
          double sum = 0; for (size_t e = 0; e < jb.elems; e++) sum += f[e];
          float mean = (float)(sum / jb.elems);
          half* h = (half*)cx.pin[0];
          for (size_t e = 0; e < jb.elems; e++) h[e] = __float2half_rn(f[e] - mean);
          cudaMemcpyAsync(jb.dst, h, jb.elems * 2, cudaMemcpyHostToDevice,
                          Engine::instance().gpu(jb.device).stream(2));
          cudaStreamSynchronize(Engine::instance().gpu(jb.device).stream(2));
          break;
        }
      }
    }
    if (!ram_only) { cudaDeviceSynchronize(); cudaFreeHost(cx.pin[0]); cudaFreeHost(cx.pin[1]); cudaEventDestroy(cx.ev[0]); cudaEventDestroy(cx.ev[1]); }
    free(cx.tmp);
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
  for (auto& t : th) t.join();
  if (!ram_only) { cudaSetDevice(Engine::instance().gpu(0).phys_idx()); cudaDeviceSynchronize();
                   cudaSetDevice(Engine::instance().gpu(1).phys_idx()); cudaDeviceSynchronize(); }
  if (verbose) {
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    printf("[model] load time %.1f s (%.1f GB/s)\n", dt / 1000.0, (double)(ram_bytes+gpu0_bytes+gpu1_bytes)/(dt/1000.0)/(1<<30));
  }
}

bool Model::load(const std::string& dir, bool ram_only, bool verbose) {
  Loader ld;
  bool ok = ld.load(dir, ram_only, verbose);
  if (!ok) return false;
  *static_cast<Model*>(this) = std::move(ld);
  return true;
}

}  // namespace helios