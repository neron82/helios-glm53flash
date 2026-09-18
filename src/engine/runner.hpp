#pragma once
// Helios runner: layer-major forward pass over the Model (KDA/MLA attention, dense stem and
// sparse MoE FFN with GPU1 expert slot streaming), plus prefill chunking and sampling.
#include "core/model.hpp"
#include "core/device.hpp"
#include "engine/cache.hpp"
#include "engine/slotmgr.hpp"
#include "engine/sampler.hpp"
#include "tokenizer/tokenizer.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace helios {

struct MoEPlan {
  int concurrency = 8;
  int max_tokens_per_expert = 16;
};

class Runner {
public:
  // Allocates workspaces on both GPUs. max_chunk = prefill chunk size (rows per forward).
  bool init(Model& m, Cache& c, SlotMgr& sm, Tokenizer* tk, int max_chunk = 256);

  void reset();                                     // fresh sequence: clear caches/states
  void prefill(const std::vector<int>& ids);        // process prompt tokens from pos 0
  void decode(const std::vector<int>& ids);         // process 1..k tokens (extends sequence)
  const half* logits_dev() const;                   // logits of the last decoded token
  int pos() const { return pos_; }
  int context_cap() const { return c_ ? c_->cap() : 0; }

  // Full generation loop (prompt already tokenized). on_token may return false to stop.
  std::vector<int> generate(const std::vector<int>& prompt, const GenParams& p,
                            const std::function<bool(int)>& on_token = nullptr);

  struct Timings {
    double prefill_ms = 0, decode_ms = 0;
    int prefill_tokens = 0, decode_tokens = 0;
  };
  const Timings& timings() const { return tm_; }

  // Debug: run the first `upto` layers on `ids` and dump the 4 mHC streams [4,n,4096] fp32
  // (and optionally the attn/ffn sublayer outputs of the last executed layer) to a file.
  bool bench_gemm(int max_m);
  bool bench_arena_copy(int n_slabs);
  bool dump_layers(const std::vector<int>& ids, int upto, const std::string& path,
                   const std::string& sub_path = "");

private:
  void run_chunk(int n, int pos, bool prefill);
  // One transformer layer over `n` tokens whose mHC streams live at `streams` (R,H,D) contiguous.
  void layer_step(Layer& L, int n, int pos, float* streams);
  void kda_layer(Layer& L, int n, int pos);
  void mla_layer(Layer& L, int n, int pos);
  void ffn_layer(Layer& L, int n, int pos);
  void dense_mlp(Layer& L, int n);
  void moe_ffn(Layer& L, int n);
  void final_head(int n);   // n = rows in c_->xh; logits come from the LAST row

  // ---- MTP speculative decoding ----
  // The MTP head is a PLAIN residual block (no mHC) fed by eh_proj(cat(hnorm(embed), enorm(trunk_h))),
  // so it reuses mla_layer/ffn_layer on a synthetic Layer whose mla_ord is the draft cache slot.
  // Its 288 experts are fully resident at m_->mtp.experts_gpu1, bypassing the slot manager entirely.
  bool mtp_ready_ = false;
  Layer mtp_layer_;
  half* mtp_ckpt_ = nullptr;       // KDA conv+recurrent state checkpoint (fp16-free, raw bytes)
  half* mtp_hid_ = nullptr;        // [4096] fp16 draft hidden state (never aliases trunk buffers)
  half* logits_multi_ = nullptr;   // [(k+1), vocab] fp16 verify logits
  half* host_logits_multi_ = nullptr;  // pinned host mirror
  float* logits32_multi_ = nullptr;// [(k+1), vocab] fp32
  // KDA state capture for draft verification: the recurrent state must be restored to the round's
  // start and advanced over exactly the ACCEPTED prefix. The KDA layer reads its input from c_->xa,
  // so the inputs for the verified rows are captured too and replayed.
  bool capture_ = false;           // kda_layer checkpoints state + inputs while set
  int n_capture_ = 0;              // rows captured per KDA layer
  half* kda_in_ = nullptr;         // [n_kda][maxcap][4096] fp16 captured layer inputs
  size_t kda_in_stride_ = 0;
  size_t kda_state_bytes() const;
  void kda_capture_begin(int n);   // snapshot every KDA layer's state + declare the row count
  void kda_rollback(int n, int pos);// restore snapshots, replay n accepted rows per layer
  void mtp_build();
  int mtp_step(int tok, int pos);  // one draft token; writes the draft KV at `pos`
  void final_head_multi(int n);    // logits for ALL n rows (verify), into logits_multi_
  std::vector<int> generate_mtp(const std::vector<int>& prompt, const GenParams& p,
                                const std::function<bool(int)>& on_token);

  Model* m_ = nullptr;
  Cache* c_ = nullptr;
  SlotMgr* sm_ = nullptr;
  Tokenizer* tk_ = nullptr;
  int max_chunk_ = 256;
  int pos_ = 0;
  int last_rows_ = 0;              // rows produced by the most recent run_chunk (seed row for MTP)

  // GPU0 scratch
  struct {
    int* tokens = nullptr;       // [max] token ids
    half* logits16 = nullptr;    // [vocab] fp16 logits of the last row
    float* logits32 = nullptr;   // [vocab] fp32 logits (gemv output)
    float* qkv = nullptr;        // [max,24576] f32
    void *mqkv_bf16 = nullptr, *conv_out_bf16 = nullptr;   // [24576,max] / [max,24576] bf16
    void* kda_rec_bf16 = nullptr;// [max,8192] bf16
    half* kda_norm = nullptr;    // [max,8192] fp16
    half* f_mid = nullptr;       // [max,128] fp16
    float* f_out = nullptr;      // [max,8192] f32
    float* b_out = nullptr;      // [max,64] f32
    half* g_mid = nullptr;       // [max,128] fp16
    float* g_out = nullptr;      // [max,8192] f32
    float* kda_g = nullptr;      // [max,8192] f32
    void* beta_bf16 = nullptr;   // [max,64] bf16
    half* qlat_a = nullptr;      // [max,1536] fp16 (q_a latent after norm)
    half* q_nope = nullptr;      // [max,64,256] fp16
    half* q_lat = nullptr;       // [max,64,512] fp16
    half* lat_out = nullptr;     // [max,64,512] fp16
    half* o_abs = nullptr;       // [max,16384] fp16
    half* ckv_new = nullptr;     // [max,512] fp16 (kv_a latent)
    half* idx_q = nullptr;       // [max,32,128] fp16
    half* idx_k = nullptr;       // [max,128] fp16
    half* idx_gate = nullptr;    // [max,128] fp16
    half* idx_w = nullptr;       // [max,32] fp16
    half* scores = nullptr;      // [max, npools]
    int* topk_idx = nullptr;     // [max,512] i32
    int* raw_idx = nullptr;      // [max,2052] i32
    int* qpos = nullptr;         // [max] i32
    float* attn_out = nullptr;   // [max,4096] f32
    half* attn_out16 = nullptr;  // [max,4096] fp16
    float* ffn_out = nullptr;    // [max,4096] f32
    half* ffn_out16 = nullptr;   // [max,4096] fp16
    float* mlp_gate = nullptr;   // [max,12288] f32 (dense stem / shared expert)
    float* mlp_up = nullptr;     // [max,12288] f32
    half* mlp_act = nullptr;     // [max,12288] fp16
    float* mlp_down = nullptr;   // [max,4096] f32
    float* shared_out = nullptr; // [max,4096] f32
    half* a_had = nullptr;       // gemm input-transform scratch [max,24576]
    half* a_had2 = nullptr;      // dedicated scratch for the MLA projections
    float* hc_partials = nullptr;
    float* hc_post = nullptr, *hc_comb = nullptr;
    half* hc_collapsed = nullptr;
    float* rms_dummy = nullptr;
    half* norm_out = nullptr;    // [max,4096] fp16
    half* mtp_in = nullptr;      // [8192] fp16: cat(hnorm(embed), enorm(hidden)) for eh_proj
    void* conv_w_bf16 = nullptr; // per-KDA-layer bf16 copies, [n_kda][24576*4]
    void* dt_bias_bf16 = nullptr;
    void* onorm_bf16 = nullptr;
    int* helper_i32 = nullptr;
    char* pin_h = nullptr;       // pinned host staging (GPU0 <-> GPU1)
    int64_t* host_ids = nullptr;     // pinned: topk ids of the MoE step
    half* host_logits = nullptr;     // pinned: logits for the sampler
  } w0;

  // GPU1 scratch
  struct {
    half* x = nullptr;             // [max,4096] fp16 input to MoE
    half* router_scores = nullptr; // [max,288] fp16
    int64_t* topk_i64 = nullptr;   // [max,8]
    half* topk_w = nullptr;        // [max,8]
    float* y = nullptr;            // [max,4096] f32 accumulated output
    float* back = nullptr;         // host-staged return buffer (pinned)
    void* moe_tg = nullptr, *moe_tu = nullptr, *moe_ig = nullptr, *moe_iu = nullptr;
    int64_t* ec = nullptr;         // [289]
    int64_t* tsorted = nullptr;    // [max*8]
    half* wsorted = nullptr;       // [max*8]
    int64_t* perm_ws = nullptr;    // [3*(288+2)]
    void** tables = nullptr;       // [9*288] device pointer tables
    char* stage = nullptr;         // spare GPU1 scratch
    // dense-MoE scratch: per-expert gather -> three plain gemms -> weighted scatter-add
    half* dense_x = nullptr;       // [max,4096] gathered rows
    float* dense_g = nullptr;      // [max,2048] gate
    float* dense_u = nullptr;      // [max,2048] up
    half* dense_a = nullptr;       // [max,2048] swiglu output
    float* dense_d = nullptr;      // [max,4096] down
    half* dense_had = nullptr;     // Hadamard input scratch [max,4096]
    float* host_y = nullptr;       // pinned: MoE output staging
  } w1;

  size_t moe_off_[12] = {};      // slab piece offsets from the loader
  int routed_mul1_ = 0;          // mul1 word shared by all routed experts
  MoEPlan moe_plan_;
  Timings tm_;
};

// OpenAI-compatible HTTP front end (server.cpp). default_max_tokens <= 0 keeps the built-in default.
int run_server(Runner& runner, Tokenizer& tk, const std::string& host, int port, int n_threads = 4,
               const std::string& api_key = std::string(), int default_max_tokens = 0);

}  // namespace helios