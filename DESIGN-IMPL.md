# Helios implementation blueprint (supersedes DESIGN.md where they differ)

This blueprint describes the original 2.05 bpw checkpoint and historical implementation plan.
For the mixed quant and optional image frontend, see [README.md](README.md).

## Model facts (verified against config.json + shard headers)
- 45 layers: KDA at linear_attn_config.kda_layers (34), sparse-MLA at [3,7,…,43] (11).
  All 11 MLA layers indexer_mode="full" (indexer_types all full, len 45).
- hidden 4096, vocab 154880 (embed bf16 UNQUANTIZED 1.27GB), lm_head 5-bit (head_bits),
  tie=false. Dense stem layers 0-2 inter 12288 @3-bit (trellis K=48). MoE 288+1 shared,
  top-8, moe_inter 2048, experts @2-bit (K=32), shared @4-bit (K=64), routed_scale 2.5,
  swiglu_limit 10, sigmoid noaux_tc (n_group=1 → nogroup kernel), bias f32[288] centered→fp16.
- KDA: 64 heads × hd 128; qkv fused [24576,4096] @4-bit; o_proj [4096,8192] @4-bit;
  conv1d bf16 [24576,1,4]; b_proj fp16 [64,4096]; f_a [128,4096] f_b [8192,128] fp16
  (rank 128); g_a/g_b same shape (output gate); dt_bias f32 [8192] per-channel; A_log f32 [64];
  o_norm bf16 [128] shared across heads; gate = -5·sigmoid(exp(A_log)·(f+dt_bias));
  beta = sigmoid(b·x); gated norm = rms(o)·w·sigmoid(gate_g).
- MLA: q_a [1536,4096]@4b + ln bf16[1536] + q_b [16384,1536]@4b (NoPE, qk_nope=256, v=256);
  kv_a [512,4096]@4b + ln bf16[512]; kv_b fp16 [32768,512] UNQUANTIZED → w_uk_flat (512,64×256)
  + w_uv_flat (512,64×256); o_proj [4096,16384]@4b. scale = 256^-0.5. Absorbed: q_lat=q_nope@W_uk[h];
  scores=q_lat·ckv; o_lat=probs·ckv; o=o_lat@W_uv[h].
- Indexer (all MLA layers): wq_b [4096,1536]@4b (from q_a latent), wk fp16 [128,4096],
  k_norm LN fp16 (w+b), weights_proj fp16 [32,4096], gate fp16 [128,4096], APE f32 [4,128].
  kpool P=4: pool key = Σ softmax(gate+APE)(dim=P)·k per CHANNEL; score = Σ_h w[h]·relu(q·k_pool·128^-0.5)·32^-0.5;
  causal: pool p visible iff p·4+3 ≤ q_pos; topk 512 pools → expand ×4 raw + tail (pos+1)%4 raw tokens.
  Plane per token: [k(128)||gate(128)] fp16; pool plane 128 fp16 per pool.
- mHC every site: fn f32 [24,16384], base f32[24], scale f32[3] (per-part scalar).
  mix: flat=RMSnorm(streams.flatten) (eps 1e-5); w=flat@fnᵀ; pre=sigmoid(pre_w·s+b)+1e-6;
  post=2·sigmoid(...); comb=(w·s+b) reshape 4×4 → softmax(dim=-1)+eps → colnorm+eps →
  19× (rownorm+eps, colnorm+eps); collapsed=Σ pre·streams (→fp16 to sublayer).
  apply_: x[h,d] = post[h]·y[d] + Σ_k comb[k,h]·x[k,d] in-place. Head: mean over 4 streams.
  ExpandStreams: embed broadcast ×4 fp32.
- MTP: 1 layer (num_nextn_predict_layers), tensors in final shard (pending download):
  enorm/hnorm/eh_proj [4096, 8192]/shared_head.norm + layer-45 attn+MoE (full-attn type).
- EXL3 decode: per tensor group {trellis (k/16,n/16,16·K int16), suh f16[k], svh f16[n], mul1 i32}.
  K = trellis.shape[-1]/16 = bits. Blocks of 256 values, tensor-core lane order, MSB-first
  bitstream; decode = 16-bit window extract + mul1 codebook: x*=0x83DCD12D; dp4a(x,0x01010101,0x6400);
  half(sum)·0.00677 + (-10.39). Weights Hadamard-rotated: GEMM applies A_had (128-group FWHT)
  prologue + svh post-scale + output hadamard epilogue. Reuse exllamav3 kernels verbatim.

## Hardware
GPU0 (07:00.0) PCIe 8GT/s x4 ≈3.4GB/s | GPU1 (2b:00.0) 16GT/s x16 ≈25GB/s | RAM 128GB DDR4-3200
27GB/s read | 3700X 8C/16T AVX2 | sm_86 ×2, no NVLink (PHB P2P weak — route via host pinned).

## Execution layout (functional split)
GPU0 TRUNK: embed (fp16 1.27GB), all attention weights (KDA qkv/o/gate projections, MLA
q/kv/o + kv_b flats, indexer small weights), dense stem, shared experts, mHC tensors (fp16),
norms, lm_head @5bit, KV cache (fp16 latent 2.75GB@250k + idx planes 1.4GB + pool 0.18GB +
KDA states fp32 ~0.2GB(+spec history)), stream stack fp32 workspaces. Est. 8-10GB @250k ctx.
GPU1 MOE: router gate+bias, grouped expert GEMM kernels, expert slot pools (~17GB:
~60% heat-pinned static + 40% dynamic LRU), streaming scheduler + pinned DMA rings,
MTP draft MoE. CPU: RAM expert arena 78GB (full pool, contiguous per-expert slabs),
tokenizer, HTTP, staging pinned rings.
Per-layer exchange: hidden (R×4096 fp32 ≤ 32KB/token) via pinned bounce (P2P disabled).

## Decode cost model (per token, bs=1)
GPU0: KDA 34×recurrent(64h×128²) ~1.2ms + MLA 11×(indexer 32×128×visible + topk attn 64×512×2048)
~0.8ms + mHC 90 sites ~0.25ms + stem+shared+projections+embed+head ~1.2ms → ~3.5ms ≈ 28 tok/s.
GPU1: 8 experts × 42 layers: zero-hit 2.15GB/token @27GB/s = 12.6 tok/s; with slot hit-rate
h, effective bytes (1−h)·2.15GB → h=0.6 → ~6 tok/s?? NO: 0.4·2.15=0.86GB → 31 tok/s GPU1-bound.
System ceiling ≈ min(GPU0, GPU1-stream) ≈ 25-30 tok/s at h≥0.6; MTP ×4 draft → 40-60 tok/s target.
Levers: hit rate (heat+pins+predictor), MTP, fp8 latent cache (GPU0 traffic), static experts on
GPU0 co-compute (v2 if GPU0 becomes bottleneck).

## Prefill strategy (KEY)
Layer-major full-residency streaming: for each sparse layer, stream ALL 288 experts (1.76GB,
~65ms @27GB/s) into GPU1 slot pool ONCE, then compute the entire prompt's sub-chunks (256-512)
against resident experts; evict; next layer. Trunk layers on GPU0 process sub-chunks in lockstep.
Result: expert streaming cost is O(layers), not O(tokens). Prefill becomes compute-bound:
~8k-10k tok/s projected for long prompts (indexer+KDA chunked dominate). Dense stem/attention
normal chunked prefill.

## Kernel plan
PORTED from exllamav3 ext (strip torch, raw-pointer launchers, stream param):
  quant/{exl3_dq,codebook,pack,reconstruct,hadamard*,exl3_gemm*,exl3_gemv*,exl3_moe*,util,ptx,reduction,kernel_map,inner},
  norm.cu (rms/gated_rms/layer_norm), hc_mix.cu (mix_partials/finalize/apply/head),
  routing.cu (ds3_nogroup sigmoid, both bsz paths), dsa_topk.cu (radix topk),
  gdn.cu decode kernels (conv_update, kda_gate_op, recurrent channelwise, lowrank gemv),
  activation.cu (swiglu clamped), add.cu.
CUSTOM-WRITE (CUDA, sm_86):
  MLA decode dense flash (absorbed, D=512 latent, NoPE) + sparse gather flash (topk 2048 rows),
  MLA prefill flash causal chunked (absorb-form), indexer scoring kernel (kpool plane, tiled
  topk merge), kpool plane-append/update kernels, KDA CHUNKED PREFILL (FLA-style UT transform,
  chunk 64, channelwise decay — hardest kernel), expert slot manager + pointer-table builder,
  streaming scheduler (pinned ring DMA + LRU + heat), sampler, tokenizer (tokenizer.json parser).
SKIP: rope (NoPE), q_cache (fp16 cache), attention.cu, parallel/* (no TP), ngram, ple, softcap.

## Numerics
Streams fp32; activations fp16; KDA states fp32; cache fp16; expert GEMM fp16 (EXL3);
router fp16 logits + f32-centered bias fp16; sinkhorn fp32; swiglu limit ±10 per colibri/GLM ref.

## Sequencing
1. Kernel ports (agents) + device.cpp/loader/model-builder (me) in parallel.
2. Custom attention kernels + KDA chunked prefill.
3. Engine loop decode+prefill, HTTP, tokenizer.
4. Integration vs downloaded model; RAM residency load test; GPU load test w/ server stop LAST.
5. Tuning: hit rates, MTP, fp8 cache option, GPU0 static experts (v2).