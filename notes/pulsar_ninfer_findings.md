# Pulsar + ninfer findings — condensed for Helios (full: history://PulsarNinferScout)

## Adopted decisions
- EXPERT COMPUTE GPU-ONLY: pulsar q* CPU-lane assumes ~3-4 TOPS AVX2-i8mm host dots;
  3700X lacks VNNP; EXL3 trellis decode on CPU adds heavy overhead; colibri #1119 says
  AVX2 kernels bind before I/O on Zen2/3 → CPU lane rejected. q* = all-fill on GPU1.
- MTP economics (pulsar measured): speculation LOSES on streaming decode unless draft
  experts resident → make MTP layer's 288 experts FULLY VRAM-resident on GPU1 (1.74GB).
  Accept rate ninfer MTP3 on 3090: 57-61%.
- Cross-layer prefetch: pulsar runs NEXT layer's router on THIS layer's ffn input
  (background thread, own stream) + ping-pong staging + async H2D side stream. Implement
  as prefetch thread on GPU1 + RAM reads. Instrument pred_hits/pred_total.
- Heat/census: persist per-(layer,expert) counts sidecar (.helios.warm, 24B LE records),
  merge by PER-RUN MAX (running sums ossify — measured poison case), rank whole
  gate/up/down triples, heat-gate admission on prefill, LRU-force on decode. Sink/shared
  never in tier (always resident anyway).
- ExpertPtrs kernel contract: per-launch explicit pointer arrays, NULL = inactive.
- CUDA-graph decode (ninfer lifecycle: capture→instantiate→execUpdate→upload) = later
  optimization once correct; graph-stable workspaces from day 1 (fixed addresses).
- L2 persistence window (cudaLimitPersistingL2CacheSize + accessPolicyWindow) on hot
  expert slots + KV — neither repo uses it; free headroom for us.
- Split-K grids sized to 82 SMs (3090, NOT 84 like 3090 Ti); thread-local device cache
  (colibri lesson); transfer_stream separate from compute stream per device.
- Spec-verify needs recurrent-state rollback: ninfer ReplaySSM/StateImage host+device
  checkpoint slots — KDA states must survive rejected drafts (checkpoint per spec step).
- KV compression option later: ninfer INT8-G64 + Hadamard-D256 rotation on K (D=256
  matches our qk/v head dim; latent 512 = 2×256). fp16 first; INT8-G64 latent later.
- Prefill honesty: prefill serialized one-at-a-time (ninfer cohort scheduler);
  naive prefetch flood during prefill measured 1.43 vs 2.63 tok/s — gate prefetch off
  during prefill (our full-layer streaming prefill sidesteps it anyway).

## Reference numbers (sanity targets)
- ninfer Qwen3.8-27B dense-ish on ONE 3090: C1 70 tok/s decode, prefill ~860 tok/s,
  171K ctx INT8-G64. Our target is MoE-streaming: 25-40 tok/s + MTP is realistic.
- pulsar 2×16GB box: GLM-5.2 744B 2.7 tok/s; tier serves ~90% of expert computations.
- pulsar measured 3090 H2D 13.1 GB/s (Gen4 x16 healthy); our GPU0 x4 Gen3 ≈3.4GB/s
  → GPU0 must NEVER serve streamed experts (static-only if v2 co-compute).
- MLA layer-split design doc (pulsar/docs/mla-layer-split-design.md): per-layer attn
  stacking across cards — backup plan if GPU0 cache capacity ever binds at 250k+.

## Kernel structure references
- ninfer GDN chunked: prepare_wy_wu / state_passing / output + recurrent.cu — structure
  for our KDA chunked prefill (adapt scalar→channelwise decay).
- ninfer T-ladder: gemv → small_t (warp token tiles share one weight read) → simt → mma
  → medium_t_splitk. Our decode m=1..8 → exl3_gemv (QTIP path) covers m≤8 already.
- pulsar indexer: idx_scores_batch f16 m16n8k16 with fused relu-weight epilogue +
  idx_topk — our custom indexer scoring kernel blueprint.
- pulsar sconv_kernel/sconv_state_kernel: short conv with streaming state (k=4 KDA conv).