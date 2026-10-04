# Colibri findings — condensed for Helios (full report: history://ColibriScout)

## What colibri is
Pure-C tiered streaming engine (SSD→RAM→VRAM), one .c file per model family.
c/glm53.c = complete CPU GLM-5.3-Flash engine (3747 lines) on a CUSTOM int4-gs64
safetensors container (NOT EXL3). No CUDA path for glm53 (Metal/Vulkan only, experts
deliberately excluded). Advanced streaming machinery (io_uring, PILOT prefetch, learned
pins, RAM→VRAM CUDA tier, multi-SSD) lives in c/colibri.c (GLM-5.2) + qwen36_tier.c/h +
tier.h + backend_cuda.cu.

## Correctness oracle value (glm53.c + headers)
- delta_attention.h coli_kda_step: token-exact KDA recurrence vs transformers ref:
  short conv+SiLU, L2-norm eps-inside-sqrt, decay->delta->write two-pass.
- sparse_index.h: DSA k-pool rules: per-CHANNEL softmax pool mixture (gate+APE),
  complete-pool gating (pool*P+P-1 <= q_pos), unconditional tail (width topk+P-1),
  ReLU scores, deterministic ties. Matches exllamav3 exactly.
- hyper_connections.h: coli_hc_split_sinkhorn/pre/post — bit-compatible mHC Sinkhorn.
- swiglu_clamped (glm53.c:1013): GATE ceiling only (min(gate,limit)), UP clamped both
  sides (±limit), out = silu(gate_clamped) * up_clamped. limit=10.
- router asymmetry: score=sigmoid(gate·x); SELECTION argmax over score+bias; WEIGHTS use
  PURE score; normalize /(sum+1e-20) * 2.5.
- absorbed MLA identity q·(W_k c_j) = (W_kᵀ q)·c_j; out = W_v(Σ α_j c_j); scale 1/√qk_nope.
- final collapse: unweighted mean over 4 streams, then final RMSNorm, lm_head.
- tensor names identical to exllamav3 keys (verified vs our dump).
- config validation: layer_types authoritative, cross-check linear_attn_config.

## Streaming systems lessons
- Expert slot = 6 contiguous pieces (gate q4+qs, up q4+qs, down q4+qs); detect contig →
  single pread. EXL3 analogue: merge contiguous tensor runs (suh/svh/mul1/trellis adjacent).
- Union of distinct experts per chunk, processed in BLOCKS ≤ cache capacity (else slots
  overwritten mid-use). Parallel reads (OpenMP / io_uring QD).
- Auto budget: MemAvailable − 3GB floor. LRU slot cache per layer.
- PILOT: predict next-layer top-K from current post-attn state: recall 71.6% (prev-token
  only 41.3%). Shared-expert-corrected prediction +2.3% recall.
- Heat telemetry (.coli_usage) → learned pins (hot-store), REPIN cadence.
- tier.h: LFRU promotion, 25% + 4-slot hysteresis, heat halved every 1024 ticks,
  background-thread uploads via pinned staging — decode NEVER blocks on placement.
  VRAM miss → CPU fallback overlapping in-flight GPU groups.
- Trunk placement economics: dense trunk byte = 1.0 value/token-read; routed expert byte
  = 2·p (p = routing prob).

## Two-GPU measured lessons (qwen36 tier, 8GB cards)
- Thread-local current-device cache: cudaSetDevice per call DOUBLED expert matmul time
  when serial expert loop alternated devices.
- Spreading experts across both cards → every layer gated by slower take(): 10.79 vs 11.09.
  Hand-split (experts on ONE card, trunk across both) = 17.11 vs auto 14.80. → HELIOS:
  experts+router on GPU1 (fast link), trunk on GPU0 — matches our functional split.
- Warm tier gains: 9.63→12.92 tok/s (+34%) on 8GB; RTX 3070 16.55 tok/s on 35B-A3B.

## Disk/CPU economics relevant to us
- GLM5.3 token touches 42 sparse layers × 8 × 14.2MB = 4.8GB (int4-gs64 container).
  EXL3 2-bit experts: ~2.15GB/token — our RAM-read bound ~12 tok/s zero-hit, higher with hits.
- Zen2/Zen3 AVX2-only CPU: expert matmul binds BEFORE disk (#1119) → CUDA W4A16 expert
  compute is worth it on our 3700X even at moderate hit rates. CPU MoE rejected (confirms).
- QD scaling on NVMe: 72 MB/s QD1 → 207 QD16.

## Not reusable
- No EXL3 anywhere in colibri. No CUDA glm53 path. No MTP wired. Python launcher/gateway
  dispensable. stdio serve protocol simple/clean (worth imitating for internals).