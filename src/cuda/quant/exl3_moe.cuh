#pragma once
// Fused MoE MLP API (ported from exllamav3_ext/quant/exl3_moe.cuh). Raw pointers + explicit
// stream; the *_gr graph-capture variant was dropped. See README_PORT.md.

#include "helios_shim.cuh"
#include "exl3_moe_common.cuh"

#include <stdint.h>

namespace helios
{
namespace exl3
{

// Per-expert pointer tables, each of length num_experts, all DEVICE arrays of pointers:
//   *_trellis -> EXL3 trellis of that expert's projection, logically (K / 16, N / 16, 16 * bits) uint16
//   *_suh     -> packed input scales/flips, logically (K / 16) float16
//   *_svh     -> packed output scales/flips, logically (N / 16) float16
// All nine tables must be the same length. gate/up/down must share one codebook (mcg xor mul1).
struct ExpertTables
{
    const uint16_t** gate_trellis;
    const half**     gate_suh;
    const half**     gate_svh;
    const uint16_t** up_trellis;
    const half**     up_suh;
    const half**     up_svh;
    const uint16_t** down_trellis;
    const half**     down_suh;
    const half**     down_svh;
};

// Maximum number of experts the fused kernel can process concurrently, given a device's SM count
// (one expert group occupies MOE_SMS_PER_EXPERT SMs). Sizes the temp_* buffer first dimension.
int moe_max_concurrency(int device);

// Fused mixture-of-experts MLP for EXL3 weights: for every expert, gather its tokens, run
// gate/up projections (with the input Hadamard transform + suh), apply the activation
// (act_limit-clamped SiLU/GeLU/relu2) with the output transform + svh, run the down projection
// (again suh/svh), scale by each token's routing weight and atomically accumulate into y.
//
//   x                     -> input hidden state, fp16 [tokens, hidden_dim], contiguous
//   y                       -> output hidden state, fp32 [tokens, hidden_dim], CONTIGUOUS AND
//                             ZERO-INITIALIZED (accumulated atomically)
//   expert_count            -> bincount of expert ids over all tokens, int64 [num_experts + 1];
//                             the last entry is ignored (tokens activating fewer than topk experts)
//   token_sorted            -> token indices sorted by expert, int64 [tokens * topk]
//   weight_sorted           -> routing weight per sorted token, fp16 [tokens * topk]
//   temp_state_g / temp_state_u
//                           -> staged input scratch, fp16
//                              [concurrency, max_tokens_per_expert, hidden_dim]
//   temp_intermediate_g / temp_intermediate_u
//                           -> intermediate scratch, fp16
//                              [concurrency, max_tokens_per_expert, intermediate_dim]
//   tables                  -> per-expert pointer tables (see ExpertTables)
//   tokens, hidden_dim, intermediate_dim, num_experts, topk, max_tokens_per_expert, concurrency
//                           -> the extents the buffers above were allocated with
//   K_gate, K_up, K_down    -> bit widths (trellis third dim / 16). Equal values select a
//                             compile-time-bitrate kernel; otherwise the runtime-bitrate variant runs
//   mcg / mul1              -> codebook selection (exactly one must be true; the kernel only
//                             supports the mcg and mul1 codebooks)
//   act_limit               -> swiglu clamp limit (exllamav3 default 10.0)
//   act_function            -> MOE_ACT_SILU (default), MOE_ACT_GELU, MOE_ACT_RELU2_NOGATE
//   num_active              -> experts with 0 < token count <= max_tokens_per_expert, i.e. the
//                             experts this launch processes. Sizes the grid: fewer, wider expert
//                             groups when few experts are active. -1 = unknown (default: minimum
//                             group width at maximum concurrency)
//
// Limitations: hidden_dim and intermediate_dim must be multiples of 128 (256 selects the wider
// tile instantiation). All blocks of the grid must be co-resident for the group barriers, hence
// concurrency * MOE_SMS_PER_EXPERT <= device SM count. Returns nothing; errors abort.
void moe_grouped
(
    const void*         x,
    void*               y,
    const int64_t*      expert_count,
    const int64_t*      token_sorted,
    const half*         weight_sorted,
    void*               temp_state_g,
    void*               temp_state_u,
    void*               temp_intermediate_g,
    void*               temp_intermediate_u,
    const ExpertTables& tables,
    int                 tokens,
    int                 hidden_dim,
    int                 intermediate_dim,
    int                 num_experts,
    int                 topk,
    int                 max_tokens_per_expert,
    int                 concurrency,
    int                 K_gate,
    int                 K_up,
    int                 K_down,
    bool                mcg           = false,
    bool                mul1          = false,
    float               act_limit     = 10.0f,
    int                 act_function  = MOE_ACT_SILU,
    int                 num_active    = -1,
    Stream              s             = 0
);

} // namespace exl3
} // namespace helios