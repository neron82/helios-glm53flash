#pragma once
// Ported from exllamav3 exllamav3_ext/routing.cuh: DS3 (sigmoid router + pre-topk bias)
// selection only. Raw-pointer signatures, explicit stream. The std-softmax routing,
// routing_sel_norm and moe_split_* entry points are not part of the helios port.

#include "../cuda_shim.hpp"
#include <cstdint>

namespace helios { namespace aux {

// Score activation for the nogroup top-k kernels (both strictly increasing, so sorting by
// the raw logit is still valid when there is no selection bias).
enum RoutingAct : int
{
    ROUTING_ACT_SIGMOID = 0,   // DS3 / dots
    ROUTING_ACT_SQRTSP    = 1  // DSv4
};

// Single-token router GEMV on a transposed gate copy: scores = x @ gate_t.T.
// Cheaper than a GEMM call at this size; requires even k.
void routing_gemv
(
    const half* x,            // (k)
    const half* gate_t,       // (num_experts, k)
    half* scores,             // (num_experts)
    int k,
    int E,
    Stream s = 0
);

// Top-k selection from precomputed scores. use_topk picks the iterative argmax kernel
// (faster at all measured sizes); false uses the warp radix-sort kernel, which shifts the
// biased key to non-negative before sorting. Weights are the UNBIASED activated scores,
// normalized over the top-k set and multiplied by scaling_factor; indices are ordered by
// the biased key, largest first.
void routing_ds3_nogroup
(
    const half* scores,           // (bsz, num_experts)
    const half* bias,             // (num_experts) or null
    int64_t* topk_indices,        // (bsz, K)
    half* topk_weights,           // (bsz, K)
    int bsz,
    int num_experts,
    int K,
    float scaling_factor,
    bool use_topk = true,
    int act_fn = ROUTING_ACT_SIGMOID,
    Stream s = 0
);

// Decode fast path (bsz == 1): scores = hidden @ gate_t.T, then iterative top-k.
void routing_ds3_nogroup_hidden
(
    const half* hidden,           // (k)
    const half* gate_t,           // (num_experts, k)
    half* scores,                 // (num_experts) scratch
    const half* bias,             // (num_experts) or null
    int64_t* topk_indices,        // (K)
    half* topk_weights,           // (K)
    int k,
    int num_experts,
    int K,
    float scaling_factor,
    int act_fn = ROUTING_ACT_SIGMOID,
    Stream s = 0
);

}} // namespace helios::aux