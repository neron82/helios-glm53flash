#pragma once
// Ported from exllamav3 exllamav3_ext/norm.cuh: same entry points, raw-pointer signatures.
// The graph variants (rms_norm_gr / gated_rms_norm_gr) are dropped; every launcher takes an
// explicit stream instead of pulling one from the torch thread-local.

#include "dtypes.hpp"

namespace helios { namespace aux {

// res_mode 0: y = norm(x) * w
// res_mode 1: y += norm(x) * w                            (post-norm residual accumulate)
// res_mode 2: r += x; y = norm(r) * w                     (fused pre-norm residual add)
enum ResMode : int
{
    RES_NONE = 0,
    RES_POST = 1,
    RES_IN   = 2
};

// Full dispatcher (was rms_norm_impl). rows x dim is the flattened norm view: the original
// span_heads mode simply passed rows = b*s*h, dim = head_dim, so no flag is needed here.
// w may be null (unweighted norm); w must hold dim elements, or w_groups * dim when w_groups > 1.
// r is only dereferenced when res_mode == RES_IN.
void rms_norm_impl
(
    const void* x, DType tx,
    const void* w, DType tw,
    void* y, DType ty,
    const void* r, DType tr,
    int rows,
    int dim,
    float epsilon,
    float constant_bias,
    float constant_scale,
    int res_mode,
    int w_groups,
    Stream s
);

// y = norm(x) * w, or y += norm(x) * w when add_residual.
void rms_norm
(
    const void* x, DType tx,
    const void* w, DType tw,
    void* y, DType ty,
    int rows,
    int dim,
    float epsilon,
    float constant_bias = 0.0f,
    float constant_scale = 1.0f,
    bool add_residual = false,
    int w_groups = 1,
    Stream s = 0
);

// Fused pre-norm residual add: r += x (in place), y = norm(r) * w
void rms_norm_res_in
(
    const void* x, DType tx,
    const void* w, DType tw,
    void* y, DType ty,
    void* r, DType tr,
    int rows,
    int dim,
    float epsilon,
    float constant_bias = 0.0f,
    float constant_scale = 1.0f,
    Stream s = 0
);

// y = norm(x) * w * act(g), act = silu (gate_act 0) or sigmoid (gate_act 1).
// x is bfloat16 only; w float/bfloat16, y half/float, g bfloat16/float.
void gated_rms_norm
(
    const void* x,
    const void* w, DType tw,
    void* y, DType ty,
    const void* g, DType tg,
    int rows,
    int dim,
    float epsilon,
    float constant_bias = 0.0f,
    int w_groups = 1,
    bool gate_first = false,
    int gate_act = 0,
    Stream s = 0
);

}} // namespace helios::aux