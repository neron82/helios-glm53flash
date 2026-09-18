#pragma once
// Ported from exllamav3 exllamav3_ext/activation.cuh: raw-pointer signatures, explicit
// stream; the CUDA-graph (_gr) variants are not part of the helios port.

#include "../cuda_shim.hpp"
#include <cstddef>

namespace helios { namespace aux {

// Elementwise activation selector for act_mul_kernel_h / act_mul_kernel_f.
enum ActivationType : int
{
    ACT_SILU     = 0,   // silu(x) * y
    ACT_GELU     = 1,   // tanh-approx gelu(x) * y
    ACT_RELU2    = 2,   // relu(x)^2 * y
    ACT_SILU_OAI = 3,   // gpt-oss clamped swiglu: (clamp(y,-L,L)+1) * min(x,L) * sigmoid(1.702*min(x,L))
    ACT_RELU     = 4    // relu(x) * y (x == y gives relu^2(y), non-gated MoE paths)
};

// act(x) * y -> z. z is always fp16; x and y are both fp16 or both fp32 (float_input).
// act_limit != 0 clamps y to [-limit, limit] and x from above before the multiply
// (ignored by ACT_SILU_OAI, which takes its limit from the same argument internally).
// numel is the element count of x (must be even). In-place is legal when z == x or z == y.
void silu_mul
(
    const void* x, const void* y, half* z,
    bool float_input, float act_limit, size_t numel, Stream s = 0
);
void silu_oai_mul
(
    const void* x, const void* y, half* z,
    bool float_input, float act_limit, size_t numel, Stream s = 0
);
void gelu_mul
(
    const void* x, const void* y, half* z,
    bool float_input, float act_limit, size_t numel, Stream s = 0
);
void relu2_mul
(
    const void* x, const void* y, half* z,
    bool float_input, float act_limit, size_t numel, Stream s = 0
);
void relu_mul
(
    const void* x, const void* y, half* z,
    bool float_input, float act_limit, size_t numel, Stream s = 0
);

// xielu(x, alpha_p, alpha_n) -> y; fp32 input only, fp16 output. alpha_p / alpha_n are the
// raw alpha scalars; the launcher applies softplus (linear above 20) to both and adds 0.5
// to alpha_n, exactly as the original did from the CPU tensors.
void xielu
(
    const float* x, half* y,
    float alpha_p, float alpha_n, size_t numel, Stream s = 0
);

// z = x * sigmoid(y) + z, where y holds one scalar per row (size(-1) == 1); all fp32.
// dim = x.size(-1), numel = x.numel().
void add_sigmoid_gate
(
    const float* x, const float* y, float* z,
    size_t numel, int dim, Stream s = 0
);

// x *= sigmoid(y), x and y fp16 with identical element counts, numel even.
void mul_sigmoid_
(
    half* x, const half* y, size_t numel, Stream s = 0
);

// x *= sigmoid(y), where x is [B, S, H, D] and y is [B, S, H] (both fp16).
// numel = x.numel(), dim = D (must be even).
void mul_sigmoid_broadcast_
(
    half* x, const half* y, size_t numel, size_t dim, Stream s = 0
);

// x *= softplus(y), same shapes as mul_sigmoid_broadcast_ (Laguna attention gate).
void mul_softplus_broadcast_
(
    half* x, const half* y, size_t numel, size_t dim, Stream s = 0
);

// z = x * sigmoid(y @ w) + z, one block per row: x/z fp32, y/w fp16, w size(-1) == 1.
// bsz = x.numel() / dim.
void add_sigmoid_gate_proj
(
    const float* x, const half* y, float* z, const half* w,
    size_t bsz, int dim, Stream s = 0
);

// Split per-head-interleaved projection output [.., heads, (q: head_dim, g: head_dim)] into
// contiguous q and g (fp16). head_dim must be a multiple of 8; q_numel == g.numel() and
// qg.numel() == 2 * q_numel.
void deinterleave_qg
(
    const half* qg, half* q, half* g,
    int head_dim, size_t q_numel, Stream s = 0
);

}} // namespace helios::aux