// Ported from exllamav3 exllamav3_ext/activation.cu. Kernel bodies unchanged (they live in
// activation_kernels.cuh plus deinterleave_qg_kernel below); launchers take raw pointers +
// explicit sizes + an explicit stream. The CUDA-graph variants (_gr / record_param) are not
// part of the helios port.
#include "activation.cuh"
#include "helios_shim.cuh"
#include <cmath>
#include <cstddef>

namespace helios { namespace aux {

#define NUM_THREADS 256
#define NUM_THREADS_P 1024

#include "activation_kernels.cuh"

// act(x) * y -> z, z fp16; x/y both fp16 or both fp32 (float_input). act_limit != 0 clamps
// y symmetrically and x from above before the multiply. In-place is legal when z == x or y.
template <int ACTIVATION_TYPE>
static void act_mul_launch
(
    const void* x,
    const void* y,
    half* z,
    bool float_input,
    const float act_limit,
    const size_t numel,
    Stream stream
)
{
    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    if (float_input)
    {
        act_mul_kernel_f<ACTIVATION_TYPE><<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const float*) x,
            (const float*) y,
            z,
            act_limit,
            numel
        );
    }
    else
    {
        act_mul_kernel_h<ACTIVATION_TYPE><<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const half*) x,
            (const half*) y,
            z,
            act_limit,
            numel
        );
    }
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// silu(x) * y -> z
void silu_mul
(
    const void* x,
    const void* y,
    half* z,
    bool float_input,
    const float act_limit,
    const size_t numel,
    Stream stream
)
{
    act_mul_launch<ACT_SILU>(x, y, z, float_input, act_limit, numel, stream);
}

// gpt-oss clamped swiglu: (clamp(y, -limit, limit) + 1) * g * sigmoid(1.702 * g),
// g = min(x, limit) -> z
void silu_oai_mul
(
    const void* x,
    const void* y,
    half* z,
    bool float_input,
    const float act_limit,
    const size_t numel,
    Stream stream
)
{
    act_mul_launch<ACT_SILU_OAI>(x, y, z, float_input, act_limit, numel, stream);
}

// gelu(x) * y -> z (tanh approximation)
void gelu_mul
(
    const void* x,
    const void* y,
    half* z,
    bool float_input,
    const float act_limit,
    const size_t numel,
    Stream stream
)
{
    act_mul_launch<ACT_GELU>(x, y, z, float_input, act_limit, numel, stream);
}

// relu^2(x) * y -> z
void relu2_mul
(
    const void* x,
    const void* y,
    half* z,
    bool float_input,
    const float act_limit,
    const size_t numel,
    Stream stream
)
{
    act_mul_launch<ACT_RELU2>(x, y, z, float_input, act_limit, numel, stream);
}

// relu(x) * y -> z. With x == y this computes relu^2(y) exactly, which is how the non-gated
// MoE paths (NemotronH) apply their activation through the same (g, u, a) call shape
void relu_mul
(
    const void* x,
    const void* y,
    half* z,
    bool float_input,
    const float act_limit,
    const size_t numel,
    Stream stream
)
{
    act_mul_launch<ACT_RELU>(x, y, z, float_input, act_limit, numel, stream);
}

// xielu(x, alpha_p, alpha_n) -> y (fp32 input only, fp16 output). alpha_p / alpha_n are the
// RAW alpha scalars; the launcher applies the same softplus-with-linear-threshold transform
// the original did host-side (and +0.5 to alpha_n) before passing them to the kernel.
void xielu
(
    const float* x,
    half* y,
    float alpha_p,
    float alpha_n,
    const size_t numel,
    Stream stream
)
{
    auto get_alpha = [] (float a)
    {
        return a > 20.0f ? a : log1pf(expf(a));
    };

    float p = get_alpha(alpha_p);
    float n = get_alpha(alpha_n) + 0.5f;

    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    xielu_kernel_f<<<blocks, NUM_THREADS, 0, stream>>>
    (
        x,
        y,
        numel,
        p,
        n
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// x * sigmoid(y) + z -> z (y is one scalar per row: size(-1) == 1, hence gdim == 1)
void add_sigmoid_gate
(
    const float* x,
    const float* y,
    float* z,
    const size_t numel,
    int dim,
    Stream stream
)
{
    size_t blocks = CEIL_DIVIDE(numel, NUM_THREADS);
    add_sigmoid_kernel_f<<<blocks, NUM_THREADS, 0, stream>>>
    (
        x,
        y,
        z,
        numel,
        dim
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// x *= sigmoid(y), x and y the same shape, numel even
void mul_sigmoid_
(
    half* x,
    const half* y,
    const size_t numel,
    Stream stream
)
{
    HELIOS_AUX_CHECK(numel % 2 == 0, "x.numel() must be even");

    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    mul_sigmoid_kernel_h<<<blocks, NUM_THREADS, 0, stream>>>
    (
        x,
        y,
        numel
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// x *= sigmoid(y), where x is [B, S, H, D] and y is [B, S, H] (dim = D, numel = x.numel())
void mul_sigmoid_broadcast_
(
    half* x,
    const half* y,
    const size_t numel,
    size_t dim,
    Stream stream
)
{
    HELIOS_AUX_CHECK(dim % 2 == 0, "x.size(3) must be even");

    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    mul_sigmoid_broadcast_kernel_h<<<blocks, NUM_THREADS, 0, stream>>>
    (
        x,
        y,
        numel,
        dim
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// x *= softplus(y), where x is [B, S, H, D] and y is [B, S, H] (Laguna attention gate)
void mul_softplus_broadcast_
(
    half* x,
    const half* y,
    const size_t numel,
    size_t dim,
    Stream stream
)
{
    HELIOS_AUX_CHECK(dim % 2 == 0, "x.size(3) must be even");

    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    mul_softplus_broadcast_kernel_h<<<blocks, NUM_THREADS, 0, stream>>>
    (
        x,
        y,
        numel,
        dim
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// x * sigmoid(y @ w) + z -> z, one block per row (x/z fp32, y/w fp16, w size(-1) == 1)
void add_sigmoid_gate_proj
(
    const float* x,
    const half* y,
    float* z,
    const half* w,
    size_t bsz,
    int dim,
    Stream stream
)
{
    add_sigmoid_proj_kernel_f<<<bsz, NUM_THREADS_P, 0, stream>>>
    (
        x,
        y,
        z,
        w,
        bsz,
        dim
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// Split per-head-interleaved projection output [.., heads, (q: head_dim, g: head_dim)] into
// contiguous q and g tensors. Replaces a chunk/reshape copy pair (and the contiguous() the RoPE
// kernel would otherwise force on the strided q view)

__global__ __launch_bounds__(NUM_THREADS)
void deinterleave_qg_kernel
(
    const uint4* __restrict__ qg,
    uint4* __restrict__ q,
    uint4* __restrict__ g,
    const int hd8,                  // head_dim / 8
    const size_t n8                 // rows * heads * head_dim / 8
)
{
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= n8) return;
    size_t d = i % hd8;
    size_t h = i / hd8;
    size_t src = h * 2 * hd8 + d;
    q[i] = qg[src];
    g[i] = qg[src + hd8];
}

void deinterleave_qg
(
    const half* qg,               // (.., heads * 2 * head_dim) half
    half* q,                      // out (.., heads * head_dim) half
    half* g,                      // out (.., heads * head_dim) half
    int head_dim,
    size_t q_numel,               // q.numel() == g.numel(), qg.numel() == 2 * q_numel
    Stream stream
)
{
    HELIOS_AUX_CHECK(head_dim % 8 == 0, "head_dim must be a multiple of 8");
    HELIOS_AUX_CHECK(q_numel % 8 == 0, "q.numel() must be a multiple of 8");

    int hd8 = head_dim / 8;
    size_t n8 = q_numel / 8;
    size_t blocks = CEIL_DIVIDE(n8, NUM_THREADS);

    deinterleave_qg_kernel<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (const uint4*) qg,
        (uint4*) q,
        (uint4*) g,
        hd8,
        n8
    );

    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

}} // namespace helios::aux
