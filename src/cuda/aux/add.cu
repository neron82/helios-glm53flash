// Ported from exllamav3 exllamav3_ext/add.cu. Kernel bodies unchanged; launchers take raw
// pointers + explicit dtypes/dims + an explicit stream. The CUDA-graph variants (_gr /
// record_param) are not part of the helios port.
#include "add.cuh"
#include "helios_shim.cuh"
#include <cstdint>

namespace helios { namespace aux {

#define NUM_THREADS 1024

#define KERNEL_DEF(xt, yt, zt, kernel, fn) \
__launch_bounds__(NUM_THREADS) \
__global__ void kernel \
( \
    const xt* __restrict__ x, \
    const yt* __restrict__ y, \
    zt* __restrict__ z, \
    const uint64_t numel_x, \
    const uint64_t numel_y \
) \
{ \
    uint64_t idx = ((uint64_t)blockIdx.x * NUM_THREADS + (uint64_t)threadIdx.x); \
    if (idx >= numel_x) return; \
    xt a = x[idx]; \
    yt b = y[idx % numel_y]; \
    z[idx] = fn; \
}

KERNEL_DEF(half,  half,  half,  add_kernel_hhh, __hadd(a, b))
KERNEL_DEF(half,  half,  float, add_kernel_hhf, __half2float(__hadd(a, b)))
KERNEL_DEF(half,  float, half,  add_kernel_hfh, __float2half_rn(__half2float(a) + b))
KERNEL_DEF(half,  float, float, add_kernel_hff, __half2float(a) + b)
KERNEL_DEF(float, half,  half,  add_kernel_fhh, __float2half_rn(a + __half2float(b)))
KERNEL_DEF(float, half,  float, add_kernel_fhf, a + __half2float(b))
KERNEL_DEF(float, float, half,  add_kernel_ffh, __float2half_rn(a + b))
KERNEL_DEF(float, float, float, add_kernel_fff, a + b)

#undef KERNEL_DEF

/*
x + y -> z
Works inplace if x == z or y == z
*/

// z = x + y, in-place if z == x or z == y. y broadcasts when numel_y < numel_x (must divide
// numel_x exactly); otherwise both operands have numel_x elements.
void add
(
    const void* x, DType xt,
    const void* y, DType yt,
    void* z, DType zt,
    const uint64_t numel_x,
    const uint64_t numel_y,
    Stream stream
)
{
    uint64_t blocks = CEIL_DIVIDE(numel_x, (uint64_t) NUM_THREADS);
    if (numel_y != numel_x)
    {
        HELIOS_AUX_CHECK(numel_y < numel_x, "Tensor shape mismatch (y > x)");
        HELIOS_AUX_CHECK(numel_x % numel_y == 0, "Tensor shape mismatch (y must divide x)");
    }

    #define INSTANCE(xt_, yt_, zt_, xt__, yt__, zt__, kernel) \
    if (xt == xt_ && yt == yt_ && zt == zt_) \
    { \
        kernel<<<(int) blocks, NUM_THREADS, 0, stream>>> \
        ( \
            (const xt__*) x, \
            (const yt__*) y, \
            (zt__*) z, \
            numel_x, \
            numel_y \
        ); \
        HELIOS_CUDA_CHECK(cudaPeekAtLastError()); \
    }

    INSTANCE(kHalf,  kHalf,  kHalf,  half,  half,  half , add_kernel_hhh)
    INSTANCE(kHalf,  kHalf,  kFloat, half,  half,  float, add_kernel_hhf)
    INSTANCE(kHalf,  kFloat, kHalf,  half,  float, half , add_kernel_hfh)
    INSTANCE(kHalf,  kFloat, kFloat, half,  float, float, add_kernel_hff)
    INSTANCE(kFloat, kHalf,  kHalf,  float, half,  half , add_kernel_fhh)
    INSTANCE(kFloat, kHalf,  kFloat, float, half,  float, add_kernel_fhf)
    INSTANCE(kFloat, kFloat, kHalf,  float, float, half , add_kernel_ffh)
    INSTANCE(kFloat, kFloat, kFloat, float, float, float, add_kernel_fff)

    #undef INSTANCE

    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// Strided row-block copy: dst[r, :width] = src[r, :width] for 2D tensors whose row strides may
// differ (zero-padded staging buffers around GEMMs with padded dims)

#define C2D_THREADS 256

template <typename T>
__global__ void copy2d_kernel
(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int src_stride,
    const int dst_stride,
    const int width
)
{
    int col = blockIdx.x * C2D_THREADS + threadIdx.x;
    int row = blockIdx.y;
    if (col >= width) return;
    dst[(int64_t) row * dst_stride + col] = src[(int64_t) row * src_stride + col];
}

void copy2d
(
    const void* src,
    void* dst,
    DType dt,
    int rows,
    int src_stride,
    int dst_stride,
    int width,
    Stream stream
)
{
    dim3 grid(CEIL_DIVIDE(width, C2D_THREADS), rows);

    #define INSTANCE(T) \
    { \
        copy2d_kernel<T><<<grid, C2D_THREADS, 0, stream>>> \
        ( \
            (const T*) src, \
            (T*) dst, \
            src_stride, \
            dst_stride, \
            width \
        ); \
    }

    if (dt == kHalf) INSTANCE(half)
    else if (dt == kFloat) INSTANCE(float)
    else HELIOS_AUX_CHECK(false, "copy2d: unsupported dtype");

    #undef INSTANCE

    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// Per-expert bias adds for the block-sparse MLP graphs. All inputs are static buffers (the
// routing kernel writes sel/weights in place), so the nodes never need patching:
// - moe_bias_add: interm[k, :] += bias[sel[k]] for the gate/up intermediates
// - moe_bias_add_weighted: out[0, :] += sum_k w[k] * bias[sel[k]], correcting the weighted
//   expert reduction for the down bias (bias applies before the routing weight)

__global__ void moe_bias_add_kernel
(
    half* __restrict__ interm,
    const uintptr_t* __restrict__ bias_ptrs,
    const int64_t* __restrict__ sel,
    const int stride,
    const int width,
    const int min_expert,
    const int max_expert,
    const int packed
)
{
    int col = blockIdx.x * C2D_THREADS + threadIdx.x;
    int k = blockIdx.y;
    if (col >= width) return;
    // Expert-parallel split: sel holds global expert indices but the pointer table only covers the
    // local range. Skip foreign experts. At num_tokens == 1 the mgemm PACKS its output rows to the
    // local entries of sel (in order), so the bias lands at the packed row; at num_tokens > 1 the
    // mgemm masks in place (position-preserving), so the bias lands at the slot's own row
    int64_t e = sel[k];
    int row = k;
    if (min_expert >= 0)
    {
        if (e < min_expert || e >= max_expert) return;
        e -= min_expert;
        if (packed)
        {
            row = 0;
            for (int i = 0; i < k; ++i)
            {
                int64_t ei = sel[i];
                if (ei >= min_expert && ei < max_expert) row++;
            }
        }
    }
    const half* b = (const half*) bias_ptrs[e];
    int64_t i = (int64_t) row * stride + col;
    interm[i] = __hadd(interm[i], b[col]);
}

__global__ void moe_bias_add_weighted_kernel
(
    float* __restrict__ out,
    const uintptr_t* __restrict__ bias_ptrs,
    const int64_t* __restrict__ sel,
    const half* __restrict__ weights,
    const int num_sel,      // experts per token (top_k); grid.y = token
    const int width,
    const int min_expert,
    const int max_expert,
    const int out_stride    // row stride of out, in elements
)
{
    int col = blockIdx.x * C2D_THREADS + threadIdx.x;
    int t = blockIdx.y;
    if (col >= width) return;
    const int64_t* sel_t = sel + (int64_t) t * num_sel;
    const half* weights_t = weights + (int64_t) t * num_sel;
    float* out_t = out + (int64_t) t * out_stride;
    float acc = out_t[col];
    for (int k = 0; k < num_sel; ++k)
    {
        // Foreign experts contribute on their own rank only
        int64_t e = sel_t[k];
        if (min_expert >= 0)
        {
            if (e < min_expert || e >= max_expert) continue;
            e -= min_expert;
        }
        const half* b = (const half*) bias_ptrs[e];
        acc += __half2float(weights_t[k]) * __half2float(b[col]);
    }
    out_t[col] = acc;
}

// interm[row, :width] += bias_ptrs[expert][col], one block row per selection.
// min_expert < 0 disables the local-range filter; packed selects the packed-row layout
// (num_tokens == 1 mgemm packing) for the row index.
void moe_bias_add
(
    half* interm,
    const uintptr_t* bias_ptrs,       // device table of per-expert bias pointers
    const int64_t* sel,               // (num_sel,) global expert indices
    int num_sel,
    int stride,
    int width,
    int min_expert,
    int max_expert,
    int packed,
    Stream stream
)
{
    dim3 grid(CEIL_DIVIDE(width, C2D_THREADS), num_sel);

    moe_bias_add_kernel<<<grid, C2D_THREADS, 0, stream>>>
    (
        interm,
        bias_ptrs,
        sel,
        stride,
        width,
        min_expert,
        max_expert,
        packed
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// out[token, col] += sum_k weights[token, k] * bias_ptrs[sel[token, k]][col], correcting the
// weighted expert reduction for the down bias (bias applies before the routing weight).
void moe_bias_add_weighted
(
    float* out,
    const uintptr_t* bias_ptrs,
    const int64_t* sel,               // (num_tokens, num_sel)
    const half* weights,              // (num_tokens, num_sel)
    int num_tokens,
    int num_sel,
    int width,
    int min_expert,
    int max_expert,
    int out_stride,
    Stream stream
)
{
    dim3 grid(CEIL_DIVIDE(width, C2D_THREADS), num_tokens);

    moe_bias_add_weighted_kernel<<<grid, C2D_THREADS, 0, stream>>>
    (
        out,
        bias_ptrs,
        sel,
        weights,
        num_sel,
        width,
        min_expert,
        max_expert,
        out_stride
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

}} // namespace helios::aux
