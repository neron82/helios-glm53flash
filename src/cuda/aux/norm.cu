// Ported from exllamav3 exllamav3_ext/norm.cu. Kernel bodies are unchanged; the launcher
// layer takes raw pointers + explicit dimensions + an explicit stream, and the torch
// dtype dispatch table is driven by the DType enum instead of at::ScalarType.
#include "norm.cuh"
#include "helios_shim.cuh"
#include <type_traits>
#include <cstddef>

namespace helios { namespace aux {

#define NUM_THREADS 1024
using bfloat16 = __nv_bfloat16;

// res_mode values (RES_NONE / RES_POST / RES_IN) come from norm.cuh.

template <int res_mode, typename input_t, typename output_t, typename weight_t, typename residual_t>
__global__ __launch_bounds__(NUM_THREADS)
void rms_norm_kernel
(
    const input_t* __restrict__ x,
    const weight_t* __restrict__ w,
    output_t* __restrict__ y,
    residual_t* __restrict__ r,
    const float epsilon,
    const int rows,
    const int dim,
    const float constant_bias,
    const float constant_scale,
    const int w_groups          // weight spans w_groups rows, cycled by row index (grouped norm)
)
{
    constexpr bool input_fp32 = std::is_same_v<input_t, float>;
    constexpr bool output_fp32 = std::is_same_v<output_t, float>;
    constexpr bool input_fp16 = std::is_same_v<input_t, half>;
    constexpr bool output_fp16 = std::is_same_v<output_t, half>;
    static_assert(input_fp32 || input_fp16, "rms_norm_kernel: input must be float or half type");
    static_assert(output_fp32 || output_fp16, "rms_norm_kernel: output must be float or half type");
    constexpr bool weight_bf16 = std::is_same_v<weight_t, bfloat16>;
    constexpr bool residual_fp16 = std::is_same_v<residual_t, half>;

    int t = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x;

    if (w && w_groups > 1)
        w += (size_t) (row % w_groups) * dim;

    int columns = dim / 4;
    bool single = columns <= blockDim.x;

    auto read_in = [&] (float4& f4, const input_t* addr)
    {
        if constexpr (input_fp16) read_half4<true>(f4, (const half4*) addr);
        if constexpr (input_fp32) read_float4(f4, (const float4*) addr);
    };

    auto add_resid_in = [&] (float4& x4, int column)
    {
        // r += x, rounded to the residual dtype so the result matches an unfused add
        float4 r4;
        if constexpr (residual_fp16) read_half4<false>(r4, ((const half4*) (r + row * dim)) + column);
        else                         read_float4(r4, ((const float4*) (r + row * dim)) + column);
        x4.x += r4.x;
        x4.y += r4.y;
        x4.z += r4.z;
        x4.w += r4.w;
        if constexpr (residual_fp16)
        {
            half4 h4
            (
                __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y)),
                __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w))
            );
            WRITE64(((half4*) (r + row * dim)) + column, h4);
            x4.x = LOW_TO_FLOAT(h4.x);
            x4.y = HIGH_TO_FLOAT(h4.x);
            x4.z = LOW_TO_FLOAT(h4.y);
            x4.w = HIGH_TO_FLOAT(h4.y);
        }
        else
            write_float4(x4, ((float4*) (r + row * dim)) + column);
    };

    auto apply_out = [&] (float4& x4, int column, float rmf)
    {
        if (w)
        {
            float4 w4;
            if constexpr (weight_bf16) read_bfloat164   (w4, ((const bfloat164*) w) + column);
            else                       read_half4<false>(w4, ((const half4*)     w) + column);
            if (constant_bias != 0.0f)
            {
                w4.x += constant_bias;
                w4.y += constant_bias;
                w4.z += constant_bias;
                w4.w += constant_bias;
            }
            apply4(x4, w4, rmf);
        }
        else
        {
            apply4_nw(x4, rmf);
        }

        if constexpr (res_mode == RES_POST)
        {
            float4 r4;
            if constexpr (output_fp16) read_half4<false>(r4, ((half4*) (y + row * dim)) + column);
            if constexpr (output_fp32) read_float4(r4, ((float4*) (y + row * dim)) + column);
            x4.x += r4.x;
            x4.y += r4.y;
            x4.z += r4.z;
            x4.w += r4.w;
        }

        if constexpr (output_fp16) write_half4(x4, ((half4*) (y + row * dim)) + column);
        if constexpr (output_fp32) write_float4(x4, ((float4*) (y + row * dim)) + column);
    };

    if (single)
    {
        // One float4 per thread: keep the value in a register between the two phases
        float4 x4 = {};
        float sum = 0.0f;
        if (t < columns)
        {
            read_in(x4, x + row * dim + 4 * t);
            if constexpr (res_mode == RES_IN) add_resid_in(x4, t);
            sum = sum_sq4(sum, x4);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;
        if (t < columns)
            apply_out(x4, t, rmf);
    }
    else
    {
        float sum = 0.0f;
        for (int column = t; column < columns; column += blockDim.x)
        {
            float4 x4;
            read_in(x4, x + row * dim + 4 * column);
            if constexpr (res_mode == RES_IN) add_resid_in(x4, column);
            sum = sum_sq4(sum, x4);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;

        for (int column = t; column < columns; column += blockDim.x)
        {
            float4 x4;
            // For RES_IN the summed values were written back to r in the first pass
            if constexpr (res_mode == RES_IN)
            {
                if constexpr (residual_fp16) read_half4<false>(x4, ((const half4*) (r + row * dim)) + column);
                else                         read_float4(x4, ((const float4*) (r + row * dim)) + column);
            }
            else
                read_in(x4, x + row * dim + 4 * column);
            apply_out(x4, column, rmf);
        }
    }
}

/*
Compute RMSNorm: y = x * w / sqrt(row_mean(x * x) + epsilon)
- Can operate in-place if y == x
- x can be either float or half dtype
- y can be either float or half dtype
- w can be either bfloat16 or half dtype
*/
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
)
{
    HELIOS_AUX_CHECK(dim % 4 == 0, "rms_norm: dim must be divisible by 4");
    // w must hold dim elements (w_groups == 1) or w_groups * dim elements (grouped norm)
    HELIOS_AUX_CHECK(w_groups >= 1, "rms_norm: w_groups must be >= 1");

    if (res_mode == RES_IN)
        HELIOS_AUX_CHECK(r != nullptr, "rms_norm: res_mode RES_IN requires residual pointer");

    // Size the block to the row so short rows don't idle warps through the reduction
    int threads = MIN(NUM_THREADS, CEIL_DIVIDE(dim / 4, 32) * 32);

    dim3 blockDim(threads, 1, 1);
    dim3 gridDim(rows, 1, 1);

    // Launch macro
    #define __(_tx, __tx, _tw, __tw, _ty, __ty, _res, _tr, __tr)                                   \
    if (tx == _tx && tw == _tw && ty == _ty && res_mode == _res && tr == _tr)                      \
        rms_norm_kernel<_res, __tx, __ty, __tw, __tr><<<gridDim, blockDim, 0, s>>>                 \
        (                                                                                          \
            (const __tx*) x,                                                                       \
            (const __tw*) w,                                                                       \
            (__ty*) y,                                                                             \
            (__tr*) r,                                                                             \
            epsilon,                                                               \
            rows,                                                                   \
            dim,                                                                    \
            constant_bias,                                                          \
            constant_scale,                                                         \
            w_groups                                                                \
        );

    //      x_type________ w_type_____________  y_type_______        mode      r_type
         __(kHalf,  half,  kHalf,     half,     kHalf,  half,  RES_NONE, kHalf,  half)
    else __(kHalf,  half,  kHalf,     half,     kFloat, float, RES_NONE, kHalf,  half)
    else __(kFloat, float, kHalf,     half,     kHalf,  half,  RES_NONE, kFloat, float)
    else __(kFloat, float, kHalf,     half,     kFloat, float, RES_NONE, kFloat, float)
    else __(kFloat, float, kBFloat16, bfloat16, kHalf,  half,  RES_NONE, kFloat, float)
    else __(kFloat, float, kBFloat16, bfloat16, kFloat, float, RES_NONE, kFloat, float)
    else __(kHalf,  half,  kBFloat16, bfloat16, kHalf,  half,  RES_NONE, kHalf,  half)
    else __(kHalf,  half,  kBFloat16, bfloat16, kFloat, float, RES_NONE, kHalf,  half)
    else __(kHalf,  half,  kHalf,     half,     kHalf,  half,  RES_POST, kHalf,  half)
    else __(kHalf,  half,  kHalf,     half,     kFloat, float, RES_POST, kHalf,  half)
    else __(kFloat, float, kHalf,     half,     kHalf,  half,  RES_POST, kFloat, float)
    else __(kFloat, float, kHalf,     half,     kFloat, float, RES_POST, kFloat, float)
    else __(kFloat, float, kBFloat16, bfloat16, kHalf,  half,  RES_POST, kFloat, float)
    else __(kFloat, float, kBFloat16, bfloat16, kFloat, float, RES_POST, kFloat, float)
    else __(kHalf,  half,  kBFloat16, bfloat16, kHalf,  half,  RES_POST, kHalf,  half)
    else __(kHalf,  half,  kBFloat16, bfloat16, kFloat, float, RES_POST, kHalf,  half)
    else __(kHalf,  half,  kHalf,     half,     kHalf,  half,  RES_IN,   kHalf,  half)
    else __(kHalf,  half,  kHalf,     half,     kHalf,  half,  RES_IN,   kFloat, float)
    else __(kFloat, float, kHalf,     half,     kHalf,  half,  RES_IN,   kHalf,  half)
    else __(kFloat, float, kHalf,     half,     kHalf,  half,  RES_IN,   kFloat, float)
    else __(kFloat, float, kBFloat16, bfloat16, kHalf,  half,  RES_IN,   kHalf,  half)
    else __(kFloat, float, kBFloat16, bfloat16, kHalf,  half,  RES_IN,   kFloat, float)
    else __(kHalf,  half,  kBFloat16, bfloat16, kHalf,  half,  RES_IN,   kHalf,  half)
    else __(kHalf,  half,  kBFloat16, bfloat16, kHalf,  half,  RES_IN,   kFloat, float)

    else HELIOS_AUX_CHECK(false, "rms_norm: Invalid datatypes for input/output");
    #undef __

    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// y = norm(x) * w (add_residual: y += norm(x) * w). w may be null (unweighted norm).
void rms_norm
(
    const void* x, DType tx,
    const void* w, DType tw,
    void* y, DType ty,
    int rows,
    int dim,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool add_residual,
    int w_groups,
    Stream s
)
{
    // Outside RES_IN the residual pointer is never dereferenced; the original derived
    // r_type from x_type in that case, which the table below reproduces.
    rms_norm_impl(x, tx, w, tw, y, ty, nullptr, tx, rows, dim, epsilon,
                  constant_bias, constant_scale, add_residual ? RES_POST : RES_NONE,
                  w_groups, s);
}

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
    float constant_bias,
    float constant_scale,
    Stream s
)
{
    rms_norm_impl(x, tx, w, tw, y, ty, r, tr, rows, dim, epsilon,
                  constant_bias, constant_scale, RES_IN, 1, s);
}


template <int num_threads, typename output_t, typename weight_t, typename gate_t>
__global__ __launch_bounds__(num_threads)
void gated_rms_norm_kernel
(
    const bfloat16* __restrict__ x,
    const weight_t* __restrict__ w,
    output_t* __restrict__ y,
    const gate_t* __restrict__ g,
    const float epsilon,
    const int rows,
    const int dim,
    float constant_bias,
    const int w_groups,         // weight spans w_groups rows, cycled by row index (Mamba2 group norm)
    const bool gate_first,      // apply silu(g) before the norm instead of after (Mamba2 style)
    const int gate_act          // 0 = silu (GDN/Mamba2), 1 = sigmoid (KDA)
)
{
    #define _gate_fn(v) (gate_act == 1 ? _sigmoid_f(v) : _silu(v))
    constexpr bool output_fp32 = std::is_same_v<output_t, float>;
    constexpr bool output_fp16 = std::is_same_v<output_t, half>;
    constexpr bool weight_bf16 = std::is_same_v<weight_t, bfloat16>;
    constexpr bool gate_fp32   = std::is_same_v<gate_t,   float>;

    int t = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x;

    int columns = dim / 4;
    const weight_t* w_row = w + (size_t) (row % w_groups) * dim;

    // Compute sum of squares
    float sum = 0.0f;
    for (int column = t; column < columns; column += num_threads)
    {
        float4 x4;
        read_bfloat164(x4, ((const bfloat164*) (x + row * dim)) + column);
        if (gate_first)
        {
            float4 g4;
            if constexpr (gate_fp32)   read_float4   (g4, ((const float4*)    (g + row * dim)) + column);
            else                       read_bfloat164(g4, ((const bfloat164*) (g + row * dim)) + column);
            x4.x *= _gate_fn(g4.x);
            x4.y *= _gate_fn(g4.y);
            x4.z *= _gate_fn(g4.z);
            x4.w *= _gate_fn(g4.w);
        }
        sum = sum_sq4(sum, x4);
    }
    sum = reduce<num_threads>(sum, warp_id, lane_id);

    // Get norm
    float rmf = rsqrtf(sum / (float)dim + epsilon);

    // Normalize x, scaling by w * silu(g)
    for (int column = t; column < columns; column += num_threads)
    {
        float4 x4;
        float4 w4;
        float4 g4;

        read_bfloat164(x4, ((const bfloat164*) (x + row * dim)) + column);
        if constexpr (weight_bf16) read_bfloat164(w4, ((const bfloat164*) w_row) + column);
        else                       read_float4   (w4, ((const float4*)    w_row) + column);
        if constexpr (gate_fp32)   read_float4   (g4, ((const float4*)    (g + row * dim)) + column);
        else                       read_bfloat164(g4, ((const bfloat164*) (g + row * dim)) + column);

        if (constant_bias != 0.0f)
        {
            w4.x += constant_bias;
            w4.y += constant_bias;
            w4.z += constant_bias;
            w4.w += constant_bias;
        }

        if (gate_first)
        {
            x4.x *= _gate_fn(g4.x);
            x4.y *= _gate_fn(g4.y);
            x4.z *= _gate_fn(g4.z);
            x4.w *= _gate_fn(g4.w);

            apply4(x4, w4, rmf);
        }
        else
        {
            apply4(x4, w4, rmf);

            x4.x *= _gate_fn(g4.x);
            x4.y *= _gate_fn(g4.y);
            x4.z *= _gate_fn(g4.z);
            x4.w *= _gate_fn(g4.w);
        }

        if constexpr (output_fp16) write_half4(x4, ((half4*) (y + row * dim)) + column);
        if constexpr (output_fp32) write_float4(x4, ((float4*) (y + row * dim)) + column);
    }
    #undef _gate_fn
}


/*
Compute RMSNorm: y = x * w / sqrt(row_mean(x * x) + epsilon) * act(g), act = silu or sigmoid
- bfloat16 input only, half/float output
- w_groups > 1: w holds w_groups weight rows of size dim, selected by (row % w_groups). Used for
  Mamba2 group norm where the norm spans dim channels but the weight covers the full inner dim
- gate_first: y = norm(x * silu(g)) * w instead of norm(x) * w * silu(g) (Mamba2 style)
*/
void gated_rms_norm
(
    const void* x,
    const void* w, DType tw,
    void* y, DType ty,
    const void* g, DType tg,
    int rows,
    int dim,
    float epsilon,
    float constant_bias,
    int w_groups,
    bool gate_first,
    int gate_act,
    Stream s
)
{
    HELIOS_AUX_CHECK(dim % 4 == 0, "gated_rms_norm: dim must be divisible by 4");
    HELIOS_AUX_CHECK(w_groups >= 1, "gated_rms_norm: w must have w_groups * dim elements");

    bool small = (dim <= 256);

    dim3 blockDim(small ? 32 : NUM_THREADS, 1, 1);
    dim3 gridDim(rows, 1, 1);

    constexpr DType tx = kBFloat16;

    // Launch macro
    #define __(_tx, __tx, _tw, __tw, _ty, __ty, _tg, __tg, _small, __num_threads)               \
    if (small == _small && tx == _tx && tw == _tw && ty == _ty && tg == _tg)                    \
        gated_rms_norm_kernel<__num_threads><<<gridDim, blockDim, 0, s>>>                       \
        (                                                                                       \
            (const __tx*) x,                                                                    \
            (const __tw*) w,                                                                    \
            (__ty*) y,                                                                          \
            (const __tg*) g,                                                                    \
            epsilon,                                                                            \
            rows,                                                                               \
            dim,                                                                                \
            constant_bias,                                                                      \
            w_groups,                                                                           \
            gate_first,                                                                         \
            gate_act                                                                            \
        );

    //      x_type_____________  w_type_____________  y_type_______  g_type_____________  small  num_threads
         __(kBFloat16, bfloat16, kFloat,    float,    kHalf,  half,  kBFloat16, bfloat16, true,  32         )
    else __(kBFloat16, bfloat16, kFloat,    float,    kHalf,  half,  kBFloat16, bfloat16, false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kFloat,    float,    kFloat, float, kBFloat16, bfloat16, true,  32         )
    else __(kBFloat16, bfloat16, kFloat,    float,    kFloat, float, kBFloat16, bfloat16, false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kHalf,  half,  kBFloat16, bfloat16, true,  32         )
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kHalf,  half,  kBFloat16, bfloat16, false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kFloat, float, kBFloat16, bfloat16, true,  32         )
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kFloat, float, kBFloat16, bfloat16, false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kFloat,    float,    kHalf,  half,  kFloat,    float,    true,  32         )
    else __(kBFloat16, bfloat16, kFloat,    float,    kHalf,  half,  kFloat,    float,    false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kFloat,    float,    kFloat, float, kFloat,    float,    true,  32         )
    else __(kBFloat16, bfloat16, kFloat,    float,    kFloat, float, kFloat,    float,    false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kHalf,  half,  kFloat,    float,    true,  32         )
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kHalf,  half,  kFloat,    float,    false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kFloat, float, kFloat,    float,    true,  32         )
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kFloat, float, kFloat,    float,    false, NUM_THREADS)

    else HELIOS_AUX_CHECK(false, "gated_rms_norm: Invalid datatypes for input/output");
    #undef __

    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

}} // namespace helios::aux
