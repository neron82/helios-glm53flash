#include <cuda_fp16.h>
#include "hadamard.cuh"
#include "helios_shim.cuh"
#include "hadamard_inner.cuh"

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_hf_r_128_kernel
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    input_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale, r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_ff_r_128_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    input_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale, r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_hf_r_128_dual_kernel
(
    const half* __restrict__ input1_ptr,
    half* __restrict__ output1_ptr,
    const half* __restrict__ scale_1,
    const half* __restrict__ input2_ptr,
    half* __restrict__ output2_ptr,
    const half* __restrict__ scale_2,
    const float r_scale
)
{
    input1_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output1_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<pre_scale, post_scale>(input1_ptr, output1_ptr, scale_1, r_scale);

    input2_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output2_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<pre_scale, post_scale>(input2_ptr, output2_ptr, scale_2, r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_ff_r_128_dual_kernel
(
    const float* __restrict__ input1_ptr,
    float* __restrict__ output1_ptr,
    const half* __restrict__ scale_1,
    const float* __restrict__ input2_ptr,
    float* __restrict__ output2_ptr,
    const half* __restrict__ scale_2,
    const float r_scale
)
{
    input1_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output1_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner<pre_scale, post_scale>(input1_ptr, output1_ptr, scale_1, r_scale);

    input2_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output2_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner<pre_scale, post_scale>(input2_ptr, output2_ptr, scale_2, r_scale);
}

/*
Compute y = (x.view(-1, 128) @ had_128).view(x.shape)
Works inplace if y == x
x and y must be same dtype, either float16 or float32
*/
namespace helios::exl3
{
void had_r_128
(
    const void* input,
    void* output,
    const void* pre_scale,
    const void* post_scale,
    int rows,
    int cols,
    DType dt,
    float scale,
    Stream s
)
{
    HELIOS_ASSERT(!(pre_scale && post_scale), "Cannot mix pre and post scale in had_r_128");
    HELIOS_ASSERT(cols % 128 == 0, "cols must be divisible by 128");
    // Caller responsibility (was TORCH_CHECK_SHAPES_FULL / TORCH_CHECK_DIM / TORCH_CHECK_DTYPE):
    // input and output have the same shape rows * cols of dtype dt, scale buffers fp16
    // with rows * cols entries when provided.

    int blocks = cols / 128;
    float r_scale = scale * 0.088388347648f; // scale / sqrt(128)

    dim3 blockDim(32);
    dim3 gridDim(rows, blocks);

    if (dt == DType::Half)
    {
        if (pre_scale)
            had_hf_r_128_kernel<true, false><<<gridDim, blockDim, 0, s>>>
            (
                (const half*) input,
                (half*) output,
                (const half*) pre_scale,
                r_scale
            );
        else if (post_scale)
            had_hf_r_128_kernel<false, true><<<gridDim, blockDim, 0, s>>>
            (
                (const half*) input,
                (half*) output,
                (const half*) post_scale,
                r_scale
            );
        else
            had_hf_r_128_kernel<false, false><<<gridDim, blockDim, 0, s>>>
            (
                (const half*) input,
                (half*) output,
                (const half*) nullptr,
                r_scale
            );
        cuda_check(cudaPeekAtLastError());
    }

    else if (dt == DType::Float)
    {
        if (pre_scale)
            had_ff_r_128_kernel<true, false><<<gridDim, blockDim, 0, s>>>
            (
                (const float*) input,
                (float*) output,
                (const half*) pre_scale,
                r_scale
            );
        else if (post_scale)
            had_ff_r_128_kernel<false, true><<<gridDim, blockDim, 0, s>>>
            (
                (const float*) input,
                (float*) output,
                (const half*) post_scale,
                r_scale
            );
        else
            had_ff_r_128_kernel<false, false><<<gridDim, blockDim, 0, s>>>
            (
                (const float*) input,
                (float*) output,
                (const half*) nullptr,
                r_scale
            );
        cuda_check(cudaPeekAtLastError());
    }

    else HELIOS_ASSERT(false, "unsupported datatype");
}
} // namespace helios::exl3


namespace helios::exl3
{
void had_r_128_dual
(
    const void* input1,
    void* output1,
    const void* pre_scale1,
    const void* post_scale1,
    const void* input2,
    void* output2,
    const void* pre_scale2,
    const void* post_scale2,
    int rows,
    int cols,
    DType dt,
    float scale,
    Stream s
)
{
    HELIOS_ASSERT(
        (!!pre_scale1 == !!pre_scale2) && (!!post_scale1 == !!post_scale2),
        "Cannot mix scaling modes in dual had"
    );
    HELIOS_ASSERT(!(pre_scale1 && post_scale1), "Cannot mix pre and post scale in dual had");
    HELIOS_ASSERT(cols % 128 == 0, "cols must be divisible by 128");
    // Caller responsibility (was TORCH_CHECK_SHAPES_FULL on all four tensors):
    // input1/output1/input2/output2 share shape rows * cols of dtype dt.

    int blocks = cols / 128;
    float r_scale = scale * 0.088388347648f; // scale / sqrt(128)

    dim3 blockDim(32);
    dim3 gridDim(rows, blocks);

    if (dt == DType::Half)
    {
        if (pre_scale1)
            had_hf_r_128_dual_kernel<true, false><<<gridDim, blockDim, 0, s>>>
            (
                (const half*) input1,
                (half*) output1,
                (const half*) pre_scale1,
                (const half*) input2,
                (half*) output2,
                (const half*) pre_scale2,
                r_scale
            );
        else if (post_scale1)
            had_hf_r_128_dual_kernel<false, true><<<gridDim, blockDim, 0, s>>>
            (
                (const half*) input1,
                (half*) output1,
                (const half*) post_scale1,
                (const half*) input2,
                (half*) output2,
                (const half*) post_scale2,
                r_scale
            );
        else
            had_hf_r_128_dual_kernel<false, false><<<gridDim, blockDim, 0, s>>>
            (
                (const half*) input1,
                (half*) output1,
                (const half*) nullptr,
                (const half*) input2,
                (half*) output2,
                (const half*) nullptr,
                r_scale
            );
        cuda_check(cudaPeekAtLastError());
    }

    else if (dt == DType::Float)
    {
        if (pre_scale1)
            had_ff_r_128_dual_kernel<true, false><<<gridDim, blockDim, 0, s>>>
            (
                (const float*) input1,
                (float*) output1,
                (const half*) pre_scale1,
                (const float*) input2,
                (float*) output2,
                (const half*) pre_scale2,
                r_scale
            );
        else if (post_scale1)
            had_ff_r_128_dual_kernel<false, true><<<gridDim, blockDim, 0, s>>>
            (
                (const float*) input1,
                (float*) output1,
                (const half*) post_scale1,
                (const float*) input2,
                (float*) output2,
                (const half*) post_scale2,
                r_scale
            );
        else
            had_ff_r_128_dual_kernel<false, false><<<gridDim, blockDim, 0, s>>>
            (
                (const float*) input1,
                (float*) output1,
                (const half*) nullptr,
                (const float*) input2,
                (float*) output2,
                (const half*) nullptr,
                r_scale
            );
        cuda_check(cudaPeekAtLastError());
    }

    else HELIOS_ASSERT(false, "unsupported datatype");
}
} // namespace helios::exl3
