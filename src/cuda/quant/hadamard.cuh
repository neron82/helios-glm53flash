#pragma once

#include "helios_shim.cuh"

namespace helios
{
namespace exl3
{

// y <- had_128(x) applied to each consecutive 128-element group, scaled by `scale`
// (the kernel multiplies by scale / sqrt(128)). rows*cols elements of dtype dt.
// Exactly one of pre_scale / post_scale may be non-null (fp16, rows * cols entries);
// if both are null the plain transform runs. Works in place when input == output.
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
    Stream s = 0
);

// Same transform applied to two tensors in one launch; scaling mode must match.
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
    Stream s = 0
);

} // namespace exl3
} // namespace helios