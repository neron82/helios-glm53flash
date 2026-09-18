#pragma once

#include "helios_shim.cuh"

namespace helios
{
namespace exl3
{

// Count inf / NaN values in a fp16 or fp32 buffer.
//   y: two uint64 counters [inf, nan]; caller zeroes them beforehand.
void count_inf_nan
(
    const void* x,
    unsigned long long* y,
    uint64_t numel,
    DType dt,
    Stream s = 0
);

} // namespace exl3
} // namespace helios