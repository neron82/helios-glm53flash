#pragma once
#ifndef HELIOS_KERNEL_SHIM_INCLUDED
#define HELIOS_KERNEL_SHIM_INCLUDED
// Host/device helper shim for the ported exllamav3 EXL3 quant core.
//
// Extracted from exllamav3_ext/util.h, exllamav3_ext/util.cuh and exllamav3_ext/compat.cuh —
// only the entities actually referenced by the ported quant files are kept here. Everything
// torch-flavoured (TORCH_CHECK_* / TORCH_CHECK_DTYPE / OPTPTR / at::Tensor shape assertions)
// is gone: the ported launchers take raw pointers plus explicit dimensions and assert with
// HELIOS_ASSERT / HELIOS_CUDA_CHECK instead.

#include "../cuda_shim.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <array>

// ---------------------------------------------------------------------------
// From exllamav3_ext/util.h (plain macros, no torch dependency)
// ---------------------------------------------------------------------------

#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// Launcher precondition checks: the original used TORCH_CHECK / TORCH_CHECK_SHAPES /
// TORCH_CHECK_SIZE on at::Tensors. Same intent, plain asserts on the explicit dimensions.
#define HELIOS_ASSERT(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "helios assertion failed: %s at %s:%d\n", (msg), __FILE__, __LINE__); abort(); } } while (0)

// The ported launchers kept their original `cuda_check(cudaPeekAtLastError())` tail verbatim;
// this maps it onto the helios CUDA error macro.
#define cuda_check(ans) HELIOS_CUDA_CHECK(ans)

// ---------------------------------------------------------------------------
// From exllamav3_ext/util.cuh (device vector types / unions / byte swap)
// ---------------------------------------------------------------------------

typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    // __device__ __host__ so both the kernels and the host-side test replicas can build these
    half4() = default;
    __device__ __host__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ __host__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
}
half4;

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ __host__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ __host__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ __host__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ __host__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ __host__ half_uint16(half val) : as_half(val) {}
    __device__ __host__ half_uint16() : as_uint16(0) {}
};

#define SWAP16(__x) __byte_perm(__x, 0, 0x1032)

// reduction.cuh's block_reduce_max_h uses this sentinel.
#define NEG_INF_F16 __ushort_as_half(0xFC00)

// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
// From exllamav3_ext/compat.cuh (needed by hadamard_inner.cuh's gelu path)
#endif  // __CUDA_ARCH__

// ---------------------------------------------------------------------------

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

#if defined(USE_ROCM) || !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750 || CUDART_VERSION < 11000))

__inline__ __device__ float tanh_opt(float x)
{
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
}

#else

__inline__ __device__ float tanh_opt(float x)
{
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
}

#endif

// ---------------------------------------------------------------------------
// Replacements for the torch dtype dispatch the original launchers used
// ---------------------------------------------------------------------------

namespace helios
{
namespace exl3
{

// Stands in for at::kHalf / at::kFloat dtype switches in count_inf_nan / had_r_128.
enum class DType : int
{
    Half  = 0,
    Float = 1
};

} // namespace exl3
} // namespace helios
#endif  // HELIOS_KERNEL_SHIM_INCLUDED
