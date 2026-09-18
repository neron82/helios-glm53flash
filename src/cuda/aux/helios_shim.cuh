#pragma once
#ifndef HELIOS_KERNEL_SHIM_INCLUDED
#define HELIOS_KERNEL_SHIM_INCLUDED
// Shared helper layer for the ported exllamav3 aux kernels.
//
// Extracted from exllamav3's util.h, util.cuh, reduction.cuh and compat.cuh plus the
// device helpers that were duplicated across norm.cu / activation_kernels.cuh / gdn.cu.
// Helper bodies are byte-identical to the originals except:
//   - cuBLAS error plumbing dropped (no cuBLAS in helios)
//   - TORCH_CHECK_* / cuda_check replaced by HELIOS_AUX_CHECK / HELIOS_CUDA_CHECK
//   - USE_ROCM branches dropped (target is sm_86 / CUDA 13)
//
// Include from .cu files only: the device helpers are __device__-qualified.

#include "../cuda_shim.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Host-side macros (were util.h)
// ---------------------------------------------------------------------------

#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// Replaces TORCH_CHECK for launcher preconditions. The ported launchers take raw
// pointers, so dtype checks are encoded in the signatures; shape/count checks stay
// as runtime assertions on the explicit dimension arguments.
#define HELIOS_AUX_CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "helios::aux precondition failed: %s at %s:%d\n", msg, __FILE__, __LINE__); \
    abort(); } } while (0)

// ---------------------------------------------------------------------------
// Vector types and packed load/store (were util.cuh)
// ---------------------------------------------------------------------------

typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    __device__ half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
}
half4;

typedef struct __align__(8) bfloat164
{
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __device__ bfloat164() = default;
    __device__ bfloat164(__nv_bfloat162 x_, __nv_bfloat162 y_): x(x_), y(y_) {}
    __device__ bfloat164(__nv_bfloat16 b0, __nv_bfloat16 b1, __nv_bfloat16 b2, __nv_bfloat16 b3) :
        x(__halves2bfloat162(b0, b1)),
        y(__halves2bfloat162(b2, b3)) {}
}
bfloat164;

typedef struct __align__(16) half8
{
    half2 x;
    half2 y;
    half2 z;
    half2 w;
     __device__ half8() = default;
     __device__ half8(half2 x_, half2 y_, half2 z_, half2 w_) : x(x_), y(y_), z(z_), w(w_) {}
     __device__ half8(half h0, half h1, half h2, half h3, half h4, half h5, half h6, half h7) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)),
         z(__halves2half2(h4, h5)),
         w(__halves2half2(h6, h7)) {}
}
half8;

#define READ128(__x, __y) ((uint4*)&__x)[0] = ((uint4*)(__y))[0];
#define WRITE128(__x, __y) ((uint4*)__x)[0] = ((uint4*)(&__y))[0];
#define READ64(__x, __y) ((uint2*)&__x)[0] = ((uint2*)(__y))[0];
#define WRITE64(__x, __y) ((uint2*)__x)[0] = ((uint2*)(&__y))[0];

#define LOW_TO_FLOAT(__x) __half2float(__low2half(__x))
#define HIGH_TO_FLOAT(__x) __half2float(__high2half(__x))

#define CLAMP(__x, __min, __max) fmaxf(__min, fminf(__x, __max))
#define CLAMP_FP16(__x) CLAMP(__x, -65504.0f, 65504.0f)

#define SWAP16(__x) __byte_perm(__x, 0, 0x1032)

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

__device__ inline float fxor(float v, uint32_t mask)
{
    uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
    *vi ^= mask;
    return v;
}

__device__ inline half2 h2xor(half2 v, uint32_t mask)
{
    uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
    *vi ^= mask;
    return v;
}

#define NEG_INF_F16 __ushort_as_half(0xFC00)
#define POS_INF_F16 __ushort_as_half(0x7C00)

// ---------------------------------------------------------------------------
// Warp / block reductions (were reduction.cuh)
// ---------------------------------------------------------------------------

struct ValIdx
{
    float val;
    int idx;
};

__device__ inline ValIdx warp_reduce_argmax(ValIdx v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_val = __shfl_down_sync(0xffffffff, v.val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        if (other_val > v.val)
        {
            v.val = other_val;
            v.idx = other_idx;
        }
    }
    return v;
}

__device__ inline half warp_reduce_max_h(half v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        half2 other_v = __shfl_down_sync(0xffffffff, __half2half2(v), offset);
        v = __hmax(v, __low2half(other_v));
    }
    return v;
}

__device__ inline float warp_reduce_max_f(float v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v = fmaxf(v, other_v);
    }
    return v;
}

__device__ inline float warp_reduce_sum_f(float v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    return v;
}

__device__ inline float warp_reduce_sum_last_k(float v, int K)
{
    int lane_id = threadIdx.x % 32;
    if (lane_id < (32 - K)) v = 0.0f;
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    return v;
}

__device__ inline float warp_reduce_sum_first_k(float v, int K)
{
    int lane_id = threadIdx.x % 32;
    if (lane_id >= K) v = 0.0f;
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    return v;
}

__device__ inline float block_reduce_sum_broadcast_f(float v, int num_threads)
{
    __shared__ float shared[32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = warp_reduce_sum_f(v);

    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();

    int max_warp_id = num_threads / 32;
    if (warp_id == 0)
    {
        v = lane_id < max_warp_id ? shared[lane_id] : 0.0f;
        v = warp_reduce_sum_f(v);
        shared[0] = v;
    }
    __syncthreads();
    v = shared[0];
    return v;
}

// ---------------------------------------------------------------------------
// Fixed-size / dynamic block sum reductions + packed quad IO (were norm.cu)
// ---------------------------------------------------------------------------

template <int num_threads>
__device__ inline float reduce(float sum, int warp_id, int lane_id)
{
    if constexpr (num_threads <= 32)
    {
        // Shuffle to sum across lanes
        __shared__ float sums[num_threads / 32];
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
        return sum;
    }
    else
    {
        // Shuffle to sum across lanes
        __shared__ float sums[num_threads / 32];
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
        if (lane_id == 0) sums[warp_id] = sum;
        __syncthreads();

        // Load partial sums from across warps, shuffle again across lanes
        sum = sums[lane_id];
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

        return sum;
    }
}

// Block-size-agnostic reduction (any multiple of 32 threads)
__device__ inline float reduce_dyn(float sum, int warp_id, int lane_id)
{
    __shared__ float sums[32];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    int num_warps = blockDim.x / 32;
    if (num_warps == 1) return sum;
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();
    sum = lane_id < num_warps ? sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    return sum;
}

template <bool clamp>
__device__ inline void read_half4(float4& f4, const half4* addr)
{
    half4 h4;
    READ64(h4, addr);
    f4.x = LOW_TO_FLOAT(h4.x);
    f4.y = HIGH_TO_FLOAT(h4.x);
    f4.z = LOW_TO_FLOAT(h4.y);
    f4.w = HIGH_TO_FLOAT(h4.y);
    if constexpr (clamp)
    {
        f4.x = CLAMP_FP16(f4.x);
        f4.y = CLAMP_FP16(f4.y);
        f4.z = CLAMP_FP16(f4.z);
        f4.w = CLAMP_FP16(f4.w);
    }
}

__device__ inline void read_bfloat164(float4& f4, const bfloat164* addr)
{
    bfloat164 h4;
    READ64(h4, addr);
    f4.x = __bfloat162float(__low2bfloat16(h4.x));
    f4.y = __bfloat162float(__high2bfloat16(h4.x));
    f4.z = __bfloat162float(__low2bfloat16(h4.y));
    f4.w = __bfloat162float(__high2bfloat16(h4.y));
}

__device__ inline void read_float4(float4& f4, const float4* addr)
{
    READ128(f4, addr);
}

__device__ inline void write_half4(const float4& f4, half4* addr)
{
    half4 h4
    (
        __halves2half2(__float2half_rn(f4.x), __float2half_rn(f4.y)),
        __halves2half2(__float2half_rn(f4.z), __float2half_rn(f4.w))
    );
    WRITE64(addr, h4);
}

__device__ inline void write_bfloat164(const float4& f4, bfloat164* addr)
{
    bfloat164 h4
    (
        __halves2bfloat162(__float2bfloat16_rz(f4.x), __float2bfloat16_rz(f4.y)),
        __halves2bfloat162(__float2bfloat16_rz(f4.z), __float2bfloat16_rz(f4.w))
    );
    WRITE64(addr, h4);
}

__device__ inline void write_float4(const float4& f4, float4* addr)
{
    WRITE128(addr, f4);
}

__device__ inline float sum_sq4(float lsum, const float4& f4)
{
    lsum = fma(f4.x, f4.x, lsum);
    lsum = fma(f4.y, f4.y, lsum);
    lsum = fma(f4.z, f4.z, lsum);
    lsum = fma(f4.w, f4.w, lsum);
    return lsum;
}

__device__ inline void apply4(float4& x4, const float4& w4, const float rmf)
{
    x4.x = x4.x * w4.x * rmf;
    x4.y = x4.y * w4.y * rmf;
    x4.z = x4.z * w4.z * rmf;
    x4.w = x4.w * w4.w * rmf;
}

__device__ inline void apply4_nw(float4& x4, const float rmf)
{
    x4.x = x4.x * rmf;
    x4.y = x4.y * rmf;
    x4.z = x4.z * rmf;
    x4.w = x4.w * rmf;
}

// ---------------------------------------------------------------------------
// Activation primitives shared by norm / activation / gdn
// (were duplicated in norm.cu, activation_kernels.cuh and gdn.cu)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float _silu(float x)
{
    float e     = __expf(-x);
    float recip = __fdividef(1.0f, 1.0f + e);
    return x * recip;
}

__device__ __forceinline__ float _sigmoid_f(float x)
{
    return __fdividef(1.0f, 1.0f + __expf(-x));
}

__device__ __forceinline__ float _sigmoid_fast_exp(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

// Approximate tanh (were compat.cuh); sm_86 always takes the PTX tanh.approx path.
// These are __device__-only, so no __CUDA_ARCH__ guard is needed (and one breaks callers compiled in
// the host pass, which is what the original unbalanced #ifdef did).
__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
}

#endif  // HELIOS_KERNEL_SHIM_INCLUDED
