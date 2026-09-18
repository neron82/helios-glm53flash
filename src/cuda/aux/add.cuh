#pragma once
// Ported from exllamav3 exllamav3_ext/add.cuh: raw-pointer signatures, explicit stream;
// the CUDA-graph (_gr) variants are not part of the helios port.

#include "dtypes.hpp"
#include <cstdint>

namespace helios { namespace aux {

// z = x + y elementwise; works in place when z == x or z == y. x/z/y dtypes are given
// independently (the kernel table covers all 8 half/float combinations). y broadcasts when
// numel_y < numel_x, in which case numel_x % numel_y must be 0.
void add
(
    const void* x, DType xt,
    const void* y, DType yt,
    void* z, DType zt,
    uint64_t numel_x,
    uint64_t numel_y,
    Stream s = 0
);

// Strided row-block copy: dst[r, :width] = src[r, :width] for 2D buffers whose row strides
// may differ (zero-padded staging buffers around GEMMs with padded dims).
void copy2d
(
    const void* src,
    void* dst,
    DType dt,                 // kHalf or kFloat
    int rows,
    int src_stride,
    int dst_stride,
    int width,
    Stream s = 0
);

// Per-expert bias add for block-sparse MLP paths: interm[row, :] += bias_ptrs[sel[k]][:].
// bias_ptrs is a DEVICE table of per-expert bias pointers (uintptr_t). min_expert < 0
// disables the local-range filter (sel holds global ids); packed != 0 places the bias at
// the packed output row (single-token mgemm packing) instead of the selection's own row.
void moe_bias_add
(
    half* interm,
    const uintptr_t* bias_ptrs,
    const int64_t* sel,       // (num_sel,)
    int num_sel,
    int stride,
    int width,
    int min_expert,
    int max_expert,
    int packed,
    Stream s = 0
);

// out[token, :] += sum_k weights[token, k] * bias_ptrs[sel[token, k]][], correcting the
// weighted expert reduction for the down bias (bias applies before the routing weight).
void moe_bias_add_weighted
(
    float* out,
    const uintptr_t* bias_ptrs,
    const int64_t* sel,       // (num_tokens, num_sel)
    const half* weights,      // (num_tokens, num_sel)
    int num_tokens,
    int num_sel,
    int width,
    int min_expert,
    int max_expert,
    int out_stride,
    Stream s = 0
);

}} // namespace helios::aux