#pragma once

#include "helios_shim.cuh"

namespace helios
{
namespace exl3
{

// Dequantize a slice of a packed trellis tensor into fp16.
//   packed:   uint16[packed_rows * packed_cols * (256 * K / 16)]
//   unpacked: fp16[packed_rows * 16 * unpacked_cols], slice starting at column n_offset
//   mcg -> codebook 1, mul1 -> codebook 2, neither -> codebook 0
// n_offset must be a multiple of 128 and n_offset + unpacked_cols <= packed_cols * 16.
void reconstruct_slice
(
    void* unpacked,
    const void* packed,
    int packed_rows,
    int packed_cols,
    int unpacked_cols,
    int K,
    bool mcg,
    bool mul1,
    int64_t n_offset,
    Stream s = 0
);

// Full-width reconstruct (unpacked_cols == packed_cols * 16).
void reconstruct
(
    void* unpacked,
    const void* packed,
    int packed_rows,
    int packed_cols,
    int unpacked_cols,
    int K,
    bool mcg,
    bool mul1,
    Stream s = 0
);

// Fused reconstruct + both-side Hadamard: emits the ORIGINAL-basis weights
//   W = diag(suh) . H128 . W_hat . H128 . diag(svh)
// per 128x128 tile (1/sqrt(128) per side). unpacked_rows/unpacked_cols must both be
// multiples of 128; suh must hold >= unpacked_rows entries, svh >= unpacked_cols
// (svh pre-offset by the caller when n_offset != 0).
void reconstruct_had_slice
(
    void* unpacked,
    const void* packed,
    const void* suh,
    const void* svh,
    int packed_rows,
    int packed_cols,
    int unpacked_rows,
    int unpacked_cols,
    int K,
    bool mcg,
    bool mul1,
    int64_t n_offset,
    Stream s = 0
);

} // namespace exl3
} // namespace helios