#pragma once

#include "helios_shim.cuh"

namespace helios
{
namespace exl3
{

// Pack rows*cols blocks of 256 K-bit indices into MSB-first uint16 trellis words.
//   packed:   uint16[rows * cols * (256 * K / 16)]
//   unpacked: uint16[rows * cols * 256]   (one index per value, low K bits significant)
void pack_trellis
(
    void* packed,
    const void* unpacked,
    int rows,
    int cols,
    int K,
    Stream s = 0
);

// Inverse of pack_trellis: bit-exact index round trip (values above K bits are zero).
// NOTE: the original launched this with gridDim(cols, rows) — swapped relative to
// pack_trellis — because the kernel indexes with gridDim.x * blockIdx.y + blockIdx.x.
void unpack_trellis
(
    void* unpacked,
    const void* packed,
    int rows,
    int cols,
    int K,
    Stream s = 0
);

// Fold 16 fp16 sign bits into one uint16 per column (first value ends up in bit 15).
//   packed:   uint16[cols]
//   unpacked: fp16[cols * 16]
void pack_signs
(
    void* packed,
    const void* unpacked,
    int cols,
    Stream s = 0
);

} // namespace exl3
} // namespace helios