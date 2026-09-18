#pragma once
// Public EXL3 matmul API (ported from exllamav3_ext/quant/exl3_gemm.cuh).
//
// Torch is gone: every entry point takes raw device pointers plus explicit dimensions and an
// explicit cudaStream_t (last argument, default 0 = the legacy default stream). The original
// `*_gr` graph-capture variants and the cooperative-kernel autotuner were dropped; the plain
// deterministic selection path (formerly taken when force_shape_idx / force_num_sms were set)
// is now the only path. See README_PORT.md.

#include "helios_shim.cuh"

namespace helios
{
namespace exl3
{

// One EXL3-quantized linear, as a bundle of device pointers. Mirrors the caller-side
// "GroupWords" the engine keeps per tensor:
//   trellis -> EXL3 trellis, logically (K / 16, N / 16, 16 * bits) uint16
//   suh     -> packed input scales/flips, logically (K / 16) float16
//   svh     -> packed output scales/flips, logically (N / 16) float16
//   mul1    -> nonzero selects the mul1 codebook (cb 2); zero selects the default codebook (cb 0).
//             The mcg codebook (cb 1) is selected through the `mcg` argument of the entry points
//             (the original forbade mcg and mul1 at the same time).
//
// Semantics: the weights are Hadamard-rotated per 128-wide k-group, so `x` MUST be transformed
// on input (had prologue with suh, done inside the kernel into `a_had`), the dequantized product
// is post-scaled by svh, and the output carries a Hadamard epilogue. Net effect per 128x128 tile:
//     W = diag(suh) * H128 * W_hat * H128 * diag(svh),   H128 scaled by 1/sqrt(128) per side.
struct GroupWords
{
    const uint16_t* trellis;
    const half*     suh;
    const half*     svh;
    int             mul1;
};

// C[M, N] = A[M, K] @ B[K, N], B EXL3-quantized.
//
//   y       -> C, row-major [M, N], float16 or float32 (y_fp32), contiguous, need not be zeroed
//   x       -> A, row-major [M, K], float16, contiguous
//   w       -> quantized B bundle (see GroupWords); suh, svh and a_had are REQUIRED: the kernel
//              applies the input/output Hadamard stages unconditionally
//   a_had   -> float16 scratch with room for M * K elements (input transform target)
//   M, N, K -> matmul extents; bits = quantization bit width (trellis third dim / 16)
//
// Limitations: K % 16 == 0, N % 128 == 0 (and, for the Hadamard stages to be meaningful,
// K and N multiples of 128). M <= 16 per kernel slab; larger M loops over 16-row slabs.
// Small M (<= 8) may be routed to the GEMV kernel when the shape heuristic applies; that
// returns 90. Otherwise returns the selected shape index (2, 3 or 4).
int gemm
(
    void*               y,
    const void*         x,
    const GroupWords&   w,
    int                 M,
    int                 N,
    int                 K,
    int                 bits,
    bool                y_fp32          = false,
    Stream              s               = 0,
    void*               a_had           = nullptr,
    bool                mcg             = false,
    int                 force_shape_idx = 0,
    int                 force_num_sms   = 0
);

// Forced small-m GEMV path (exl3_gemv_kernel.cuh). Requires suh, svh and a_had. Eligibility
// and configuration are identical to the automatic route inside gemm(); asserts if ineligible.
void gemv
(
    void*               y,
    const void*         x,
    const GroupWords&   w,
    int                 M,
    int                 N,
    int                 K,
    int                 bits,
    bool                y_fp32  = false,
    Stream              s       = 0,
    void*               a_had   = nullptr,
    bool                mcg     = false
);

// Batched / multi-matrix matmul (formerly exl3_mgemm). b_list, suh_list and svh_list are DEVICE
// arrays of pointers, one entry per quantized matrix; y is [bszm_out, M, N] (or the per-matrix
// widths and output pointers come from size_n_list / c_ptrs).
//
//   x            -> float16 [bszm_in, M, K]
//   y            -> float16 or float32 [bszm_out, M, N]
//   a_had        -> float16 scratch, room for bszm * M * K elements (one transformed slab per slot)
//   b_list       -> device array of trellis pointers, [num_matrices]
//   suh_list     -> device array of suh pointers, [num_matrices]
//   svh_list     -> device array of svh pointers, [num_matrices]
//   indices      -> device int64 [*, num_indices] of b_list slots; negative entries skip a slot
//   weights      -> device float16, parallel to indices; when given, every transformed slot result
//                   is scaled by weights[j] and all active slots are summed into y[0] (y doubles as
//                   per-expert scratch; only y[0] holds the reduced result)
//   min/max_index -> range filter, slots outside [min_index, max_index) are dropped and retained
//                   indices rebased by min_index (at most 128 slots, the kernel's compaction capacity)
//   num_tokens   -> with weights, split the slots into num_tokens contiguous groups, each reduced
//                   into its own output row
//   size_n_list / c_ptrs -> per-matrix output widths (int) and output pointers; mutually exclusive
//                   with weights / range filtering / multi-token mode. y then only supplies the
//                   element type and the maximum width used for lock and shape sizing.
//   a_had_elems  -> capacity of a_had in elements; 0 skips the capacity assertion
//
// Returns the selected shape index.
int mgemm
(
    void*                 y,
    const void*           x,
    const uint16_t**      b_list,
    const half**          suh_list,
    const half**          svh_list,
    int                   num_matrices,
    int                   M,
    int                   N,
    int                   K,
    int                   bits,
    bool                  y_fp32        = false,
    Stream                s             = 0,
    void*                 a_had         = nullptr,
    int                   bszm_in       = 1,
    int                   bszm_out      = 1,
    const int64_t*        indices       = nullptr,
    const half*           weights       = nullptr,
    int                   num_indices   = 0,
    int64_t               a_had_elems   = 0,
    int                   min_index     = -1,
    int                   max_index     = 0,
    int                   num_tokens    = 1,
    const int*            size_n_list   = nullptr,
    void**                c_ptrs        = nullptr,
    bool                  mcg           = false,
    bool                  mul1          = false,
    int                   force_shape_idx = 0,
    int                   force_num_sms = 0
);

} // namespace exl3
} // namespace helios