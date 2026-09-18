#pragma once
// Small-m GEMV dispatch (ported from exllamav3_ext/quant/exl3_gemv.cuh). The kernel lives in
// exl3_gemv_kernel.cuh and shares exl3_gemm_kernel's launch signature, so the kernel argument
// block built by gemm() can be handed over unchanged.
//
// Dropped versus the original: the EXL3_GEMV / EXL3_GEMV_SMEM environment knobs. The tuned
// defaults are now compile-time constants (MODE = 1 heuristic, SMEM_EXTRACTION = false, i.e.
// shuffle-based fragment extraction; the shared-memory staging kernel variants are not built).

#include "exl3_gemm.cuh"

namespace helios
{
namespace exl3
{
namespace gemv_detail
{

// Tuned defaults formerly read from the environment:
//   MODE: 0 disables the path, 1 = shape heuristic (default), 2 = use wherever the hard
//   constraints allow. 3/4 force narrow/wide for testing; retained for explicit overrides.
constexpr int MODE = 1;

// false = shuffle extraction (default), true = shared-memory staged extraction.
constexpr bool SMEM_EXTRACTION = false;

// -1: not eligible, 0: narrow config, 1: wide config.
int exl3_gemv_cfg
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    int mode,
    int narrow_coresident
);

void* exl3_gemv_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem);

} // namespace gemv_detail

// Attempts the GEMV path for the given kernel argument block. Returns true (and has launched)
// when the shape heuristic accepts; otherwise false and the caller runs the regular kernel.
// kernel_args is the EXL3_GEMM_ARGS parameter block: &A, &B, &C, &size_m, &size_k, &size_n,
// &locks, &suh, &A_had, &svh.
bool exl3_gemv_try_launch
(
    void**        kernel_args,
    int           size_m,
    int           size_k,
    int           size_n,
    int           K,
    int           cb,
    bool          c_fp32,
    bool          has_su_sv,
    int           device,
    Stream        s,
    void**        launched_kernel = nullptr,
    bool          force           = false
);

} // namespace exl3
} // namespace helios