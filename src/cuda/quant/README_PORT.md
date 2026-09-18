# EXL3 Quantization Core Port — README_PORT.md

This directory contains a standalone C++20/CUDA port of the EXL3 quantization
core from `exllamav3/exllamav3_ext/`. Kernel bodies are preserved byte-exact;
only torch-dependent launcher wrappers, autotuning, and graph-capture
infrastructure have been replaced with raw-pointer C++ APIs and explicit
stream parameters.

## File Mapping

| Original (exllamav3_ext) | Ported (helios/src/cuda/quant) | Notes |
|--------------------------|-------------------------------|-------|
| `quant/codebook.cuh` | `codebook.cuh` | Preserved; host/device macros adjusted |
| `quant/exl3_dq.cuh` | `exl3_dq.cuh` | Preserved; host/device macros adjusted |
| `quant/pack.cu/.cuh` | `pack.cu/.cuh` | Wrapped in `namespace helios::exl3`; kernel body unchanged |
| `quant/reconstruct.cu/.cuh` | `reconstruct.cu/.cuh` | Wrapped; kernel body unchanged |
| `quant/hadamard.cu/.cuh` | `hadamard.cu/.cuh` | Wrapped; kernels unchanged |
| `quant/hadamard_inner.cuh` | `hadamard_inner.cuh` | Copied verbatim |
| `quant/util.cu/.cuh` | `util.cu/.cuh` | Wrapped; kernels unchanged |
| `quant/ptx.cuh` | `ptx.cuh` | Copied verbatim |
| `quant/reduction.cuh` | `reduction.cuh` | Copied verbatim |
| `quant/exl3_devctx.cu/.cuh` | `exl3_devctx.cu/.cuh` | Wrapped; host-only implementation |
| `quant/exl3_kernel_map.cu/.cuh` | `exl3_kernel_map.cu/.cuh` | Pruned: bit widths 1,7; GEMM shape 1 |
| `quant/exl3_gemm.cu/.cuh` | `exl3_gemm.cu/.cuh` | Raw-pointer launcher; no autotune |
| `quant/exl3_gemm_inner.cuh` | `exl3_gemm_inner.cuh` | Copied verbatim |
| `quant/exl3_gemm_kernel.cuh` | `exl3_gemm_kernel.cuh` | Copied verbatim |
| `quant/exl3_gemv.cu/.cuh` | `exl3_gemv.cu/.cuh` | Raw-pointer launcher; no env-var mode selection |
| `quant/exl3_gemv_kernel.cuh` | `exl3_gemv_kernel.cuh` | Copied verbatim |
| `quant/exl3_moe.cu/.cuh` | `exl3_moe.cu/.cuh` | Raw-pointer launcher; pruned K=1,2,5,6,7,8 fixed-bitrate |
| `quant/exl3_moe_kernel.cuh` | `exl3_moe_kernel.cuh` | Copied verbatim |
| `compat.cuh` (selected) | `helios_shim.cuh` | CEIL_DIVIDE, half types, tanh_opt, MAX/MIN |
| `ptx.cuh` (selected) | `ptx.cuh` | Atomic intrinsics only |

## Public API (namespace helios::exl3)

All launchers take explicit raw pointers, integer dimensions, and an optional
`Stream` parameter (default = stream 0). No torch/ATen types.

```cpp
// Packing / reconstruction
void pack_trellis(uint16_t* pack, const uint16_t* idx, int rows, int cols, int bits, Stream s = 0);
void unpack_trellis(uint16_t* idx, const uint16_t* pack, int rows, int cols, int bits, Stream s = 0);
void reconstruct(half* bhat, const uint16_t* pack, int rows, int cols, int n, int bits, bool mcg, bool mul1, Stream s = 0);

// Hadamard transforms (128-point, in-place)
void suh(half* a, int m, int k, Stream s = 0);
void svh(half* c, int m, int n, Stream s = 0);

// GEMM: C = A @ B_quant (fp32 output)
int gemm(float* c, const half* a, const GroupWords& w, int m, int n, int k,
         int bits, bool y_fp32 = true, Stream s = 0,
         half* a_had = nullptr, bool mcg = false,
         int force_shape_idx = 0, int force_num_sms = 0);

// MGEMM: batched GEMM with expert table
void mgemm(...);  // see exl3_moe.cuh for full signature

// GEMV: single-row GEMM (fast path for M=1)
void gemv(half* c, const half* a, const GroupWords& w, int n, int k,
          int bits, bool mcg, Stream s = 0);

// MoE grouped dispatch
int moe_max_concurrency(DevCtx* dev, int expert_count, int expert_group_size);
void moe_grouped(...);  // see exl3_moe.cuh for full signature

// Kernel map (internal, exposed for MGEMM/MoE)
void* get_gemm_kernel_ptr(int cc, int m, int k, int n, int bits, bool multi, int shape_idx);
void* get_mgemm_kernel_ptr(int cc, int k, int n, int bits, bool multi, int shape_idx);
int select_gemm_shape(int cc, int m, int k, int n, int bits, bool multi, int bszm_in, int bszm_out);
```

## Trellis Layout

Logical shape: `(K / 16, N / 16, 16 * bits)` where:
- First dimension = K / 16 "trellis rows"
- Second dimension = N / 16 "trellis columns"
- Third dimension = 16 * bits bits of codebook indices per weight group

Each 16x16 weight group stores `16 * 16 * bits` bits packed into `256 * bits / 16` uint16 words.

## Codebook Semantics

- `mcg = true`: Use codebook 1 (mcg variant)
- `mcg = false, mul1 = false`: Use codebook 0 (default)
- `mcg = false, mul1 = true`: Use codebook 2 (mul1 variant)
- `reconstruct(..., mcg, mul1, ...)` decodes the trellis back to fp16 weight matrix

## Hadamard Transform Requirements

GEMM/GEMV kernels assume the following transform pipeline:

```
Effective weight: W_eff = diag(suh) · H128 · W_hat · H128 · diag(svh)
```

Where:
- `H128` = 128-point Fast Walsh-Hadamard Transform (applied in groups of 128)
- `suh` = fp16 input-side scaling vector of length K
- `svh` = fp16 output-side scaling vector of length N
- `W_hat` = reconstructed weight matrix (no transforms)

The GEMM launcher internally:
1. Calls `suh(a)` — scales A by suh
2. Calls `a_had` transform (if `a_had != nullptr`)
3. Runs the quantized GEMM kernel (which internally handles B-side transforms)
4. Calls `svh(c)` — scales output by svh

The Hadamard transform scale factor is `RS = 0.08838834764831845f` (applied as `1/sqrt(128)` per dimension).

## Stream and Scratch Requirements

- All launchers accept an optional `Stream` (cudaStream_t) parameter. Default is stream 0.
- GEMM kernels require cooperative launch (`cudaLaunchCooperativeKernel`) and use
  `cg::this_grid()` / `grid.sync()` internally. The launcher handles this.
- `a_had` scratch buffer must be provided by caller for GEMM when `a_had != nullptr`
  (size: `M * K` fp16 elements).
- GEMV uses shuffle-based shared-memory extraction by default (compile-time `MODE = 1`,
  `SMEM_EXTRACTION = false`).

## Pruned Components

The following original components were intentionally dropped from this port:

1. **Cooperative autotuner** (`coop_autotune.cu/.cuh`) — deterministic selector path used instead
2. **Graph capture variants** (`graph.cuh`) — no torch graph layer
3. **GEMM shape 1** — unused on target path; shape 2 covers all required dimensions
4. **Bit widths 1 and 7** — no comp-unit TUs compiled; selector tables have null entries
5. **GEMV shared-memory extraction variants** — environment-controlled variants dropped
6. **int8 GEMV path** (`exl3_gemv_int8.cu`) — environment/graph-heavy, out of scope
7. **MoE fixed-bitrate K=1,2,5,6,7,8** — only K=0,3,4 instantiated; others fall back to K=0 runtime-bitrate kernel
8. **Environment variable mode selection** — compile-time constants only

## Build

```bash
cd src/cuda/quant
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

Requires: CUDA 13.0 (`/usr/local/cuda-13.0`), CMake 3.28, gcc 13. Target: sm_86 (RTX 3090).

Output: `build/libexl3_quant_core.a` (static library) + test binaries.

## Testing

```bash
./build/test_reconstruct   # pack/unpack/reconstruct round-trip
./build/test_gemm_smoke    # GEMM/GEMV accuracy vs host reference
```

GEMM smoke test validates:
- GEMV dispatch (M=1) vs GEMM dispatch (M=64)
- bits=2 and bits=4 codebook paths
- Relative RMS error < 5e-2 (observed ~4e-4)

## Host Test Scaffolding

`test/host_ptx.hpp` provides host-side mirrors of device intrinsics:
- `fshift`, `bfe16`, `dp4a`, `fadd`, `fmul`, `fadd_sat`, `fmul_sat` (fp16)
- `f32_to_f16`, `f16_to_f32`, `h16_to_f32`, `f32_to_h16` (conversion)

Used by `test_reconstruct` and `test_gemm_smoke` for host-side reference computations.

## Deviations from Original

1. **Autotuning replaced by deterministic selector** — `select_gemm_shape` uses fixed
   logic instead of runtime profiling
2. **No graph capture** — kernels launch directly via `cudaLaunchKernel`/`cudaLaunchCooperativeKernel`
3. **Compile-time constants** — `MODE`, `SMEM_EXTRACTION`, `FORCE_CG` are `constexpr`
4. **Raw pointers instead of torch tensors** — all `at::Tensor` arguments replaced with
   pointer + explicit dimension parameters
5. **No device-side codebook tables** — codebook lookups use the same shared memory
   layout as the original kernels
