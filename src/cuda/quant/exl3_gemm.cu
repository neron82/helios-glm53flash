// EXL3 matmul launchers (ported from exllamav3_ext/quant/exl3_gemm.cu).
//
// Kernel bodies and the shape-selection heuristics are byte-exact; only the wrappers changed:
//   at::Tensor / c10::optional<at::Tensor>  -> raw device pointers + explicit extents
//   TORCH_CHECK / TORCH_CHECK_*             -> HELIOS_ASSERT
//   at::cuda::getCurrentCUDAStream()        -> explicit Stream argument (last, default 0)
//   cudaGetDevice()                         -> kept (the caller sets the device; no CUDAGuard)
//   Graph* capture parameter patching       -> dropped (helios has no CUDA-graph capture layer)
//   CoopKernelAutotuner                     -> dropped; the deterministic selector is the only
//                                              path (formerly the force_shape_idx/force_num_sms path)
//   experimental fused int8-activation GEMV -> dropped (see README_PORT.md)
// Cooperative launch is preserved: the kernels use cg::this_grid()/grid.sync(), so they must be
// launched with cudaLaunchCooperativeKernel and a co-resident grid.

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float32, contiguous. Does not need
  to be zero-initialized
- suh: required, packed input scales/flips, shape (k//16), dtype float16
- A_had: required with suh, temporary storage for the input transform, size and dtype as A
- svh: required, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "helios_shim.cuh"
#include "exl3_gemm_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "exl3_gemv.cuh"
#include <set>

namespace helios
{
namespace exl3
{

namespace
{

std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

// One cudaFuncSetAttribute per kernel per device (dynamic smem is always SMEM_MAX).
void set_kernel_attr(int device, void* kernel)
{
    // The attribute is per-context, so re-apply it on every call (cheap) instead of caching by
    // device: a stale or missing attribute caps dynamic smem at 48KB and every launch of these
    // kernels fails (occupancy 0 -> "too many blocks in cooperative launch" / "invalid argument").
    (void)device;
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
    if (e != cudaSuccess)
    {
        int smem_attr = 0, optin = 0;
        cudaFuncGetAttributes(nullptr, kernel);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        fprintf(stderr, "[exl3_gemm] cudaFuncSetAttribute(%d) failed on dev %d: %s (optin=%d)\n",
                (int)SMEM_MAX, device, cudaGetErrorString(e), optin);
        (void)smem_attr;
    }
}

int current_device()
{
    int device = 0;
    cuda_check(cudaGetDevice(&device));
    return device;
}

} // namespace

int gemm
(
    void*               y,
    const void*         x,
    const GroupWords&   w,
    int                 M,
    int                 N,
    int                 K,
    int                 bits,
    bool                y_fp32,
    Stream              s,
    void*               a_had,
    bool                mcg,
    int                 force_shape_idx,
    int                 force_num_sms
)
{
    HELIOS_ASSERT(x && y && w.trellis, "exl3_gemm: null tensor pointer");
    HELIOS_ASSERT(M >= 1, "exl3_gemm: m must be >= 1");
    HELIOS_ASSERT(K % 16 == 0, "exl3_gemm: k must be divisible by 16");
    HELIOS_ASSERT(N % 128 == 0, "exl3_gemm: n must be divisible by 128");
    HELIOS_ASSERT(!(mcg && w.mul1), "Specified both mcg and mul1");
    // The kernel's Hadamard prologue/epilogue dereference suh / svh unconditionally, so unlike
    // the original (where the optional tensors could in principle be absent) all three are
    // mandatory here.
    HELIOS_ASSERT(w.suh && w.svh && a_had,
        "exl3_gemm requires suh, svh and a_had (room for m * k float16): the kernel applies the "
        "input Hadamard transform into a_had and the output transform + svh unconditionally");

    int device = current_device();
    int num_sms = force_num_sms ? force_num_sms : DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);
    if (const char* lw = getenv("HELIOS_LOCKS_EXTRA")) {
        static int* big_locks = nullptr;
        size_t extra = (size_t)atoll(lw) * 1024 * 1024;
        if (!big_locks) {
            cudaSetDevice(device);
            cudaMalloc(&big_locks, extra);
            cudaMemset(big_locks, 0, extra);
        }
        locks = big_locks;
    }

    // Dispatch
    int cb = 0;
    if (mcg) cb = 1;
    if (w.mul1) cb = 2;

    const half* A_ptr = (const half*) x;
    const uint16_t* B_ptr = w.trellis;
    void* C_ptr = y;
    const half* suh_ptr = w.suh;
    half* A_had_ptr = (half*) a_had;
    const half* svh_ptr = w.svh;

    const int size_m = M;
    const int size_k = K;
    const int size_n = N;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    // QTIP-style GEMV path for small m (exl3_gemv_kernel.cuh). Same kernel arguments, so it is a
    // drop-in; falls through to the regular kernel when the heuristic declines.
    if (force_shape_idx <= 0 && force_num_sms <= 0)
    {
        if (exl3_gemv_try_launch
        (
            kernelArgs, size_m, size_k, size_n, bits, cb, y_fp32,
            true, device, s, nullptr, false
        ))
        {
            cuda_check(cudaPeekAtLastError());
            return 90;
        }
    }

    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel = select_exl3_gemm_kernel
    (
        cc, size_m, size_k, size_n, bits, y_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb
    );

    // Fit the dynamic smem request to the device: static + dynamic must stay within the
    // per-block opt-in limit, otherwise the launch fails (cooperative: "too many blocks",
    // plain: "invalid argument") because occupancy computes to zero.
    size_t smem_launch = SMEM_MAX;
    {
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        cudaFuncAttributes fa{};
        cudaFuncGetAttributes(&fa, (void*) kernel);
        size_t avail = (size_t)optin > fa.sharedSizeBytes ? (size_t)optin - fa.sharedSizeBytes : 0;
        if (smem_launch > avail) smem_launch = avail;
        if (fa.sharedSizeBytes + smem_launch > (size_t)optin) {
            fprintf(stderr, "[exl3_gemm] shape %d needs static %zu + %zu dynamic > optin %d\n",
                    shape_idx, fa.sharedSizeBytes, smem_launch, optin);
        }
    }
    set_kernel_attr(device, (void*) kernel);

    // Cooperative launches must fit co-resident: clamp the grid to the occupancy limit so a
    // shape whose smem footprint allows only one block/SM (or fewer) still launches.
    {
        int max_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_per_sm, (void*) kernel, block_dim, smem_launch);
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        int limit = MAX(1, max_per_sm * sm_count);
        if (num_sms > limit) num_sms = limit;
    }

    cudaError_t le = cudaErrorInvalidValue;
    for (int blocks = num_sms; blocks >= 1; blocks = (blocks > 1 ? blocks / 2 : 0))
    {
        (void)cudaGetLastError();
        le = cudaLaunchCooperativeKernel((void*) kernel, blocks, block_dim, kernelArgs, smem_launch, s);
        if (le == cudaSuccess) break;
        if (le != cudaErrorCooperativeLaunchTooLarge) break;
    }
    if (le != cudaSuccess)
    {
        // The kernels contain no grid-wide synchronisation (no cooperative_groups usage), so a
        // plain launch is a valid fallback when the driver refuses the cooperative grid size.
        (void)cudaGetLastError();
        le = cudaLaunchKernel((void*) kernel, num_sms, block_dim, kernelArgs, smem_launch, s);
        if (le == cudaSuccess) return shape_idx;
        int max_per_sm = 0, sm_count = 0, dev_now = -1, static_smem = 0, dev_smem = 0, dev_smem_optin = 0;
        cudaFuncAttributes fa2{};
        cudaFuncGetAttributes(&fa2, (void*)kernel);
        cudaGetDevice(&dev_now);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_per_sm, (void*) kernel, block_dim, smem_launch);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_now);
        cudaFuncAttributes fa{};
        cudaFuncGetAttributes(&fa, (void*)kernel);
        static_smem = (int)fa.sharedSizeBytes;
        cudaDeviceGetAttribute(&dev_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_now);
        cudaDeviceGetAttribute(&dev_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_now);
        fprintf(stderr, "[exl3_gemm] coop launch failed: M=%d N=%d K=%d bits=%d shape=%d blocks=%d "
                        "block_dim=%d smem=%zu err=%s | dev=%d(sel %d) sms=%d max_per_sm=%d "
                        "smem_per_sm=%d smem_optin=%d static=%d maxdyn=%d regs=%d\n",
                size_m, size_n, size_k, bits, shape_idx, (int)num_sms, block_dim, smem_launch,
                cudaGetErrorString(le), dev_now, device, sm_count, max_per_sm, dev_smem,
                dev_smem_optin, static_smem, (int)fa2.maxDynamicSharedSizeBytes, fa2.numRegs);
    }
    cuda_check(le);
    return shape_idx;
}

/*
EXL3 batched/multi-matrix matmul.

This is not a conventional batched A @ B. b_list, suh_list and svh_list are device arrays of
addresses (one address per quantized matrix), rather than the matrix data themselves. Entry q of
each table describes one linear:

    b_list[q]   -> EXL3 trellis, logically (k / 16, n / 16, 16 * bits) uint16
    suh_list[q] -> packed input scales/flips, logically (k / 16) float16
    svh_list[q] -> packed output scales/flips, logically (n / 16) float16

x is contiguous float16 [bszm_in, m, k], y is contiguous float16 or float32
[bszm_out, m, n], and a_had is float16 scratch with room for every active matrix. The kernel
applies the input Hadamard transform into a_had, performs the selected EXL3 matmul, then applies
the output transform.

The active matrix/output slot j selects q = indices[j] when indices is given, or q = j otherwise.
This supports the following modes:

- Multiple inputs and outputs: x[j] @ b_list[q] -> y[j].
- One input, multiple outputs: when bszm_in == 1, x[0] is broadcast and transformed separately for
  each selected b_list[q], producing y[j]. Used for e.g. fused gate/up projections.
- Indexed matrices: indices is a contiguous device int64 array of num_indices entries; negative
  indices skip that slot.
- Weighted MoE reduction: weights is float16, parallel to indices. Each transformed result is
  multiplied by weights[j], then all active y[j] are summed into y[0]. y therefore also serves as
  per-expert scratch; only y[0] is the reduced result.
- Expert-range filtering: with min_index >= 0, selections outside [min_index, max_index) are
  removed and retained indices are rebased by min_index. At num_tokens == 1 the retained indices
  (and their weights) are compacted; at num_tokens > 1 out-of-range slots are instead masked to -1
  in place, preserving the per-token slot groups the final reduction depends on.

Without weights, every active y[j] is a separate output. The active slot count is
max(bszm_in, bszm_out), capped to num_indices when indices is present.

Limitations: k must be divisible by 16 and n by 128. Range filtering supports at most 128 slots
(the kernel's index-compaction capacity).
*/

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
    bool                  y_fp32,
    Stream                s,
    void*                 a_had,
    int                   bszm_in,
    int                   bszm_out,
    const int64_t*        indices,
    const half*           weights,
    int                   num_indices,
    int64_t               a_had_elems,
    int                   min_index,
    int                   max_index,
    int                   num_tokens,
    const int*            size_n_list,
    void**                c_ptrs,
    bool                  mcg,
    bool                  mul1,
    int                   force_shape_idx,
    int                   force_num_sms
)
{
    HELIOS_ASSERT(x && y && b_list && suh_list && svh_list && a_had, "exl3_mgemm: null pointer");
    HELIOS_ASSERT(num_matrices > 0, "exl3_mgemm: empty pointer table");
    HELIOS_ASSERT(!(mcg && mul1), "exl3_mgemm: Specified both mcg and mul1");

    if (size_n_list)
    {
        HELIOS_ASSERT(c_ptrs, "exl3_mgemm: size_n_list requires c_ptrs");
        HELIOS_ASSERT(num_tokens == 1 && min_index < 0 && !weights,
                    "exl3_mgemm: per-matrix widths incompatible with multi-token/filtering/weights");
        bszm_out = num_matrices;
    }
    int bszm = MAX(bszm_in, bszm_out);

    // The kernel writes one hadamard-transformed input slab PER MATRIX (a_had + j * m * k);
    // an undersized scratch is silent OOB corruption (found the hard way)
    if (a_had_elems > 0)
        HELIOS_ASSERT(a_had_elems >= (int64_t) bszm * M * K,
                      "exl3_mgemm: a_had must hold bszm * m * k elements");

    if (indices)
    {
        HELIOS_ASSERT(num_indices > 0, "exl3_mgemm: num_indices must be given with indices");
        HELIOS_ASSERT(num_indices <= bszm_in || num_indices <= bszm_out,
                      "mgemm: too many indices for batch");
        if (bszm_in > num_indices) bszm_in = num_indices;
        if (bszm_out > num_indices) bszm_out = num_indices;
        bszm = MAX(bszm_in, bszm_out);
    }

    const int size_m = M;
    const int size_k = K;
    const int size_n = N;

    // Device properties
    int device = current_device();
    int total_sms = DevCtx::instance().get_num_sms(device);
    int num_sms = force_num_sms ? force_num_sms : total_sms;
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);
    if (const char* lw = getenv("HELIOS_LOCKS_EXTRA")) {
        static int* big_locks = nullptr;
        size_t extra = (size_t)atoll(lw) * 1024 * 1024;
        if (!big_locks) {
            cudaSetDevice(device);
            cudaMalloc(&big_locks, extra);
            cudaMemset(big_locks, 0, extra);
        }
        locks = big_locks;
    }

    const half* A_ptr     = (const half*) x;
    void* C_ptr           = y;
    half* A_had_ptr       = (half*) a_had;
    const int64_t* indices_ptr = indices;
    const half* weights_ptr    = weights;

    // Dispatch
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int shape_idx;
    int block_dim;
    fp_exl3_mgemm_kernel kernel = select_exl3_mgemm_kernel
    (
        cc, size_m, size_k, size_n, bits, y_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb, bszm_in, bszm_out
    );
    int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
    int tiles = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
    num_sms = tiles;
    if (num_sms * bszm > total_sms) num_sms = MAX(total_sms / bszm, 1);
    if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
    int concurrency = MIN(total_sms / num_sms, bszm);

    // Launch bigger grid if possible
    dim3 block_grid(num_sms, 1, concurrency);

    // Fit the dynamic smem request to the device: static + dynamic must stay within the
    // per-block opt-in limit, otherwise the launch fails (cooperative: "too many blocks",
    // plain: "invalid argument") because occupancy computes to zero.
    size_t smem_launch = SMEM_MAX;
    {
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        cudaFuncAttributes fa{};
        cudaFuncGetAttributes(&fa, (void*) kernel);
        size_t avail = (size_t)optin > fa.sharedSizeBytes ? (size_t)optin - fa.sharedSizeBytes : 0;
        if (smem_launch > avail) smem_launch = avail;
        if (fa.sharedSizeBytes + smem_launch > (size_t)optin) {
            fprintf(stderr, "[exl3_gemm] shape %d needs static %zu + %zu dynamic > optin %d\n",
                    shape_idx, fa.sharedSizeBytes, smem_launch, optin);
        }
    }
    set_kernel_attr(device, (void*) kernel);

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& b_list,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_list,
        (void*)& A_had_ptr,
        (void*)& svh_list,
        (void*)& indices_ptr,
        (void*)& weights_ptr,
        (void*)& bszm_in,
        (void*)& bszm_out,
        (void*)& min_index,
        (void*)& max_index,
        (void*)& num_tokens,
        (void*)& size_n_list,
        (void*)& c_ptrs
    };

    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        block_grid,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        s
    );

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

} // namespace exl3
} // namespace helios