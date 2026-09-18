// Small-m GEMV path (ported from exllamav3_ext/quant/exl3_gemv.cu). Kernel bodies untouched;
// only the launcher lost torch: at::Tensor -> raw pointers, TORCH_CHECK -> HELIOS_ASSERT,
// getCurrentCUDAStream() -> explicit Stream, getenv() knobs -> constexpr defaults.
//
/*
QTIP-style small-m GEMV path, kernel in exl3_gemv_kernel.cuh. Dispatched from gemm() via
exl3_gemv_try_launch when the shape heuristic applies, or forced through the gemv() entry point.
Kernel arguments are identical to exl3_gemm_kernel.

Heuristic envelope (measured, RTX 3090, 4 bpw, m <= 8, vs the tuned regular kernel): the narrow
config wins 15-60% at attention-projection sizes (n <= 4096), the wide config wins ~8% at
large-n/small-k FFN sizes. Big-k x big-n shapes lose slightly and fall through to the regular
kernel, as do other architectures (Ada/Blackwell are memory-bound here and keep the regular
kernel), bpw != 4, and m > 8.
*/

#include <cuda_fp16.h>
#include "exl3_gemv.cuh"

#include <cooperative_groups.h>
#include "helios_shim.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_devctx.cuh"
#include <map>

namespace helios
{
namespace exl3
{
namespace gemv_detail
{

// -1: not eligible, 0: narrow config, 1: wide config. narrow_coresident = number of narrow-config
// blocks that fit on the device at once (its grid is one block per 32 output columns)
int exl3_gemv_cfg(int cc, int size_m, int size_k, int size_n, int K, int cb, int mode, int narrow_coresident)
{
    if (mode == 0) return -1;
    if (K < 2 || K > 4) return -1;
    if (K != 4 && cb == 0) return -1;
    if (size_m > EXL3_GEMV_MAX_M) return -1;
    if (size_k % 128 || size_n % 128) return -1;
    //if (cc != CC_AMPERE) return -1;  // measured win on Ampere; Ada/Blackwell are memory-bound here
    if (mode == 2) return size_n <= 8192 ? 0 : 1;
    if (mode == 3) return 0;   // testing: force narrow config
    if (mode == 4) return 1;   // testing: force wide config

    // The narrow config wins (up to ~30%) whenever its grid fits in a single co-resident wave;
    // in the 1..2-wave zone the trailing partial wave costs more than the kernel gains unless
    // per-group work is small (small k). The wide config covers a band of large-n shapes with
    // small-to-mid k. Everything else runs the regular block-pipelined kernel.
    // Per-bits envelopes: 2 bpw is decode-bound and won at every measured shape on both archs;
    // 3 bpw wins everywhere on Ada but only in the narrow envelope on Ampere
    if (K == 2) return size_n <= 8192 ? 0 : 1;
    if (K == 3 && cc == CC_ADA) return size_n <= 8192 ? 0 : 1;
    if (size_n / 32 <= narrow_coresident) return 0;
    if (size_k <= 2048 && size_n <= 8192) return 0;
    if (K == 3) return -1;
    if (size_n >= 8192 && size_k <= 4096) return 1;
    if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120 && cc == CC_AMPERE) return 1;
    return -1;
}

void* exl3_gemv_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    // Pruned: the shared-memory extraction variants (smem = true) are not instantiated, since
    // the EXL3_GEMV_SMEM knob was replaced by the constexpr default gemv_detail::SMEM_EXTRACTION.
    // Passing smem = true here returns nullptr.
    SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
    SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    #undef SEL_GRID
    #undef SEL
    return nullptr;
}

} // namespace gemv_detail

bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
    cudaStream_t stream,
    void** launched_kernel,
    bool force
)
{
    // Free integer checks first; the device queries only run for calls that could take this path
    if (!has_su_sv) return false;
    if (K < 2 || K > 4) return false;
    if (K != 4 && cb == 0) return false;
    if (size_m > EXL3_GEMV_MAX_M) return false;
    if (size_k % 128 || size_n % 128) return false;

    int mode = force ? 2 : gemv_detail::MODE;
    if (mode == 0) return false;
    int cc = DevCtx::instance().get_cc(device);
    // if (cc != CC_AMPERE) return false;
    int mmode = size_m == 1 ? 0 : 1;
    int num_sms = DevCtx::instance().get_num_sms(device);

    // Cooperative launch: grids are capped at full co-residency (cached per kernel), and the
    // narrow config's co-residency also feeds the shape heuristic
    static std::map<void*, int> occ_cache[MAX_DEVICES];
    auto& cache = occ_cache[device];
    auto occupancy = [&] (void* kernel, int block_dim) -> int
    {
        auto it = cache.find(kernel);
        if (it != cache.end()) return it->second;
        int blocks_per_sm;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
        cache[kernel] = blocks_per_sm;
        return blocks_per_sm;
    };

    // Extraction style: shuffle (was the EXL3_GEMV_SMEM default), selectable per call for evaluation
    bool smem = gemv_detail::SMEM_EXTRACTION;

    void* narrow_kernel = gemv_detail::exl3_gemv_select_kernel(K, cb, c_fp32, mmode, 0, smem);
    if (!narrow_kernel) return false;
    int narrow_coresident = occupancy(narrow_kernel, 512) * num_sms;

    int cfg = gemv_detail::exl3_gemv_cfg(cc, size_m, size_k, size_n, K, cb, mode, narrow_coresident);
    if (cfg < 0) return false;

    void* kernel = cfg == 0 ? narrow_kernel : gemv_detail::exl3_gemv_select_kernel(K, cb, c_fp32, mmode, cfg, smem);
    if (!kernel) return false;

    int block_dim = cfg == 0 ? 512 : 256;
    int cols = cfg == 0 ? 32 : 64;

    int max_blocks = occupancy(kernel, block_dim) * num_sms;
    int grid = MIN(size_n / cols, max_blocks);
    if (grid < 1) return false;

    cuda_check(cudaLaunchCooperativeKernel
    (
        kernel,
        dim3(grid),
        dim3(block_dim),
        kernel_args,
        0,
        stream
    ));

    if (launched_kernel) *launched_kernel = kernel;
    return true;
}

void gemv
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
    bool                mcg
)
{
    HELIOS_ASSERT(w.trellis && x && y, "exl3_gemv: null tensor pointer");
    HELIOS_ASSERT(K % 16 == 0, "exl3_gemv: k must be divisible by 16");
    HELIOS_ASSERT(N % 128 == 0, "exl3_gemv: n must be divisible by 128");
    HELIOS_ASSERT(!(mcg && w.mul1), "Specified both mcg and mul1");
    HELIOS_ASSERT(w.suh && w.svh && a_had, "exl3_gemv requires suh, a_had and svh");

    const half* A_ptr     = (const half*) x;
    const uint16_t* B_ptr = w.trellis;
    void* C_ptr           = y;
    const half* suh_ptr   = w.suh;
    half* A_had_ptr       = (half*) a_had;
    const half* svh_ptr   = w.svh;

    int size_m = M;
    int size_k = K;
    int size_n = N;

    int cb = 0;
    if (mcg) cb = 1;
    if (w.mul1) cb = 2;

    int device;
    cuda_check(cudaGetDevice(&device));
    int* locks = DevCtx::instance().get_locks(device);

    void* kernel_args[] =
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

    bool ok = exl3_gemv_try_launch
    (
        kernel_args, size_m, size_k, size_n, bits, cb, y_fp32,
        true, device, s, nullptr, true
    );
    HELIOS_ASSERT(ok, "exl3_gemv: call is not eligible for the GEMV kernel");

    cuda_check(cudaPeekAtLastError());
}

} // namespace exl3
} // namespace helios