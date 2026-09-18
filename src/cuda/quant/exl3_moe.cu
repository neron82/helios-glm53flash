// Fused MoE launcher (ported from exllamav3_ext/quant/exl3_moe.cu). Kernel bodies untouched;
// only the wrapper lost torch: at::Tensor -> raw pointers + explicit extents, TORCH_CHECK ->
// HELIOS_ASSERT, getCurrentCUDAStream() -> explicit Stream, CUDAGuard -> caller sets the device.
//
// Pruned instantiations: the fixed-bitrate kernels for K = 1, 2, 5, 6, 7, 8 are not built; those
// requests fall back to the runtime-bitrate (K = 0) instantiation, which selects the bitrate per
// projection inside the kernel at a small performance cost. See README_PORT.md.

#include <cuda_fp16.h>
#include "exl3_moe.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "helios_shim.cuh"
#include "exl3_moe_instances.cuh"
#include "exl3_devctx.cuh"
#include <set>

namespace helios
{
namespace exl3
{
namespace
{

std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};

// [K][cb - 1][N_off]: K = 0 switches Kg/Ku/Kd at runtime, K > 0 = compile-time Kg = Ku = Kd
fp_exl3_moe_kernel moe_kernel_instances[] =
{
    exl3_moe_kernel_k0_n128_cb1(), exl3_moe_kernel_k0_n256_cb1(), exl3_moe_kernel_k0_n128_cb2(), exl3_moe_kernel_k0_n256_cb2(),
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    exl3_moe_kernel_k3_n128_cb1(), exl3_moe_kernel_k3_n256_cb1(), exl3_moe_kernel_k3_n128_cb2(), exl3_moe_kernel_k3_n256_cb2(),
    exl3_moe_kernel_k4_n128_cb1(), exl3_moe_kernel_k4_n256_cb1(), exl3_moe_kernel_k4_n128_cb2(), exl3_moe_kernel_k4_n256_cb2(),
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr
};

int current_device()
{
    int device = 0;
    cuda_check(cudaGetDevice(&device));
    return device;
}

} // namespace

int moe_max_concurrency(int device)
{
    int num_sms = DevCtx::instance().get_num_sms(device);
    return num_sms / MOE_SMS_PER_EXPERT;
}

void moe_grouped
(
    const void*         x,
    void*               y,
    const int64_t*      expert_count,
    const int64_t*      token_sorted,
    const half*         weight_sorted,
    void*               temp_state_g,
    void*               temp_state_u,
    void*               temp_intermediate_g,
    void*               temp_intermediate_u,
    const ExpertTables& tables,
    int                 tokens,
    int                 hidden_dim,
    int                 intermediate_dim,
    int                 num_experts,
    int                 topk,
    int                 max_tokens_per_expert,
    int                 concurrency,
    int                 K_gate,
    int                 K_up,
    int                 K_down,
    bool                mcg,
    bool                mul1,
    float               act_limit,
    int                 act_function,
    int                 num_active,
    Stream              s
)
{
    // Nothing for the fused kernel to do
    if (num_active == 0) return;

    // Validate args
    HELIOS_ASSERT(x && y && expert_count && token_sorted && weight_sorted, "exl3_moe: null pointer");
    HELIOS_ASSERT(tokens >= 1 && hidden_dim % 128 == 0 && intermediate_dim % 128 == 0,
                  "exl3_moe: hidden_dim and intermediate_dim must be multiples of 128");
    HELIOS_ASSERT(num_experts >= 1, "exl3_moe: num_experts must be >= 1");
    HELIOS_ASSERT(topk >= 1, "exl3_moe: topk must be >= 1");
    HELIOS_ASSERT(max_tokens_per_expert >= 1 && concurrency >= 1, "exl3_moe: bad scratch extents");
    HELIOS_ASSERT(tables.gate_trellis && tables.gate_suh && tables.gate_svh
               && tables.up_trellis   && tables.up_suh   && tables.up_svh
               && tables.down_trellis && tables.down_suh && tables.down_svh,
                  "exl3_moe: incomplete expert pointer table");

    HELIOS_ASSERT(mcg != mul1, "MoE kernel: Only mcg and mul1 codebooks are supported");
    const int cb_idx = mul1 ? 1 : 0;

    int K = 0;
    if (K_gate == K_up && K_up == K_down) K = K_gate;

    // Device properties
    int device = current_device();
    int num_sms = DevCtx::instance().get_num_sms(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Launch. All blocks of the grid must be co-resident for the group barriers, so groups * width <= num_sms.
    // With a known number of active experts, launch only as many groups as there are experts and widen them to
    // use the freed SMs, up to MOE_MAX_SMS_PER_EXPERT
    int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
    HELIOS_ASSERT(concurrency * MOE_SMS_PER_EXPERT <= num_sms, "Concurrency too high for device num_sms");
    int num_groups = MIN(concurrency, MOE_MAX_GROUPS);
    int group_size = MOE_SMS_PER_EXPERT;
    if (num_active > 0)
    {
        num_groups = MIN(num_groups, num_active);
        group_size = MIN(num_sms / num_groups, MOE_MAX_SMS_PER_EXPERT);
    }
    dim3 grid_dim(group_size, 1, num_groups);

    int N_off = 0;
    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;

    int sel = 4 * K + 2 * cb_idx + N_off;
    fp_exl3_moe_kernel kernel = moe_kernel_instances[sel];
    if (!kernel)
    {
        // Fixed-bitrate instance for this K was pruned from the build: use the runtime-bitrate
        // variant, which picks Kg/Ku/Kd inside the kernel. The table is [K][cb - 1][N_off], so the
        // K = 0 row is indexed by the CODEBOOK as well: picking [N_off] alone would silently select
        // the mcg kernel for mul1 weights and decode garbage.
        HELIOS_ASSERT(K >= 1 && K <= 8, "exl3_moe: bit width out of range");
        kernel = moe_kernel_instances[2 * cb_idx + N_off];
    }
    HELIOS_ASSERT(kernel, "exl3_moe: no kernel instance");

    if (moe_kernel_attr_set[device].find((void*) kernel) == moe_kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        moe_kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }

    const half* hidden_state_ptr = (const half*) x;
    half* temp_state_g_ptr = (half*) temp_state_g;
    half* temp_state_u_ptr = (half*) temp_state_u;
    half* temp_intermediate_g_ptr = (half*) temp_intermediate_g;
    half* temp_intermediate_u_ptr = (half*) temp_intermediate_u;
    float* output_state_ptr = (float*) y;

    void* kernelArgs[] =
    {
        &hidden_state_ptr,
        &temp_state_g_ptr,
        &temp_state_u_ptr,
        &temp_intermediate_g_ptr,
        &temp_intermediate_u_ptr,
        &output_state_ptr,
        (void*)& tables.gate_trellis,
        (void*)& tables.gate_suh,
        (void*)& tables.gate_svh,
        (void*)& tables.up_trellis,
        (void*)& tables.up_suh,
        (void*)& tables.up_svh,
        (void*)& tables.down_trellis,
        (void*)& tables.down_suh,
        (void*)& tables.down_svh,
        (void*)& expert_count,
        (void*)& token_sorted,
        (void*)& weight_sorted,
        (void*)& hidden_dim,
        (void*)& intermediate_dim,
        (void*)& num_experts,
        (void*)& topk,
        (void*)& max_tokens_per_expert,
        (void*)& num_groups,
        (void*)& act_limit,
        (void*)& act_function,
        (void*)& K_gate,
        (void*)& K_up,
        (void*)& K_down,
        (void*)& locks
    };

    // Fit the dynamic smem request to the device (static + dynamic <= opt-in limit).
    size_t smem_launch = SMEM_MAX;
    {
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, current_device());
        cudaFuncAttributes fa{};
        cudaFuncGetAttributes(&fa, (void*) kernel);
        size_t avail = (size_t)optin > fa.sharedSizeBytes ? (size_t)optin - fa.sharedSizeBytes : 0;
        if (smem_launch > avail) smem_launch = avail;
    }
    // The kernel requests SMEM_MAX dynamic shared memory (> 48KB), which requires opting in.
    {
        cudaError_t ae = cudaFuncSetAttribute((void*) kernel,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, smem_launch);
        if (ae != cudaSuccess)
            fprintf(stderr, "[exl3_moe] cudaFuncSetAttribute(%d) failed: %s\n", (int)SMEM_MAX,
                    cudaGetErrorString(ae));
    }
    cudaError_t le = cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        smem_launch,
        s
    );
    if (le != cudaSuccess)
    {
        // diagnostic probes: which launch parameter is rejected?
        (void)cudaGetLastError();
        cudaError_t e1 = cudaLaunchKernel((void*)kernel, dim3(1,1,1), block_dim, kernelArgs, SMEM_MAX, s);
        (void)cudaGetLastError();
        cudaError_t e2 = cudaLaunchKernel((void*)kernel, dim3(1,1,1), dim3(256,1,1), kernelArgs, SMEM_MAX, s);
        (void)cudaGetLastError();
        cudaError_t e3 = cudaLaunchKernel((void*)kernel, grid_dim, dim3(256,1,1), kernelArgs, SMEM_MAX, s);
        fprintf(stderr, "[exl3_moe] probe: 1block/512t=%s 1block/256t=%s fullgrid/256t=%s\n",
                cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3));
        (void)cudaGetLastError();
        // cross-device probe: does the same launch work on device 0?
        int orig_dev = -1; cudaGetDevice(&orig_dev);
        cudaSetDevice(0);
        cudaError_t e0 = cudaLaunchKernel((void*)kernel, dim3(1,1,1), dim3(256,1,1), kernelArgs, SMEM_MAX, s);
        fprintf(stderr, "[exl3_moe] probe on dev0 (stream from dev%d): %s\n", orig_dev, cudaGetErrorString(e0));
        (void)cudaGetLastError();
        cudaSetDevice(orig_dev);
        {
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            cudaError_t qe = cudaStreamIsCapturing(s, &cs);
            cudaError_t qe2 = cudaStreamQuery(s);
            int sdev = -1;
            cudaStreamAttrValue av{};
            fprintf(stderr, "[exl3_moe] stream=%p is_capturing=%s query=%s cap_status=%d\n",
                    (void*)s, cudaGetErrorString(qe), cudaGetErrorString(qe2), (int)cs);
            (void)sdev; (void)av;
            (void)cudaGetLastError();
        }
        // warm-up: plain launch with this kernel on the current device, stream 0 (legacy)
        cudaError_t ew = cudaLaunchKernel((void*)kernel, dim3(1,1,1), dim3(256,1,1), kernelArgs, SMEM_MAX, 0);
        fprintf(stderr, "[exl3_moe] warm-up on dev%d legacy stream: %s\n", orig_dev, cudaGetErrorString(ew));
        (void)cudaGetLastError();
        int dev_now = -1; cudaGetDevice(&dev_now);
        cudaFuncAttributes fa{};
        cudaFuncGetAttributes(&fa, (void*) kernel);
        fprintf(stderr, "[exl3_moe] launch failed: grid=(%d,%d,%d) block=%d smem=%d dev=%d "
                        "static=%zu maxdyn=%zu regs=%d err=%s\n",
                grid_dim.x, grid_dim.y, grid_dim.z, block_dim, (int)SMEM_MAX, dev_now,
                fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes, fa.numRegs,
                cudaGetErrorString(le));
    }
    cuda_check(le);
}

} // namespace exl3
} // namespace helios