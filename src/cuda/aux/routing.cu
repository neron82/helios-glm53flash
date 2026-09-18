// Ported from exllamav3 exllamav3_ext/routing.cu. Kernel bodies unchanged; launchers take
// raw pointers + explicit dimensions + an explicit stream. The std-softmax routing,
// routing_sel_norm and moe_split_* paths are not part of the helios port and were dropped;
// so was the cublas hgemm fallback (helios has no cuBLAS here), which means the caller must
// supply scores for bsz > 1 (routing_gemv covers the decode bsz == 1 path).
#include "routing.cuh"
#include "helios_shim.cuh"
#include <cstddef>

namespace helios { namespace aux {

#define MAX_NUM_EXPERTS 512
#define MAX_K 16

using bfloat16 = __nv_bfloat16;

__device__ __forceinline__
float sigmoid_stable_hf(float xf)
{
    float ez = __expf(-fabsf(xf));
    float base = ez / (1.0f + ez);
    return (xf >= 0.0f) ? 1.0f - base : base;
}

// Score activations for the nogroup top-k kernels (RoutingAct in routing.cuh). Both are
// strictly increasing, so sorting by the raw logit when there is no selection bias remains
// valid for either

template <int ACT>
__device__ __forceinline__
float routing_act(float xf)
{
    if constexpr (ACT == ROUTING_ACT_SQRTSP)
    {
        // sqrt(softplus(x)), matching torch F.softplus(beta = 1, threshold = 20)
        float sp = xf > 20.0f ? xf : log1pf(__expf(xf));
        return sqrtf(sp);
    }
    else
        return sigmoid_stable_hf(xf);
}


__device__ __forceinline__
void warp_reduce_best_f32(float& key, float& payload, int& idx)
{
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
        // Monotonic unsigned encoding of the float key, hardware max-reduce, then fetch the
        // winner's values from the lowest tied lane
        unsigned int ku = __float_as_uint(key);
        ku = (ku & 0x80000000u) ? ~ku : (ku | 0x80000000u);
        unsigned int m = __reduce_max_sync(0xffffffffu, ku);
        int src = __ffs(__ballot_sync(0xffffffffu, ku == m)) - 1;
        key = __shfl_sync(0xffffffffu, key, src);
        payload = __shfl_sync(0xffffffffu, payload, src);
        idx = __shfl_sync(0xffffffffu, idx, src);
    #else
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_key = __shfl_down_sync(0xffffffffu, key, offset);
            float other_payload = __shfl_down_sync(0xffffffffu, payload, offset);
            int other_idx = __shfl_down_sync(0xffffffffu, idx, offset);
            if (other_key > key)
            {
                key = other_key;
                payload = other_payload;
                idx = other_idx;
            }
        }

        key = __shfl_sync(0xffffffffu, key, 0);
        payload = __shfl_sync(0xffffffffu, payload, 0);
        idx = __shfl_sync(0xffffffffu, idx, 0);
    #endif
}


__device__ __forceinline__
void warp_radixsort_posf16(half& key, int& idx, int* src_lane_map)
{
    unsigned int lane_id = threadIdx.x % 32;
    const unsigned int active = 0xffffffffu;

    unsigned int ku = __half_as_ushort(key);

    #pragma unroll
    for (int bit = 0; bit < 15; ++bit)
    {
        unsigned int b = (ku >> bit) & 1;
        unsigned int ones = __ballot_sync(active, b);
        unsigned int zeros = active ^ ones;
        int nzeros = __popc(zeros);

        unsigned int below = (1 << lane_id) - 1;
        int r0 = __popc(zeros & below);
        int r1 = __popc(ones & below);

        int dest = b ? (nzeros + r1) : r0;
        int myrank = __popc(active & below);

        src_lane_map[dest] = lane_id;
        __syncwarp(active);
        int src = src_lane_map[myrank];

        ku = __shfl_sync(active, ku, src);
        idx = __shfl_sync(active, idx, src);
    }
    key = __ushort_as_half(ku);
}


__device__ __forceinline__
void warp_radixsort_posf32_pl(float& key, float& payload, int& idx, int* src_lane_map)
{
    unsigned int lane_id = threadIdx.x % 32;
    const unsigned int active = 0xffffffffu;

    unsigned int ku = __float_as_uint(key);

    #pragma unroll
    for (int bit = 0; bit < 31; ++bit)
    {
        unsigned int b = (ku >> bit) & 1u;
        unsigned int ones = __ballot_sync(active, b);
        unsigned int zeros = active ^ ones;
        int nzeros = __popc(zeros);

        unsigned int below = (1u << lane_id) - 1u;
        int r0 = __popc(zeros & below);
        int r1 = __popc(ones & below);

        int dest = b ? (nzeros + r1) : r0;
        int myrank = __popc(active & below);

        src_lane_map[dest] = lane_id;
        __syncwarp(active);
        int src = src_lane_map[myrank];

        ku = __shfl_sync(active, ku, src);
        payload = __shfl_sync(active, payload, src);
        idx = __shfl_sync(active, idx, src);
    }
    key = __uint_as_float(ku);
}


// Single-token router gemv on a transposed gate copy: scores = x @ gate_t.T. One warp per
// expert; cheaper than a cublas call at this size
#define RGEMV_WARPS 8

__global__ __launch_bounds__(RGEMV_WARPS * 32)
void routing_gemv_kernel
(
    const half* __restrict__ x,         // (k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (E)
    const int k,
    const int E
)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * RGEMV_WARPS + warp;
    if (row >= E) return;

    const half2* x2 = (const half2*) x;
    const half2* w2 = (const half2*) (gate_t + (size_t) row * k);

    float sum = 0.0f;
    for (int j = lane; j < k / 2; j += 32)
    {
        float2 xf = __half22float2(x2[j]);
        float2 wf = __half22float2(w2[j]);
        sum = fmaf(xf.x, wf.x, sum);
        sum = fmaf(xf.y, wf.y, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (lane == 0)
        scores[row] = __float2half_rn(sum);
}

// scores = x @ gate_t.T for a single token (x is (k), gate_t is (E, k), k even).
void routing_gemv
(
    const half* x,            // (k)
    const half* gate_t,       // (E, k) transposed gate copy
    half* scores,             // (E)
    int k,
    int E,
    Stream s
)
{
    HELIOS_AUX_CHECK(!(k & 1), "routing_gemv: k must be even");
    routing_gemv_kernel<<<CEIL_DIVIDE(E, RGEMV_WARPS), RGEMV_WARPS * 32, 0, s>>>
    (
        x,
        gate_t,
        scores,
        k, E
    );
    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}


template <int ACT>
__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_ds3_nogroup_topk_kernel
(
    const half* __restrict__ scores,
    const half* __restrict__ bias,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);
    bool mask = t < num_experts;

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    float* sh_key = reinterpret_cast<float*>(sh);
    float* sh_payload = reinterpret_cast<float*>(sh_key + num_warps * K);
    int* sh_idx = reinterpret_cast<int*>(sh_payload + num_warps * K);

    float logit = mask ? __half2float(scores[t]) : -1.0e30f;
    float act = bias && mask ? routing_act<ACT>(logit) : 0.0f;
    float key = mask ? (bias ? act + __half2float(bias[t]) : logit) : -1.0e30f;
    float payload = bias ? act : logit;
    int idx = mask ? t : -1;

    for (int k = 0; k < K; ++k)
    {
        float best_key = key;
        float best_payload = payload;
        int best_idx = idx;
        warp_reduce_best_f32(best_key, best_payload, best_idx);

        if (lane_id == k)
        {
            sh_key[warp_id * K + k] = best_key;
            sh_payload[warp_id * K + k] = best_payload;
            sh_idx[warp_id * K + k] = best_idx;
        }

        if (idx == best_idx) key = -1.0e30f;
    }
    __syncthreads();

    int num_candidates = num_warps * K;
    while (num_candidates > 32)
    {
        int stage_warps = CEIL_DIVIDE(num_candidates, 32);

        if (warp_id < stage_warps)
        {
            int pos = t;
            key = pos < num_candidates ? sh_key[pos] : -1.0e30f;
            payload = pos < num_candidates ? sh_payload[pos] : 0.0f;
            idx = pos < num_candidates ? sh_idx[pos] : -1;

            for (int k = 0; k < K; ++k)
            {
                float best_key = key;
                float best_payload = payload;
                int best_idx = idx;
                warp_reduce_best_f32(best_key, best_payload, best_idx);

                if (lane_id == k)
                {
                    sh_key[warp_id * K + k] = best_key;
                    sh_payload[warp_id * K + k] = best_payload;
                    sh_idx[warp_id * K + k] = best_idx;
                }

                if (idx == best_idx) key = -1.0e30f;
            }
        }
        __syncthreads();

        num_candidates = stage_warps * K;
    }

    if (warp_id == 0)
    {
        key = lane_id < num_candidates ? sh_key[lane_id] : -1.0e30f;
        payload = lane_id < num_candidates ? sh_payload[lane_id] : 0.0f;
        idx = lane_id < num_candidates ? sh_idx[lane_id] : -1;

        for (int k = 0; k < K; ++k)
        {
            float best_key = key;
            float best_payload = payload;
            int best_idx = idx;
            warp_reduce_best_f32(best_key, best_payload, best_idx);

            if (lane_id == k)
            {
                sh_payload[k] = bias ? best_payload : routing_act<ACT>(best_payload);
                sh_idx[k] = best_idx;
            }

            if (idx == best_idx) key = -1.0e30f;
        }

        __syncwarp();

        float o = lane_id < K ? sh_payload[lane_id] : 0.0f;
        float sum = warp_reduce_sum_first_k(o, K) + 1e-20f;
        if (lane_id < K)
        {
            topk_indices[lane_id] = (int64_t) sh_idx[lane_id];
            topk_weights[lane_id] = __float2half_rn(o * scaling_factor / sum);
        }
    }
}


template <int ACT>
__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_ds3_nogroup_kernel
(
    const half* __restrict__ scores,
    const half* __restrict__ bias,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);
    bool mask = t < num_experts;

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    int K_ = K + (K & 1);
    float* sh_v = reinterpret_cast<float*>(sh);
    float* sh_o = reinterpret_cast<float*>(sh_v + K_ * num_warps);
    int* sh_idx = reinterpret_cast<int*>(sh_o + K_ * num_warps);
    int* perm = reinterpret_cast<int*>(sh_idx + K_ * num_warps);
    float* reduce = reinterpret_cast<float*>(perm + 32 * num_warps);

    // Input activation
    int idx = mask ? t : -1;  // output index
    float v = mask ? routing_act<ACT>(__half2float(scores[t])) : 0.0f;  // sort key
    float o = v;  // output weight

    // Add bias and shift sigmoid(logits) to be non-negative before radix sort
    if (bias)
    {
        v += mask ? __half2float(bias[t]) : 1e30;

        float minv = v;
        for (int offset = 32 >> 1; offset > 0; offset >>= 1)
            minv = fminf(minv, __shfl_down_sync(0xffffffff, minv, offset));
        if (lane_id == 0)
            reduce[warp_id] = minv;

        __syncthreads();

        if (warp_id == 0)
        {
            minv = lane_id < num_warps ? reduce[lane_id] : 1e30;
            for (int offset = 32 >> 1; offset > 0; offset >>= 1)
                minv = fminf(minv, __shfl_down_sync(0xffffffff, minv, offset));
            if (lane_id == 0)
                reduce[0] = minv;
        }

        __syncthreads();

        v -= reduce[0];
        if (!mask) v = 0.0f;
    }

    // Sort by v
    warp_radixsort_posf32_pl(v, o, idx, perm + warp_id * 32);

    while (num_warps > 1)
    {
        if (warp_id < num_warps && lane_id >= (32 - K))
        {
            int kpos = (32 - 1) - lane_id;
            sh_v[warp_id * K + kpos] = v;
            sh_o[warp_id * K + kpos] = o;
            sh_idx[warp_id * K + kpos] = idx;
        }
        __syncthreads();

        int num_experts_k = K * num_warps;
        num_warps = CEIL_DIVIDE(num_experts_k, 32);

        if (warp_id < num_warps)
        {
            if (t < num_experts_k && mask)
            {
                v = sh_v[t];
                o = sh_o[t];
                idx = sh_idx[t];
            }
            else
            {
                v = 0.0f;
                o = 0.0f;
                idx = -1;
            }
            warp_radixsort_posf32_pl(v, o, idx, perm + warp_id * 32);
        }
        __syncthreads();
    }

    // Normalize output in warp 0 lanes 32-K .. K, store result
    if (warp_id == 0)
    {
        float sum = warp_reduce_sum_last_k(o, K) + 1e-20;
        o *= scaling_factor / sum;

        if (lane_id >= (32 - K))
        {
            int kpos = (32 - 1) - lane_id;
            topk_indices[kpos] = (int64_t) idx;
            topk_weights[kpos] = __float2half_rn(o);
        }
    }
}

/*
DS3 routing for n_group == 1, topk_group

scores: Routing logits, float16, shape (bsz, num_experts)
bias:   Pre-topk selection bias, float16, shape (num_experts), or null
topk_indices: int64, shape (bsz, k)
topk_weights: float16, shape (bsz, k)
routed_scaling_factor: float32
act_fn: score activation, ROUTING_ACT_SIGMOID (DS3/dots) or ROUTING_ACT_SQRTSP (DSv4)
*/

static void routing_ds3_launch
(
    const half* scores,
    const half* bias,
    int64_t* topk_indices,
    half* topk_weights,
    float scaling_factor,
    int bsz,
    int num_experts,
    int K,
    bool use_topk,
    int act_fn,
    Stream stream
)
{
    HELIOS_AUX_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    HELIOS_AUX_CHECK(K <= MAX_K, "Too many experts per token");
    HELIOS_AUX_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;

    if (use_topk)
    {
        // The iterative top-K kernel beats the radix-sort kernel at every measured size
        size_t shmem = num_warps * K * (2 * sizeof(float) + sizeof(int));
        auto kernel = act_fn == ROUTING_ACT_SQRTSP ?
            routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SQRTSP> :
            routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SIGMOID>;
        kernel<<<bsz, num_threads, shmem, stream>>>
        (
            scores,
            bias,
            topk_indices,
            topk_weights,
            scaling_factor,
            num_experts,
            K,
            bsz
        );
    }
    else
    {
        int K_ = K + (K & 1);
        size_t shmem = num_warps * K_ * (2 * sizeof(float) + sizeof(int))
                     + num_threads * sizeof(int)
                     + num_warps * sizeof(float);
        auto kernel = act_fn == ROUTING_ACT_SQRTSP ?
            routing_ds3_nogroup_kernel<ROUTING_ACT_SQRTSP> :
            routing_ds3_nogroup_kernel<ROUTING_ACT_SIGMOID>;
        kernel<<<bsz, num_threads, shmem, stream>>>
        (
            scores,
            bias,
            topk_indices,
            topk_weights,
            scaling_factor,
            num_experts,
            K,
            bsz
        );
    }

    HELIOS_CUDA_CHECK(cudaPeekAtLastError());
}

// Scores from a precomputed router logits buffer (was routing_ds3_nogroup_logits; the
// original routing_ds3_nogroup differed only in running routing_gemv first).
void routing_ds3_nogroup
(
    const half* scores,           // (bsz, num_experts)
    const half* bias,             // (num_experts) or null
    int64_t* topk_indices,        // (bsz, K)
    half* topk_weights,           // (bsz, K)
    int bsz,
    int num_experts,
    int K,
    float scaling_factor,
    bool use_topk,
    int act_fn,
    Stream s
)
{
    routing_ds3_launch(scores, bias, topk_indices, topk_weights, scaling_factor,
                       bsz, num_experts, K, use_topk, act_fn, s);
}

// Decode fast path: scores = hidden @ gate_t.T, then the iterative top-k kernel (the kernel
// the original full entry point always launched). Requires bsz == 1 and even k.
void routing_ds3_nogroup_hidden
(
    const half* hidden,           // (k)
    const half* gate_t,           // (num_experts, k)
    half* scores,                 // (num_experts) scratch
    const half* bias,             // (num_experts) or null
    int64_t* topk_indices,        // (K)
    half* topk_weights,           // (K)
    int k,
    int num_experts,
    int K,
    float scaling_factor,
    int act_fn,
    Stream s
)
{
    routing_gemv(hidden, gate_t, scores, k, num_experts, s);
    routing_ds3_launch(scores, bias, topk_indices, topk_weights, scaling_factor,
                       1, num_experts, K, true, act_fn, s);
}

}} // namespace helios::aux

