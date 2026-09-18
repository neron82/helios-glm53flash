#pragma once
// Ported from exllamav3 exllamav3_ext/dsa_topk.cuh: raw-pointer signatures, explicit stream.
// CUDA-graph parameter patching (dsa_topk_gr / dsa_seq_state_gr) is not part of the port.

#include "../cuda_shim.hpp"

namespace helios { namespace aux {

// top-k index selection for the DSA lightning indexer: for each of R rows pick the k largest
// fp16 scores and write their ascending indices to indices (row-major, k_pad wide), padding
// the tail with -1. scores must be dense in its innermost dim (row stride s_stride).
// t_ptr (device int, may be null) overrides T; t_seq is the host-side runtime T used by the
// legacy kernel (0 = use the static T). Split/merge path is chosen automatically for
// few-row, long-scan shapes.
void dsa_topk
(
    const half* scores,           // (R, T) half
    int s_stride,                 // scores row stride (elements)
    int* indices,                 // (R, k_pad) int32 out
    int R,
    int T,
    int k,
    int k_pad,
    const int* t_ptr = nullptr,
    int t_seq = 0,
    Stream s = 0
);

// Batched sparse-attention per-job sequence state: arr[0, b] = seqlens[b],
// arr[1, b] = seqlens[b] + q_len.
void dsa_seq_state
(
    const int* seqlens,           // (bsz,) i32
    int* arr,                     // (2, arr_stride) i32
    int bsz,
    int q_len,
    int arr_stride,
    Stream s = 0
);

// Free the lazily allocated split/merge candidate workspace (~13 MB). Optional: it is
// allocated on first split use and lives for the process otherwise.
void dsa_topk_free_workspace(void);

}} // namespace helios::aux