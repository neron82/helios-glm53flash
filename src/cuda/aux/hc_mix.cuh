#pragma once
// Ported from exllamav3 exllamav3_ext/hc_mix.cuh: same entry points, raw-pointer signatures.
// gr_mix (GatedResidual) is not part of the helios port.

#include "../cuda_shim.hpp"

namespace helios { namespace aux {

// Partials chunk count for a given row count / row length; the caller sizes the partials
// workspace to R * chunks * (M + 1) floats and passes the chunk count back to hc_mix/hc_head.
int hc_mix_num_chunks(int R, int row_len);

// mix(streams (R, H, D) fp32) -> post (R, H), comb (R, H, H), collapsed (R, D).
// fn is (2H + H^2, H * D), base (2H + H^2), scale (3); fn_half selects the fp16 weight
// variant (same fp32 dot math, half the traffic). collapsed dtype via half_out.
void hc_mix
(
    const float* streams,
    const void* fn,
    bool fn_half,
    const float* base,
    const float* scale,
    int R, int H, int D,
    int chunksA,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters,
    float* partials,
    float* post,
    float* comb,
    void* collapsed,
    bool half_out,
    Stream s = 0
);

// Final collapse: fn is (H, H * D), base (H), scale (1); no post/comb output.
void hc_head
(
    const float* streams,
    const void* fn,
    bool fn_half,
    const float* base,
    const float* scale,
    int R, int H, int D,
    int chunksA,
    float rms_eps,
    float hc_eps,
    float* partials,
    void* collapsed,
    bool half_out,
    Stream s = 0
);

// Residual update for one sublayer site: x <- post (*) y + comb^T x, updated IN PLACE.
// comb == nullptr gives x[h] += post[h] * y.
void hc_apply
(
    float* x,
    const void* y,
    bool y_half,
    const float* post,
    const float* comb,
    int R, int H, int D,
    Stream s = 0
);

}} // namespace helios::aux