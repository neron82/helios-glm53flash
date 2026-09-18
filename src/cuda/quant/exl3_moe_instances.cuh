#pragma once
// Getter declarations for the MoE kernel instantiations built as separate translation units
// (comp_units/exl3_moe_inst_*.cu). Pruned versus exllamav3: only the runtime-bitrate instances
// (K = 0, n128/n256) and the fixed-bitrate instances K = 3 and K = 4 are built; K = 1, 2, 5, 6,
// 7, 8 are not, and the launcher falls back to the K = 0 variant for those (see README_PORT.md).

#include <cuda_runtime_api.h>
#include "exl3_moe_kernel.cuh"

// Kernel function pointer type: the launchers take the address of a __global__ kernel
typedef void (*fp_exl3_moe_kernel)(const half*, half*, half*, half*, half*, float*, const uint16_t**, const half**, const half**, const uint16_t**, const half**, const half**, const uint16_t**, const half**, const half**, const int64_t*, const int64_t*, const half*, int, int, int, int, int, int, float, int, int, int, int, int*);

#define DECL_GETTER(K_, n_, cb_) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K_##_n##n_##_cb##cb_();

// Runtime bitrate (Kg/Ku/Kd chosen inside the kernel)
DECL_GETTER(0, 128, 1)
DECL_GETTER(0, 256, 1)
DECL_GETTER(0, 128, 2)
DECL_GETTER(0, 256, 2)

// Fixed bitrate, Kg = Ku = Kd
DECL_GETTER(3, 128, 1)
DECL_GETTER(3, 256, 1)
DECL_GETTER(3, 128, 2)
DECL_GETTER(3, 256, 2)
DECL_GETTER(4, 128, 1)
DECL_GETTER(4, 256, 1)
DECL_GETTER(4, 128, 2)
DECL_GETTER(4, 256, 2)

#undef DECL_GETTER