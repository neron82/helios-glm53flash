#include <cuda_fp16.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../helios_shim.cuh"
#include "../helios_shim.cuh"
#include "../ptx.cuh"
#include "../exl3_gemm_kernel.cuh"
#include "exl3_comp_unit_4.cuh"

EXL3_KERNEL_INSTANCES_CB(4, 2)
