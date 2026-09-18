#pragma once
// Minimal CUDA shim replacing torch glue in ported exllamav3 kernels.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef cuda_check
#define cuda_check(ans) HELIOS_CUDA_CHECK(ans)
#endif
#define HELIOS_CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); abort(); } } while (0)

namespace helios {

// Every ported launcher takes an explicit stream (0-meaning default is caller's choice).
using Stream = cudaStream_t;

constexpr int SM_COUNT_3090 = 82;

} // namespace helios