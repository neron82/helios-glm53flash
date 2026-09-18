# Auxiliary CUDA Kernel Port

Ported from `exllamav3/exllamav3_ext/` into standalone C++20/CUDA kernels with no
PyTorch or `torch_ext` dependencies. Target: sm_86 (RTX 3090), CUDA 13.0.

## Build

Standalone build (does not touch the parent Helios CMake):

```bash
cd helios/src/cuda/aux
cmake -GNinja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -B build
ninja -C build
./build/test_aux
```

## Architecture

- `cuda_shim.hpp` (shared, parent dir): provides `HELIOS_CUDA_CHECK`, `Stream`
  alias (`cudaStream_t`), and CUDA runtime includes.
- `helios_shim.cuh` (this dir): device-side helpers shared across .cu files
  (half/bfloat16 conversion, `ceil_divide`, fast sigmoid/exp variants).
- `dtypes.hpp` (this dir): shared `DType` enum and constants (`kHalf`, `kFloat`, `kBFloat16`).

## File Mapping

| Original | Ported | Status |
|----------|--------|--------|
| norm.cu/.cuh | norm.cu/.cuh | ✅ Complete |
| activation.cu/.cuh | activation.cu/.cuh | ✅ Complete |
| hc_mix.cu/.cuh | hc_mix.cu/.cuh | ✅ Complete |
| routing.cu/.cuh | routing.cu/.cuh | ✅ Complete |
| dsa_topk.cu/.cuh | dsa_topk.cu/.cuh | ✅ Complete |
| add.cu/.cuh | add.cu/.cuh | ✅ Complete |
| gdn.cu/.cuh | gdn.cu/.cuh | ✅ Complete (decode-only) |

## API Notes

### Namespace and Linkage

All launchers are in `namespace helios::aux`. Functions use C++ linkage
(mangled names), not `extern "C"`, per the established convention in the
quant/ directory.

### Stream Convention

Every launcher takes an explicit `Stream` as the last parameter (default 0
= default stream). No internal `getCurrentCUDAStream()` calls.

### RMS Norm (`norm.cu`)

```cpp
void rms_norm(const void* x, DType tx, const void* w, DType tw,
              void* y, DType ty, int rows, int dim,
              float epsilon, float constant_bias = 0.0f, float constant_scale = 1.0f,
              bool add_residual = false, int w_groups = 1, Stream s = 0);
```

- Supports half/float/bfloat16 inputs.
- `add_residual` (legacy) adds the normalized output to y in-place.
- `rms_norm_res_in` provides the fused pre-norm residual add.

### Activation Functions (`activation.cu`)

```cpp
void silu_mul(const void* x, const void* y, half* z, bool float_input,
              float act_limit, size_t numel, Stream s = 0);
```

- `float_input` selects fp32 vs fp16 input pair (always fp16 output).
- Other variants: `gelu_mul`, `relu_mul`, `relu2_mul`, `xielu`, `silu_oai_mul`,
  `add_sigmoid_gate`, `mul_sigmoid_`, `mul_sigmoid_broadcast_`,
  `mul_softplus_broadcast_`, `add_sigmoid_gate_proj`, `deinterleave_qg`.

### HyperConnection Mix (`hc_mix.cu`)

```cpp
void hc_mix(const float* x, const void* w, bool w_bf16, const void* b,
            const void* hc, int H, int D, int rows, float hc_eps,
            Stream s = 0);
```

- Fixed H=4, RMS normalization, Sinkhorn iterations.
- `hc_mix_num_chunks(rows)` for chunked processing.

### DS3 Routing (`routing.cu`)

```cpp
void routing_ds3_nogroup(const half* gate_w, const half* x, int* indices,
                         half* weights, int rows, int num_experts, int topk,
                         float routed_scale, Stream s = 0);
```

- Sigmoid router + pre-topk bias.
- `routing_ds3_nogroup_hidden` variant includes expert activation.
- `routing_gemv` for half-precision GEMV.

### DSA Top-K (`dsa_topk.cu`)

```cpp
void dsa_topk(const half* scores, int s_stride, int* indices,
              int R, int T, int k, int k_pad, Stream s = 0);
```

- `k_pad` must be >= `k` (padding to power-of-2 for efficiency).
- Output is row-strided `[R, k_pad]` int32 with -1 padding beyond k.
- Automatic split/merge path for few-row, long-scan shapes.
- Persistent workspace (lazily allocated, ~13 MB); call `dsa_topk_free_workspace()` to release.

### Add/Copy (`add.cu`)

```cpp
void add(const void* a, DType ta, const void* b, DType tb,
         void* c, DType tc, int rows, int cols, Stream s = 0);
```

- Eight type combinations (half/float × half/float).
- Also: `copy2d`, `moe_bias_add`, `moe_bias_add_weighted`.

### GDN / KDA (`gdn.cu`)

Decode-only SSM kernels (Mamba2 and graph variants dropped):

```cpp
void kda_gate_op(const float* qkv, const float* b, const float* f,
                 const bfloat16* dt_bias, const void* a_log, bool a_log_fp32,
                 bfloat16* mixed_qkv, bfloat16* beta, float* g,
                 int B, int S, int F, int H, int Dk,
                 float lower_bound, float beta_scale, Stream s = 0);

void cuda_recurrent_gated_delta_rule(
    const bfloat16* mixed_qkv, const float* g, const bfloat16* beta,
    float* recurrent_state, bfloat16* core_attn_out,
    int bsz, int seqlen, int num_k_heads, int num_v_heads,
    int k_head_dim, int v_head_dim, int history_stride,
    const int* slots, bool channelwise, bool history, Stream s = 0);

void cuda_causal_conv1d_update(
    const bfloat16* x, bfloat16* conv_state, const int* slots,
    const bfloat16* weight, const bfloat16* bias, bfloat16* out,
    int bsz, int dim, int seqlen, int state_size, int K,
    bool activation, bool history, Stream s = 0);
```

Additional GDN functions: `gated_delta_net_fused_op`, `_2`, `_3`,
`gdn_ba_gemv`, `gdn_lowrank_gemv_f`, `batched_conv_rewind`, `batched_state_rewind`.

- Recurrent state is fp32; output is bfloat16.
- `history` flag enables per-step state saving for speculative decode rollback.
- KDA uses channelwise decay (128×128 head dimensions required).

## Tests

`test/test_aux.cpp` provides CPU-reference parity tests:
- RMS norm (half precision, max error < 0.01)
- SiLU multiply (half precision, max error < 0.01)
- DSA top-k (index validity and uniqueness)

GPU allocations kept < 500 MB.

## Deviations from Original

1. **Graph variants removed**: All `_gr` (CUDA graph) entry points dropped. Callers
   manage streams explicitly.
2. **Mamba2 removed**: `mamba2_dt_op`, `mamba2_fused_op`, `cuda_recurrent_mamba2`
   and their graph variants are not ported (decode-only KDA/GDN focus).
3. **Standard softmax routing removed**: `routing_ds3` (standard softmax) and
   `routing_sel_norm`, `moe_split_*` not ported; only DS3 nogroup (sigmoid + bias)
   variants retained.
4. **HC gated residual removed**: `gr_*` (gated residual) HC functions dropped.
5. **Torch workspace cache replaced**: DSA topk uses static lazy `cudaMalloc`
   instead of the torch workspace cache.
6. **Environment knobs removed**: `EXL3_DSA_TOPK_SPLIT` and similar env variable
   branches deleted; tuned defaults kept as constexpr.
7. **Kernel bodies byte-exact**: All `__global__` kernel functions are copied
   unchanged from exllamav3. Only launcher wrappers, includes, namespaces, and
   error checking were modified.