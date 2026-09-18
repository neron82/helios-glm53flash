// Smoke test for the ported EXL3 GEMM/GEMV layer (helios::exl3::gemm).
//
// Flow: random K-bit indices -> pack_trellis -> reconstruct() gives the dequantized weight tile
// B_hat on the GPU -> copied back and transformed on the host into the effective weight matrix
//     B_eff = diag(suh) . H128(k) . B_hat . H128(n) . diag(svh)      (per 128x128 group)
// which is exactly what the kernel multiplies against (input Hadamard prologue with suh, fp16
// MMA, output Hadamard epilogue with svh). A CPU matmul over B_eff is the reference; the kernel
// result must match within a loose relative tolerance (fp16 accumulation on sm_86 costs ~1%).
//
// Covers: bit widths 2 and 4, default codebook, M = 1 (routes to the small-m GEMV kernel) and
// M = 64 (tiled cooperative GEMM, four 16-row slabs). Total device allocation stays under 1 MB.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "helios_shim.cuh"
#include "pack.cuh"
#include "reconstruct.cuh"
#include "exl3_gemm.cuh"

using namespace helios;
using namespace helios::exl3;

namespace
{

constexpr float RS = 0.088388347648318447f;  // 1 / sqrt(128), the kernel's constant

int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok)
    {
        printf("  FAIL: %s\n", what);
        failures++;
    }
}

float f16_to_f32(uint16_t h)
{
    uint32_t sign = h & 0x8000u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t man  = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0)
    {
        if (man == 0) bits = sign << 16;
        else
        {
            int e = -1;
            while (!(man & 0x400u)) { man <<= 1; e--; }
            man &= 0x3ffu;
            bits = ((uint32_t)(127 - 15 + e) << 23) | (man << 13) | (sign << 16);
        }
    }
    else if (exp == 0x1f) bits = (sign << 16) | 0x7f800000u | (man << 13);
    else bits = ((exp + 127 - 15) << 23) | (man << 13) | (sign << 16);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t f32_to_f16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = bits >> 16;
    int exp = (int)((bits >> 23) & 0xffu) - 127;
    uint32_t man = bits & 0x7fffffu;
    if (exp == 128) return (uint16_t)((sign << 15) | 0x7c00u | (man ? 0x200u : 0u));
    if (exp > 15) return (uint16_t)((sign << 15) | 0x7c00u);
    if (exp < -24) return (uint16_t)(sign << 15);
    if (exp < -14)
    {
        exp += 24;
        uint32_t m = man | 0x800000u;
        uint32_t h = m >> (14 - exp);
        if ((m >> (13 - exp)) & 1u) h++;
        return (uint16_t)((sign << 15) | h);
    }
    uint32_t h = ((uint32_t)(exp + 15) << 10) | (man >> 13);
    if ((man >> 12) & 1u) h++;
    return (uint16_t)((sign << 15) | h);
}

// In-place 128-point Walsh-Hadamard transform, normalized by 1/sqrt(128) exactly as the
// kernel's stages do (butterfly in float, single scaling at the end).
void fwht128(float* v)
{
    for (int i = 1; i < 128; i <<= 1)
        for (int j = 0; j < 128; j += i << 1)
            for (int k = 0; k < i; k++)
            {
                float a = v[j + k];
                float b = v[j + k + i];
                v[j + k]     = a + b;
                v[j + k + i] = a - b;
            }
    for (int j = 0; j < 128; j++) v[j] *= RS;
}

// One (bits, M) case. Returns relative RMS error.
double run_case(int bits, int M)
{
    const int K = 256;
    const int N = 512;
    const int packed_rows = K / 16;
    const int packed_cols = N / 16;

    printf("[bits=%d M=%d]\n", bits, M);

    std::mt19937 rng(0xC0FFEEu + bits * 97 + M);
    std::uniform_int_distribution<int> idx_dist(0, (1 << bits) - 1);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

    // Random indices, one per weight value
    std::vector<uint16_t> h_idx(K * N);
    for (auto& v : h_idx) v = (uint16_t) idx_dist(rng);

    // Random suh / svh (scales + sign flips), kept small so nothing overflows
    std::vector<uint16_t> h_suh(K), h_svh(N), h_A(M * K);
    for (auto& v : h_suh) v = f32_to_f16(unit(rng));
    for (auto& v : h_svh) v = f32_to_f16(unit(rng));
    for (auto& v : h_A)   v = f32_to_f16(unit(rng));

    const size_t trellis_words = (size_t) packed_rows * packed_cols * (256 * bits / 16);
    std::vector<uint16_t> h_packed(trellis_words);

    uint16_t* d_idx    = nullptr;
    uint16_t* d_pack   = nullptr;
    half*     d_bhat   = nullptr;
    half*     d_suh    = nullptr;
    half*     d_svh    = nullptr;
    half*     d_A      = nullptr;
    half*     d_ahad   = nullptr;
    float*    d_C      = nullptr;

    HELIOS_CUDA_CHECK(cudaMalloc((void**) &d_idx,   sizeof(uint16_t) * h_idx.size()));
    HELIOS_CUDA_CHECK(cudaMalloc((void**) &d_pack,  sizeof(uint16_t) * trellis_words));
    HELIOS_CUDA_CHECK(cudaMalloc((void**) &d_bhat,  sizeof(half) * (size_t) K * N));
    HELIOS_CUDA_CHECK(cudaMalloc((void**) &d_suh,   sizeof(half) * K));
    HELIOS_CUDA_CHECK(cudaMalloc((void**) &d_svh,   sizeof(half) * N));
    HELIOS_CUDA_CHECK(cudaMalloc((void**) &d_A,     sizeof(half) * (size_t) M * K));
    HELIOS_CUDA_CHECK(cudaMalloc((void**) &d_ahad,  sizeof(half) * (size_t) M * K));
    HELIOS_CUDA_CHECK(cudaMalloc((void**) &d_C,     sizeof(float) * (size_t) M * N));

    HELIOS_CUDA_CHECK(cudaMemcpy(d_idx,  h_idx.data(),  sizeof(uint16_t) * h_idx.size(),  cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_suh,  h_suh.data(),  sizeof(half) * K,                 cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_svh,  h_svh.data(),  sizeof(half) * N,                 cudaMemcpyHostToDevice));
    HELIOS_CUDA_CHECK(cudaMemcpy(d_A,    h_A.data(),    sizeof(half) * (size_t) M * K,    cudaMemcpyHostToDevice));

    // Quantize: indices -> packed trellis
    pack_trellis(d_pack, d_idx, packed_rows, packed_cols, bits);

    // Dequantize: trellis -> B_hat (fp16, original layout, no Hadamard)
    reconstruct(d_bhat, d_pack, packed_rows, packed_cols, N, bits, false, false);

    std::vector<uint16_t> h_bhat((size_t) K * N);
    HELIOS_CUDA_CHECK(cudaMemcpy(h_bhat.data(), d_bhat, sizeof(half) * (size_t) K * N, cudaMemcpyDeviceToHost));
    HELIOS_CUDA_CHECK(cudaMemcpy(h_packed.data(), d_pack, sizeof(uint16_t) * trellis_words, cudaMemcpyDeviceToHost));

    // Host effective weight: diag(suh) . H . B_hat . H . diag(svh), per 128x128 group
    // Kernel computes: ((A * suh) * H_k) * B_hat * H_n * svh
    // Host reference must match exactly: scale A by suh, Hadamard along k,
    // multiply by B_hat, Hadamard along n, scale by svh.
    // Step 1: A_eff = (A * suh) * H along k
    std::vector<float> a_eff((size_t) M * K);
    for (int m = 0; m < M; m++)
    {
        std::vector<float> row(K);
        for (int i = 0; i < K; i++) row[i] = f16_to_f32(h_A[(size_t) m * K + i]) * f16_to_f32(h_suh[i]);
        for (int kb = 0; kb < K / 128; kb++) fwht128(row.data() + kb * 128);
        for (int i = 0; i < K; i++) a_eff[(size_t) m * K + i] = row[i];
    }

    // Step 2: C = A_eff * B_hat (no transforms on B_hat)
    std::vector<double> c_unscaled((size_t) M * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            double acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += (double) a_eff[(size_t) m * K + k] * (double) f16_to_f32(h_bhat[(size_t) k * N + n]);
            c_unscaled[(size_t) m * N + n] = acc;
        }

    // Step 3: Hadamard along n, then svh
    std::vector<double> c_ref((size_t) M * N);
    for (int m = 0; m < M; m++)
        for (int nb = 0; nb < N / 128; nb++)
        {
            std::vector<float> t(128);
            for (int j = 0; j < 128; j++) t[j] = (float) c_unscaled[(size_t) m * N + nb * 128 + j];
            fwht128(t.data());
            for (int j = 0; j < 128; j++)
                c_ref[(size_t) m * N + nb * 128 + j] = (double) t[j] * (double) f16_to_f32(h_svh[nb * 128 + j]);
        }

    // Kernel
    GroupWords w;
    w.trellis = d_pack;
    w.suh     = d_suh;
    w.svh     = d_svh;
    w.mul1    = 0;

    HELIOS_CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * (size_t) M * N));
    int shape = gemm(d_C, d_A, w, M, N, K, bits, true, 0, d_ahad, false, 0, 0);
    HELIOS_CUDA_CHECK(cudaDeviceSynchronize());
    printf("  dispatched shape %d (%s)\n", shape, shape == 90 ? "gemv" : "gemm");

    std::vector<float> h_C((size_t) M * N);
    HELIOS_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeof(float) * (size_t) M * N, cudaMemcpyDeviceToHost));
    double num = 0.0, den = 0.0, worst = 0.0;
    for (size_t i = 0; i < h_C.size(); i++)
    {
        double d = (double) h_C[i] - c_ref[i];
        num += d * d;
        den += c_ref[i] * c_ref[i];
        double rel = std::fabs(d) / std::max(1e-3, std::fabs(c_ref[i]));
        if (rel > worst) worst = rel;
    }
    double rel_rms = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    printf("  rel RMS %.3e, worst pointwise %.3e\n", rel_rms, worst);

    cudaFree(d_idx);
    cudaFree(d_pack);
    cudaFree(d_bhat);
    cudaFree(d_suh);
    cudaFree(d_svh);
    cudaFree(d_A);
    cudaFree(d_ahad);
    cudaFree(d_C);

    check(rel_rms < 5e-2, "relative RMS error below 5e-2");
    return rel_rms;
}

} // namespace

int main()
{
    int device = 0;
    HELIOS_CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    HELIOS_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("device: %s (SMs %d)\n", prop.name, prop.multiProcessorCount);

    run_case(4, 1);
    run_case(4, 64);
    run_case(2, 1);
    run_case(2, 64);

    if (failures)
    {
        printf("FAILURES: %d\n", failures);
        return 1;
    }
    printf("ALL OK\n");
    return 0;
}