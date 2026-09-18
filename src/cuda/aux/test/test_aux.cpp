#include "../../cuda_shim.hpp"
#include "../norm.cuh"
#include "../activation.cuh"
#include "../dsa_topk.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

using namespace helios;
using namespace aux;

static void fail(const char* msg) {
    std::cerr << "FAIL: " << msg << "\n";
    abort();
}

static void success(const char* msg) {
    std::cout << "PASS: " << msg << "\n";
}

// CPU reference: RMS norm with weight
static void rms_norm_ref(const half* x, const half* w, half* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float xv = __half2float(x[r * cols + c]);
            sum += xv * xv;
        }
        float norm = sqrtf(sum / cols + eps);
        for (int c = 0; c < cols; c++) {
            float xv = __half2float(x[r * cols + c]);
            float wv = __half2float(w[c]);
            out[r * cols + c] = __float2half(xv * wv / norm);
        }
    }
}

// CPU reference: silu(x) * y
static void silu_mul_ref(const half* x, const half* y, half* out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float xv = __half2float(x[r * cols + c]);
            float yv = __half2float(y[r * cols + c]);
            float s = 1.0f / (1.0f + expf(-xv));
            out[r * cols + c] = __float2half(xv * s * yv);
        }
    }
}

int main() {
    // Initialize CUDA
    HELIOS_CUDA_CHECK(cudaSetDevice(0));

    // --- RMS Norm Test ---
    {
        const int rows = 8;
        const int cols = 128;
        const float eps = 1e-6f;

        half* h_x = new half[rows * cols];
        half* h_w = new half[cols];
        half* h_out = new half[rows * cols];
        half* h_out_ref = new half[rows * cols];
        std::memset(h_out, 0, rows * cols * 2);
        std::memset(h_out_ref, 0, rows * cols * 2);

        // Fill with random-ish values
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < rows * cols; i++) h_x[i] = __float2half(dist(rng));
        for (int i = 0; i < cols; i++) h_w[i] = __float2half(1.0f + 0.1f * dist(rng));

        // CPU reference
        rms_norm_ref(h_x, h_w, h_out_ref, rows, cols, eps);

        // GPU kernel
        half* d_x, *d_w, *d_out;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_x, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_w, cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_out, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x, rows * cols * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_w, h_w, cols * 2, cudaMemcpyHostToDevice));
        rms_norm(d_x, kHalf, d_w, kHalf, d_out, kHalf, rows, cols, eps, 0.0f, 1.0f, false, 1, 0);

        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

        HELIOS_CUDA_CHECK(cudaMemcpy(h_out, d_out, rows * cols * 2, cudaMemcpyDeviceToHost));

        // Compare
        float max_err = 0.0f;
        for (int i = 0; i < rows * cols; i++) {
            float a = __half2float(h_out[i]);
            float b = __half2float(h_out_ref[i]);
            float err = fabsf(a - b);
            if (err > max_err) max_err = err;
        }
        std::cout << "RMS norm max error: " << max_err << " (half precision expected <0.01)\n";
        if (max_err > 0.01f) fail("RMS norm");
        success("RMS norm");

        delete[] h_x; delete[] h_w; delete[] h_out; delete[] h_out_ref;
        HELIOS_CUDA_CHECK(cudaFree(d_x));
        HELIOS_CUDA_CHECK(cudaFree(d_w));
        HELIOS_CUDA_CHECK(cudaFree(d_out));
    }

    // --- SILU Multiply Test ---
    {
        const int rows = 8;
        const int cols = 128;

        half* h_x = new half[rows * cols];
        half* h_y = new half[rows * cols];
        half* h_out = new half[rows * cols];
        half* h_out_ref = new half[rows * cols];
        std::memset(h_out, 0, rows * cols * 2);
        std::memset(h_out_ref, 0, rows * cols * 2);

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (int i = 0; i < rows * cols; i++) {
            h_x[i] = __float2half(dist(rng));
            h_y[i] = __float2half(dist(rng));
        }

        // CPU reference
        silu_mul_ref(h_x, h_y, h_out_ref, rows, cols);

        // GPU kernel
        half* d_x, *d_y, *d_out;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_x, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_y, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_out, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x, rows * cols * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_y, h_y, rows * cols * 2, cudaMemcpyHostToDevice));

        silu_mul(d_x, d_y, d_out, false, 100.0f, rows * cols, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

        HELIOS_CUDA_CHECK(cudaMemcpy(h_out, d_out, rows * cols * 2, cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < rows * cols; i++) {
            float a = __half2float(h_out[i]);
            float b = __half2float(h_out_ref[i]);
            float err = fabsf(a - b);
            if (err > max_err) max_err = err;
        }
        std::cout << "SILU mul max error: " << max_err << " (half precision expected <0.01)\n";
        if (max_err > 0.01f) fail("SILU mul");
        success("SILU mul");

        delete[] h_x; delete[] h_y; delete[] h_out; delete[] h_out_ref;
        HELIOS_CUDA_CHECK(cudaFree(d_x));
        HELIOS_CUDA_CHECK(cudaFree(d_y));
        HELIOS_CUDA_CHECK(cudaFree(d_out));
    }

    // --- DSA Top-K Test ---
    {
        const int rows = 8;
        const int cols = 64;
        const int k = 8;

        half* h_x = new half[rows * cols];
        int* h_indices = new int[rows * k];
        std::memset(h_indices, -1, rows * k * 4);

        std::mt19937 rng(456);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < rows * cols; i++) h_x[i] = __float2half(dist(rng));

        half* d_x;
        int* d_indices;
        HELIOS_CUDA_CHECK(cudaMalloc(&d_x, rows * cols * 2));
        HELIOS_CUDA_CHECK(cudaMalloc(&d_indices, rows * k * 4));
        HELIOS_CUDA_CHECK(cudaMemcpy(d_x, h_x, rows * cols * 2, cudaMemcpyHostToDevice));
        HELIOS_CUDA_CHECK(cudaMemset(d_indices, -1, rows * k * 4));

        dsa_topk(d_x, cols, d_indices, rows, cols, k, k, 0);
        HELIOS_CUDA_CHECK(cudaDeviceSynchronize());

        HELIOS_CUDA_CHECK(cudaMemcpy(h_indices, d_indices, rows * k * 4, cudaMemcpyDeviceToHost));

        // Verify indices are in [0, cols) and unique per row
        bool ok = true;
        for (int r = 0; r < rows && ok; r++) {
            std::vector<int> seen(cols, 0);
            for (int i = 0; i < k && ok; i++) {
                int idx = h_indices[r * k + i];
                if (idx < 0 || idx >= cols) {
                    std::cerr << "DSA topk: index " << idx << " out of range\n";
                    ok = false;
                } else if (seen[idx]) {
                    std::cerr << "DSA topk: duplicate index " << idx << " in row " << r << "\n";
                    ok = false;
                } else {
                    seen[idx] = 1;
                }
            }
        }
        if (!ok) fail("DSA topk");
        success("DSA topk");

        delete[] h_x; delete[] h_indices;
        HELIOS_CUDA_CHECK(cudaFree(d_x));
        HELIOS_CUDA_CHECK(cudaFree(d_indices));
        dsa_topk_free_workspace();
    }

    std::cout << "All aux parity tests passed.\n";
    return 0;
}