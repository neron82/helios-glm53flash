#pragma once
// Host-side stand-ins for the handful of ptx.cuh entities the EXL3 dequant path uses,
// plus the PTX intrinsics/instructions the dequant code relies on. Each replacement
// implements the documented semantics of the instruction it stands in for; the real
// device code (exl3_dq.cuh / codebook.cuh, included into the test through host_dq.hpp /
// host_codebook.hpp) is otherwise used verbatim. See README_PORT.md §"Bit-exact test".

#include "helios_shim.cuh"

// Tensor core fragment types (mirrors ptx.cuh, minus the mma asm helpers which the
// dequant path never uses and which cannot compile on the host).
template <typename T, int n>
struct Vec
{
    T elems[n];
    T& operator[](int i) { return elems[i]; }
};

using FragB = Vec<half2, 2>;

// shf.r.clamp.b32 / funnelshift_r / the 64-bit window shift used by fshift(): the low 32
// bits of ((hi << 32) | lo) >> shift, saturating to 0 for shift >= 64 (what the hardware
// shift does; the device code can produce shifts > 64 for K=8).
static inline uint32_t host_shift_window(uint32_t lo, uint32_t hi, int shift)
{
    if (shift < 0) shift = 0;
    if (shift >= 64) return 0u;
    uint64_t v = ((uint64_t)hi << 32) | (uint64_t)lo;
    return (uint32_t)(v >> shift);
}

// bfe.u32 dst, src, pos, 16 — zero-fills when pos + 16 exceeds the source width.
static inline uint32_t host_bfe16(uint32_t src, int pos)
{
    if (pos < 0 || pos >= 32) return 0u;
    return (src >> pos) & 0xffffu;
}

// bfe.u64 with an immediate offset/length over a 64-bit window.
static inline uint64_t host_bfe64(uint32_t lo, uint32_t hi, int off, int len)
{
    uint64_t v = ((uint64_t)hi << 32) | (uint64_t)lo;
    if (off < 0) off = 0;
    if (off >= 64 || len <= 0) return 0ull;
    if (off + len > 64) len = 64 - off;
    return (v >> off) & (len >= 64 ? ~0ull : ((1ull << len) - 1ull));
}

// dp4a — signed? no: unsigned byte-wise dot product plus accumulator.
static inline uint32_t host_dp4a(uint32_t a, uint32_t b, uint32_t acc)
{
    uint32_t s = acc;
    for (int i = 0; i < 4; ++i)
        s += ((a >> (i * 8)) & 0xffu) * ((b >> (i * 8)) & 0xffu);
    return s;
}

// ---------------------------------------------------------------------------
// Exact fp16 arithmetic (single rounding). CUDA's host implementations of the
// __h* intrinsics go through float, which can double-round; these go through
// double, which is exact for every fp16 operand pair.
// ---------------------------------------------------------------------------

static inline uint16_t hbits(half h)
{
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
static inline half hfrom(uint16_t u)
{
    half h;
    memcpy(&h, &u, 2);
    return h;
}
static inline float htof(uint16_t u)
{
    uint32_t sign = (uint32_t)(u & 0x8000) << 16;
    uint32_t expo = (u >> 10) & 0x1f;
    uint32_t mant = u & 0x3ff;
    uint32_t bits;
    if (expo == 0)
    {
        if (mant == 0) bits = sign;
        else
        {
            uint32_t e = 1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3ff;
            bits = sign | ((e + 112) << 23) | (mant << 13);
        }
    }
    else if (expo == 0x1f) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((expo + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
static uint16_t f64_to_h(double v);   // defined in the test TU

static inline uint16_t hadd_bits(uint16_t a, uint16_t b) { return f64_to_h((double)htof(a) + (double)htof(b)); }
static inline uint16_t hsub_bits(uint16_t a, uint16_t b) { return f64_to_h((double)htof(a) - (double)htof(b)); }
static inline uint16_t hmul_bits(uint16_t a, uint16_t b) { return f64_to_h((double)htof(a) * (double)htof(b)); }
static inline uint16_t hfma_bits(uint16_t a, uint16_t b, uint16_t c)
{
    return f64_to_h((double)htof(a) * (double)htof(b) + (double)htof(c));
}

static inline half2 hadd2_host(half2 a, half2 b) { return __halves2half2(hfrom(hadd_bits(hbits(a.x), hbits(b.x))), hfrom(hadd_bits(hbits(a.y), hbits(b.y)))); }
static inline half2 hsub2_host(half2 a, half2 b) { return __halves2half2(hfrom(hsub_bits(hbits(a.x), hbits(b.x))), hfrom(hsub_bits(hbits(a.y), hbits(b.y)))); }
static inline half2 hmul2_host(half2 a, half2 b) { return __halves2half2(hfrom(hmul_bits(hbits(a.x), hbits(b.x))), hfrom(hmul_bits(hbits(a.y), hbits(b.y)))); }
static inline half2 hfma2_host(half2 a, half2 b, half2 c) { return __halves2half2(hfrom(hfma_bits(hbits(a.x), hbits(b.x), hbits(c.x))), hfrom(hfma_bits(hbits(a.y), hbits(b.y), hbits(c.y)))); }