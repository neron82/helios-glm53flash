// Parity tests for helios::attn kernels vs CPU fp32 references. Small sizes, <500MB GPU.
#include "attn.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
using namespace helios; using namespace helios::attn;

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { printf("CUDA err %s %d @%d\n", cudaGetErrorName(e), (int)e, __LINE__); exit(1);} } while (0)

static std::mt19937 rng(7);
static float fr() { return (rng() % 2000) / 1000.f - 1.f; }

static std::vector<half> toh(const std::vector<float>& f) {
  std::vector<half> h(f.size());
  for (size_t i = 0; i < f.size(); i++) h[i] = __float2half_rn(f[i]);
  return h;
}
static float relerr(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); i++) { num += std::fabs(a[i] - b[i]); den += std::fabs(b[i]) + 1e-6; }
  return num / den;
}


// ---------------------------------------------------------------------------
// Standalone timing of mla_sparse_decode at the engine's prefill shape, so the 4.4s/batch the engine
// reports can be reconciled against the grid's theoretical cost.
static void bench_sparse(int rows_cap) {
  int m = 8192, kidx = 2052;
  std::vector<float> qn((size_t)m * 64 * 512), ck((size_t)rows_cap * 512);
  for (auto& x : qn) x = fr(); for (auto& x : ck) x = fr() * 0.3f;
  auto qh = toh(qn), chh = toh(ck);
  std::vector<int> ri((size_t)m * kidx);
  for (int r = 0; r < m; r++) for (int i = 0; i < kidx; i++) ri[(size_t)r * kidx + i] = (int)(rng() % rows_cap);
  half *dq, *dc, *dlo; int *dri;
  CK(cudaMalloc(&dq, qh.size() * 2)); CK(cudaMalloc(&dc, chh.size() * 2));
  CK(cudaMalloc(&dri, ri.size() * 4)); CK(cudaMalloc(&dlo, (size_t)m * 64 * 512 * 2));
  CK(cudaMemcpy(dq, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, chh.data(), chh.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dri, ri.data(), ri.size() * 4, cudaMemcpyHostToDevice));
  for (int K : {1, 2}) {
    char e[8]; sprintf(e, "%d", K);
    setenv("HELIOS_SPARSE_K", e, 1);
    mla_sparse_decode(dq, dc, dri, kidx, dlo, m, 0); CK(cudaDeviceSynchronize());   // warm
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    CK(cudaEventRecord(a, 0));
    mla_sparse_decode(dq, dc, dri, kidx, dlo, m, 0);
    CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    double visits = (double)m * 64 * kidx;
    printf("bench sparse K=%d ckv_rows=%d: %.1f ms  (%.3f ns per (q,head,key) warp-iteration, %.1f GB/s key bytes)\n",
           K, rows_cap, ms, ms * 1e6 / visits, visits * 1024.0 / (ms * 1e-3) / 1e9);
    cudaEventDestroy(a); cudaEventDestroy(b);
  }
  cudaFree(dq); cudaFree(dc); cudaFree(dri); cudaFree(dlo);
}

int main(int argc, char** argv) {
  CK(cudaSetDevice(0));
  setvbuf(stdout, nullptr, _IONBF, 0);

  if (argc > 1 && !strcmp(argv[1], "bench")) { bench_sparse(argc > 2 ? atoi(argv[2]) : 4096); bench_sparse(200000); return 0; }

  int fails = 0;
  // ---- q_latent / o_absorb ----
  {
    int m = 2;
    std::vector<float> qn(m * 64 * 256), kb(32768 * 512);
    for (auto& x : qn) x = fr(); for (auto& x : kb) x = fr() * 0.3f;
    auto qnh = toh(qn), kbh = toh(kb);
    half *dq, *dkb, *dql;
    CK(cudaMalloc(&dq, qnh.size() * 2)); CK(cudaMalloc(&dkb, kbh.size() * 2)); CK(cudaMalloc(&dql, m * 64 * 512 * 2));
    CK(cudaMemcpy(dq, qnh.data(), qnh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkb, kbh.data(), kbh.size() * 2, cudaMemcpyHostToDevice));
    q_latent(dq, dkb, dql, m); CK(cudaDeviceSynchronize());
    std::vector<half> hout(m * 64 * 512); CK(cudaMemcpy(hout.data(), dql, hout.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> ref(m * 64 * 512);
    for (int r = 0; r < m; r++) for (int h = 0; h < 64; h++) for (int c = 0; c < 512; c++) {
      double acc = 0;
      for (int j = 0; j < 256; j++) acc += __half2float(qnh[r * 64 * 256 + h * 256 + j]) * __half2float(kbh[(h * 512 + j) * 512 + c]);
      ref[r * 64 * 512 + h * 512 + c] = acc;
    }
    std::vector<float> got(hout.size()); for (size_t i = 0; i < hout.size(); i++) got[i] = __half2float(hout[i]);
    float e = relerr(got, ref);
    printf("q_latent relerr %.2e %s\n", e, e < 2e-3 ? "PASS" : "FAIL"); fails += e >= 2e-3;

    // o_absorb roundtrip: lat_out random
    std::vector<float> lo(m * 64 * 512); for (auto& x : lo) x = fr();
    auto loh = toh(lo);
    half *dlo, *do_;
    CK(cudaMalloc(&dlo, loh.size() * 2)); CK(cudaMalloc(&do_, m * 64 * 256 * 2));
    CK(cudaMemcpy(dlo, loh.data(), loh.size() * 2, cudaMemcpyHostToDevice));
    o_absorb(dlo, dkb, do_, m); CK(cudaDeviceSynchronize());
    std::vector<half> oout(m * 64 * 256); CK(cudaMemcpy(oout.data(), do_, oout.size() * 2, cudaMemcpyDeviceToHost));
    ref.assign(m * 64 * 256, 0);
    for (int r = 0; r < m; r++) for (int h = 0; h < 64; h++) for (int j = 0; j < 256; j++) {
      double acc = 0;
      for (int c = 0; c < 512; c++) acc += __half2float(loh[r * 64 * 512 + h * 512 + c]) * __half2float(kbh[(h * 512 + 256 + j) * 512 + c]);
      ref[r * 64 * 256 + h * 256 + j] = acc;
    }
    got.assign(oout.size(), 0); for (size_t i = 0; i < oout.size(); i++) got[i] = __half2float(oout[i]);
    e = relerr(got, ref);
    printf("o_absorb relerr %.2e %s\n", e, e < 2e-3 ? "PASS" : "FAIL"); fails += e >= 2e-3;
    cudaFree(dq); cudaFree(dkb); cudaFree(dql); cudaFree(dlo); cudaFree(do_);
  }

  // ---- kpool_write + indexer_score ----
  {
    int np = 64, tmax = 256;
    std::vector<float> ik(np * POOL * 128), ig(np * POOL * 128), ape(4 * 128);
    for (auto& x : ik) x = fr(); for (auto& x : ig) x = fr() * 3; for (auto& x : ape) x = fr() * 2;
    auto ikh = toh(ik), igh = toh(ig);
    half *dik, *dig, *draw, *dpk, *dpk_nt; float* dape;
    const int draw_rows = tmax;   // ring sized to cover the fixture (a small ring is tested below)
    CK(cudaMalloc(&dik, ikh.size() * 2)); CK(cudaMalloc(&dig, igh.size() * 2));
    CK(cudaMalloc(&draw, tmax * 256 * 2)); CK(cudaMalloc(&dpk, np * 128 * 2));
    CK(cudaMalloc(&dpk_nt, np * 128 * 2)); CK(cudaMalloc(&dape, ape.size() * 4));
    CK(cudaMemcpy(dik, ikh.data(), ikh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dig, igh.data(), igh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dape, ape.data(), ape.size() * 4, cudaMemcpyHostToDevice));
    kpool_write(dik, dig, draw, draw_rows, dpk_nt, dape, 0, np * POOL, 0); CK(cudaDeviceSynchronize());
    std::vector<half> pk(np * 128);   // pool-major on the device now
    CK(cudaMemcpy(pk.data(), dpk_nt, pk.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> ref(np * 128);
    for (int p = 0; p < np; p++) for (int c = 0; c < 128; c++) {
      float g[4], mx = -1e30f;
      for (int i = 0; i < 4; i++) { g[i] = __half2float(igh[(p * 4 + i) * 128 + c]) + ape[i * 128 + c]; mx = std::max(mx, g[i]); }
      float s = 0, o = 0;
      for (int i = 0; i < 4; i++) { g[i] = std::exp(g[i] - mx); s += g[i]; }
      for (int i = 0; i < 4; i++) o += (g[i] / s) * __half2float(ikh[(p * 4 + i) * 128 + c]);
      ref[p * 128 + c] = o * 0.25f;
    }
    std::vector<float> got(pk.size()); for (size_t i = 0; i < pk.size(); i++) got[i] = __half2float(pk[i]);
    float e = relerr(got, ref);
    printf("kpool_write relerr %.2e %s\n", e, e < 2e-3 ? "PASS" : "FAIL"); fails += e >= 2e-3;

    // A pool belongs to whichever call writes its LAST member, so the same tokens split across calls
    // - a chunk boundary, or a prefix-cache resume - must produce identical keys. Requiring all four
    // members to fall inside one call meant the decode path (one token per call) never wrote a pool
    // at all, while the indexer's visibility rule still exposed it.
    {
      const int n_tok = 16, npool_used = n_tok / POOL;
      std::vector<half> base(np * 128), by4(np * 128), by1(np * 128);
      CK(cudaMemset(draw, 0, tmax * 256 * 2)); CK(cudaMemset(dpk_nt, 0, np * 128 * 2));
      kpool_write(dik, dig, draw, draw_rows, dpk_nt, dape, 0, n_tok, 0); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(base.data(), dpk_nt, base.size() * 2, cudaMemcpyDeviceToHost));
      CK(cudaMemset(draw, 0, tmax * 256 * 2)); CK(cudaMemset(dpk_nt, 0, np * 128 * 2));
      for (int i = 0; i < n_tok; i += POOL)
        kpool_write(dik + (size_t)i * 128, dig + (size_t)i * 128, draw, draw_rows, dpk_nt, dape, i, POOL, 0);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(by4.data(), dpk_nt, by4.size() * 2, cudaMemcpyDeviceToHost));
      CK(cudaMemset(draw, 0, tmax * 256 * 2)); CK(cudaMemset(dpk_nt, 0, np * 128 * 2));
      for (int i = 0; i < n_tok; i++)
        kpool_write(dik + (size_t)i * 128, dig + (size_t)i * 128, draw, draw_rows, dpk_nt, dape, i, 1, 0);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(by1.data(), dpk_nt, by1.size() * 2, cudaMemcpyDeviceToHost));
      int d4 = 0, d1 = 0, unwritten = 0;
      for (int p = 0; p < npool_used; p++) for (int c = 0; c < 128; c++) {
        half a = base[(size_t)p * 128 + c];
        if (a != by4[(size_t)p * 128 + c]) d4++;
        if (a != by1[(size_t)p * 128 + c]) d1++;
        if (a == __float2half(0.0f)) unwritten++;
      }
      const int total = npool_used * 128;
      printf("kpool_write split invariance: %d/%d differ (4-token calls), %d/%d (1-token calls), "
             "%d unwritten %s\n", d4, total, d1, total, unwritten,
             (d4 == 0 && d1 == 0 && unwritten == 0) ? "PASS" : "FAIL");
      fails += !(d4 == 0 && d1 == 0 && unwritten == 0);
      // Rebuild the whole plane: the indexer_score case below reads all np pools of this fixture.
      CK(cudaMemset(draw, 0, tmax * 256 * 2)); CK(cudaMemset(dpk_nt, 0, np * 128 * 2));
      kpool_write(dik, dig, draw, draw_rows, dpk_nt, dape, 0, np * POOL, 0); CK(cudaDeviceSynchronize());
    }

    // The raw rows are a ring, so the same tokens must give the same pool keys from a small ring as
    // from a full plane. Positions 12..27 through a 20-row ring wrap it, which is the case a
    // chunk-sized ring hits in a long prefill.
    {
      const int n_ring = 20, pos0 = 12, n_tok = 16;
      std::vector<half> plane(np * 128), ring(np * 128);
      CK(cudaMemset(draw, 0, tmax * 256 * 2)); CK(cudaMemset(dpk_nt, 0, np * 128 * 2));
      kpool_write(dik, dig, draw, tmax, dpk_nt, dape, pos0, n_tok, 0); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(plane.data(), dpk_nt, plane.size() * 2, cudaMemcpyDeviceToHost));
      CK(cudaMemset(draw, 0, tmax * 256 * 2)); CK(cudaMemset(dpk_nt, 0, np * 128 * 2));
      kpool_write(dik, dig, draw, n_ring, dpk_nt, dape, pos0, n_tok, 0); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(ring.data(), dpk_nt, ring.size() * 2, cudaMemcpyDeviceToHost));
      int diff = 0, written = 0, clobbered = 0;
      for (int p = pos0 / POOL; p <= (pos0 + n_tok - 1) / POOL; p++)
        for (int c = 0; c < 128; c++) {
          if (plane[(size_t)p * 128 + c] != ring[(size_t)p * 128 + c]) diff++;
          if (plane[(size_t)p * 128 + c] != __float2half(0.0f)) written++;
        }
      // Pools whose groups were complete before this call must be left alone. A launch's grid starts
      // at block 0, so those blocks run and must bail out; if they instead recompute from recycled
      // ring slots they overwrite the earlier chunks' keys with garbage.
      for (int p = 0; p < pos0 / POOL; p++)
        for (int c = 0; c < 128; c++)
          if (ring[(size_t)p * 128 + c] != __float2half(0.0f)) clobbered++;
      printf("kpool_write ring (20 rows, positions %d..%d): %d differ, %d written, %d earlier "
             "entries clobbered %s\n", pos0, pos0 + n_tok - 1, diff, written, clobbered,
             (diff == 0 && written > 0 && clobbered == 0) ? "PASS" : "FAIL");
      fails += !(diff == 0 && written > 0 && clobbered == 0);
      CK(cudaMemset(draw, 0, tmax * 256 * 2)); CK(cudaMemset(dpk_nt, 0, np * 128 * 2));
      kpool_write(dik, dig, draw, draw_rows, dpk_nt, dape, 0, np * POOL, 0); CK(cudaDeviceSynchronize());
    }

    // indexer_score: m=2, positions 255 and 100
    int m = 2;
    std::vector<float> qi(m * 32 * 128), w(m * 32);
    for (auto& x : qi) x = fr(); for (auto& x : w) x = fr();
    auto qih = toh(qi), wh = toh(w);
    half *dqi, *dw, *dsc; int* dpos;
    CK(cudaMalloc(&dqi, qih.size() * 2)); CK(cudaMalloc(&dw, wh.size() * 2));
    CK(cudaMalloc(&dsc, m * np * 2)); CK(cudaMalloc(&dpos, m * 4));
    CK(cudaMemcpy(dqi, qih.data(), qih.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice));
    int pos[2] = {255, 100};
    CK(cudaMemcpy(dpos, pos, 8, cudaMemcpyHostToDevice));
    indexer_score(dqi, dpk_nt, dw, dpos, dsc, m, np, 0); CK(cudaDeviceSynchronize());
    std::vector<half> sc(m * np); CK(cudaMemcpy(sc.data(), dsc, sc.size() * 2, cudaMemcpyDeviceToHost));
    ref.assign(m * np, 0);
    for (int r = 0; r < m; r++) for (int p = 0; p < np; p++) {
      if (p * 4 + 3 > pos[r]) { ref[r * np + p] = -INFINITY; continue; }
      double tot = 0;
      for (int h = 0; h < 32; h++) {
        double dot = 0;
        for (int d = 0; d < 128; d++) dot += __half2float(pk[p * 128 + d]) * __half2float(qih[r * 32 * 128 + h * 128 + d]);
        dot /= std::sqrt(128.0);
        if (dot > 0) tot += __half2float(wh[r * 32 + h]) * dot;
      }
      ref[r * np + p] = tot / std::sqrt(32.0);
    }
    got.assign(sc.size(), 0);
    for (size_t i = 0; i < sc.size(); i++) got[i] = __half2float(sc[i]);
    // compare: -inf entries equal both
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); i++) {
      if (ref[i] == -INFINITY) { if (got[i] != -INFINITY) num += 10; continue; }
      num += std::fabs(got[i] - ref[i]); den += std::fabs(ref[i]) + 1e-6;
    }
    e = num / den;
    printf("indexer_score relerr %.2e %s\n", e, e < 3e-3 ? "PASS" : "FAIL"); fails += e >= 3e-3;
    cudaFree(dqi); cudaFree(dw); cudaFree(dsc); cudaFree(dpos); cudaFree(dik); cudaFree(dig); cudaFree(draw); cudaFree(dpk_nt); cudaFree(dape);
  }

  // ---- pool_expand ----
  {
    int m = 2, cur = 100;
    std::vector<int> pidx(m * 512), pos(m);
    for (int i = 0; i < 512; i++) { pidx[i] = rng() % 30; pidx[m * 512 - 1 - i % 3] = -1; }
    for (int i = 0; i < m * 512; i++) pidx[i] = (i % 7 == 0) ? -1 : rng() % 30;
    pos[0] = 255; pos[1] = 100;
    int *dp, *dpos, *draw;
    CK(cudaMalloc(&dp, pidx.size() * 4)); CK(cudaMalloc(&dpos, 8)); CK(cudaMalloc(&draw, m * (512 * 4 + 4) * 4));
    CK(cudaMemcpy(dp, pidx.data(), pidx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos, pos.data(), 8, cudaMemcpyHostToDevice));
    pool_expand(dp, dpos, draw, m, cur); CK(cudaDeviceSynchronize());
    std::vector<int> raw(m * (512 * 4 + 4)); CK(cudaMemcpy(raw.data(), draw, raw.size() * 4, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int r = 0; r < m; r++) {
      for (int i = 0; i < 512; i++) for (int j = 0; j < 4; j++) {
        int want = pidx[r * 512 + i] >= 0 ? pidx[r * 512 + i] * 4 + j : -1;
        if (want > cur) want = -1;
        if (raw[r * (512 * 4 + 4) + i * 4 + j] != want) ok = false;
      }
      for (int j = 0; j < 4; j++) {
        int want = cur - 3 + j; if (want < 0) want = -1;
        if (raw[r * (512 * 4 + 4) + 2048 + j] != want) ok = false;
      }
    }
    printf("pool_expand %s\n", ok ? "PASS" : "FAIL"); fails += !ok;
    cudaFree(dp); cudaFree(dpos); cudaFree(draw);
  }

  // ---- mla_sparse_decode + mla_dense_prefill ----
  {
    int T = 512, m = 1, kidx = 64;
    std::vector<float> ql(m * 64 * 512), ckv(T * 512);
    for (auto& x : ql) x = fr(); for (auto& x : ckv) x = fr() * 0.5f;
    auto qlh = toh(ql), ckvh = toh(ckv);
    std::vector<int> ridx(kidx);
    for (int i = 0; i < kidx; i++) ridx[i] = rng() % T;
    ridx[kidx - 2] = -1;
    half *dql, *dckv, *dlo; int* dri;
    CK(cudaMalloc(&dql, qlh.size() * 2)); CK(cudaMalloc(&dckv, ckvh.size() * 2));
    CK(cudaMalloc(&dlo, m * 64 * 512 * 2)); CK(cudaMalloc(&dri, kidx * 4));
    CK(cudaMemcpy(dql, qlh.data(), qlh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dckv, ckvh.data(), ckvh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dri, ridx.data(), kidx * 4, cudaMemcpyHostToDevice));
    mla_sparse_decode(dql, dckv, dri, kidx, dlo, m); CK(cudaDeviceSynchronize());
    std::vector<half> lo(m * 64 * 512); CK(cudaMemcpy(lo.data(), dlo, lo.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> ref(m * 64 * 512);
    for (int h = 0; h < 64; h++) {
      std::vector<double> sc(kidx); double mx = -1e30;
      for (int i = 0; i < kidx; i++) {
        if (ridx[i] < 0) { sc[i] = -INFINITY; continue; }
        double d = 0;
        for (int c = 0; c < 512; c++) d += __half2float(qlh[h * 512 + c]) * __half2float(ckvh[ridx[i] * 512 + c]);
        sc[i] = d * 0.0625; mx = std::max(mx, sc[i]);
      }
      double l = 0; std::vector<double> acc(512, 0);
      for (int i = 0; i < kidx; i++) {
        if (sc[i] == -INFINITY) continue;
        double p = std::exp(sc[i] - mx); l += p;
        for (int c = 0; c < 512; c++) acc[c] += p * __half2float(ckvh[ridx[i] * 512 + c]);
      }
      for (int c = 0; c < 512; c++) ref[h * 512 + c] = acc[c] / l;
    }
    std::vector<float> got(lo.size()); for (size_t i = 0; i < lo.size(); i++) got[i] = __half2float(lo[i]);
    float e = relerr(got, ref);
    printf("mla_sparse_decode relerr %.2e %s\n", e, e < 3e-3 ? "PASS" : "FAIL"); fails += e >= 3e-3;

    // dense prefill: qlen=16 rows starting pos 0; row q attends 0..q
    int qlen = 16;
    std::vector<float> qp(qlen * 64 * 512); for (auto& x : qp) x = fr();
    auto qph = toh(qp);
    half *dqp, *dlo2;
    CK(cudaMalloc(&dqp, qph.size() * 2)); CK(cudaMalloc(&dlo2, qlen * 64 * 512 * 2));
    CK(cudaMemcpy(dqp, qph.data(), qph.size() * 2, cudaMemcpyHostToDevice));
    mla_dense_prefill(dqp, dckv, dlo2, qlen, 0); CK(cudaDeviceSynchronize());
    std::vector<half> lo2(qlen * 64 * 512); CK(cudaMemcpy(lo2.data(), dlo2, lo2.size() * 2, cudaMemcpyDeviceToHost));
    ref.assign(qlen * 64 * 512, 0);
    for (int r = 0; r < qlen; r++) for (int h = 0; h < 64; h++) {
      int tend = r; double mx = -1e30; std::vector<double> sc(tend + 1);
      for (int t = 0; t <= tend; t++) {
        double d = 0;
        for (int c = 0; c < 512; c++) d += __half2float(qph[r * 64 * 512 + h * 512 + c]) * __half2float(ckvh[t * 512 + c]);
        sc[t] = d * 0.0625; mx = std::max(mx, sc[t]);
      }
      double l = 0; std::vector<double> acc(512, 0);
      for (int t = 0; t <= tend; t++) {
        double p = std::exp(sc[t] - mx); l += p;
        for (int c = 0; c < 512; c++) acc[c] += p * __half2float(ckvh[t * 512 + c]);
      }
      for (int c = 0; c < 512; c++) ref[r * 64 * 512 + h * 512 + c] = acc[c] / l;
    }
    got.assign(lo2.size(), 0); for (size_t i = 0; i < lo2.size(); i++) got[i] = __half2float(lo2[i]);
    e = relerr(got, ref);
    printf("mla_dense_prefill relerr %.2e %s\n", e, e < 3e-3 ? "PASS" : "FAIL"); fails += e >= 3e-3;
    cudaFree(dql); cudaFree(dckv); cudaFree(dlo); cudaFree(dri); cudaFree(dqp); cudaFree(dlo2);
  }

// ---- mla_sparse_decode: K=2 head-pair kernel must be bit-identical to K=1 (same per-head math) ----
  {
    int m = 3, kidx = 300;
    std::vector<float> qn(m * 64 * 512), ck(2048 * 512);
    for (auto& x : qn) x = fr(); for (auto& x : ck) x = fr() * 0.3f;
    auto qh = toh(qn), ch = toh(ck);
    std::vector<int> ri(m * kidx);
    for (int r = 0; r < m; r++) for (int i = 0; i < kidx; i++)
      ri[r * kidx + i] = (i % 7 == 3 && i > 40) ? -1 : (int)(rng() % 2048);   // holes exercise the skip path
    half *dq, *dc, *dlo1, *dlo2; int *dri;
    CK(cudaMalloc(&dq, qh.size() * 2)); CK(cudaMalloc(&dc, ch.size() * 2));
    CK(cudaMalloc(&dri, ri.size() * 4));
    CK(cudaMalloc(&dlo1, (size_t)m * 64 * 512 * 2)); CK(cudaMalloc(&dlo2, (size_t)m * 64 * 512 * 2));
    CK(cudaMemcpy(dq, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, ch.data(), ch.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dri, ri.data(), ri.size() * 4, cudaMemcpyHostToDevice));
    setenv("HELIOS_SPARSE_K", "1", 1);
    mla_sparse_decode(dq, dc, dri, kidx, dlo1, m, 0); CK(cudaDeviceSynchronize());
    setenv("HELIOS_SPARSE_K", "2", 1);
    mla_sparse_decode(dq, dc, dri, kidx, dlo2, m, 0); CK(cudaDeviceSynchronize());
    std::vector<half> o1(m * 64 * 512), o2(m * 64 * 512);
    CK(cudaMemcpy(o1.data(), dlo1, o1.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o2.data(), dlo2, o2.size() * 2, cudaMemcpyDeviceToHost));
    // K=2 stages per-warp partials in fp16 (to keep 2 blocks/SM), so compare with a tolerance:
    // the merged output is written as fp16 regardless, so the error stays at output-rounding scale.
    double num = 0, den = 0; int nmax = 0;
    for (size_t i = 0; i < o1.size(); i++) {
      double a = __half2float(o1[i]), b = __half2float(o2[i]);
      num += std::fabs(a - b); den += std::fabs(b) + 1e-6;
      if (std::fabs(a - b) > 2e-2) nmax++;
    }
    double rel = num / den;
    printf("mla_sparse_decode K=1 vs K=2: relerr %.2e, %d/%zu beyond 2e-2 %s\n", rel, nmax,
           o1.size(), (rel < 1e-3 && nmax == 0) ? "PASS" : "FAIL");
    fails += !(rel < 1e-3 && nmax == 0);
    cudaFree(dq); cudaFree(dc); cudaFree(dri); cudaFree(dlo1); cudaFree(dlo2);
  }



  // ---- indexer_score tensor-core path vs scalar (-inf mask + finite values) ----
  {
    int m = 48, npools = 600;
    std::vector<float> qf(m * 32 * 128), kf(128 * npools), wf(m * 32);
    for (auto& x : qf) x = fr(); for (auto& x : kf) x = fr() * 0.4f; for (auto& x : wf) x = fr();
    std::vector<int> qp(m);
    for (int r = 0; r < m; r++) qp[r] = 900 + r * 5;
    auto qh = toh(qf), kh = toh(kf), wh = toh(wf);
    // build the pool-major mirror [p][128] from the [d][p] plane the scalar kernel reads
    std::vector<half> knt((size_t)npools * 128);
    for (int d = 0; d < 128; d++) for (int p = 0; p < npools; p++) knt[(size_t)p * 128 + d] = kh[(size_t)d * npools + p];
    half *dq, *dk, *dknt, *dw, *s1, *s2; int *dp;
    CK(cudaMalloc(&dq, qh.size() * 2)); CK(cudaMalloc(&dk, kh.size() * 2));
    CK(cudaMalloc(&dknt, knt.size() * 2)); CK(cudaMalloc(&dw, wh.size() * 2));
    CK(cudaMalloc(&dp, qp.size() * 4));
    CK(cudaMalloc(&s1, (size_t)m * npools * 2)); CK(cudaMalloc(&s2, (size_t)m * npools * 2));
    CK(cudaMemcpy(dq, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dknt, knt.data(), knt.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dp, qp.data(), qp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(s2, 0, (size_t)m * npools * 2));
    indexer_score_legacy(dq, dk, dw, dp, s1, m, npools, npools, 0); CK(cudaDeviceSynchronize());
    indexer_score_mma(dq, dknt, dw, dp, s2, m, npools, 0); CK(cudaDeviceSynchronize());
    std::vector<half> o1((size_t)m * npools), o2((size_t)m * npools);
    CK(cudaMemcpy(o1.data(), s1, o1.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o2.data(), s2, o2.size() * 2, cudaMemcpyDeviceToHost));
    int bad = 0, maskbad = 0, ninf1 = 0, ninf2 = 0; float mx = 0; int shown = 0;
    for (size_t i = 0; i < o1.size(); i++) {
      float a = __half2float(o1[i]), b = __half2float(o2[i]);
      bool ia = a < -1e30f, ib = b < -1e30f;
      if (ia) ninf1++; if (ib) ninf2++;
      if (ia != ib) { maskbad++; continue; }
      if (ia) continue;
      float d = std::fabs(a - b); mx = std::max(mx, d);
      if (d > 4e-3f) { bad++;
        if (shown < 5) { shown++;
          printf("  MMA MISMATCH r=%zu p=%zu scalar=%.5f mma=%.5f\n", i / npools, i % npools, a, b); } }
    }
    printf("indexer_score mma vs scalar: %d/%zu differ (max|d|=%.2e), mask mismatches %d, -inf %d/%d %s\n",
           bad, o1.size(), mx, maskbad, ninf1, ninf2,
           (bad == 0 && maskbad == 0 && ninf1 == ninf2) ? "PASS" : "FAIL");
    fails += (bad != 0 || maskbad != 0 || ninf1 != ninf2);
    cudaFree(dq); cudaFree(dk); cudaFree(dknt); cudaFree(dw); cudaFree(dp); cudaFree(s1); cudaFree(s2);
  }

  printf(fails ? "ATTN PARITY: %d FAIL\n" : "ATTN PARITY: ALL PASS\n", fails);
  return fails;
}