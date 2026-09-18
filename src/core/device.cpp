#include "core/device.hpp"
#include "cuda/cuda_shim.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <algorithm>

namespace helios {

Engine& Engine::instance() { static Engine e; return e; }

static int pci_bw_mbps(const std::string& bdf) {
  // bdf like "0000:07:00.0"
  std::string base = "/sys/bus/pci/devices/" + bdf + "/";
  int width = 0, speed = 0;
  if (std::ifstream f(base + "current_link_width"); f) f >> width;
  if (std::ifstream f2(base + "current_link_speed"); f2) { std::string s; getline(f2, s); speed = atoi(s.c_str()); }
  return width * speed; // MT/s ~ MB/s (per-lane GT/s ≈ MB/s ignoring encoding)
}

bool Device::init(int idx, size_t reserve_for_pool) {
  HELIOS_CUDA_CHECK(cudaSetDevice(idx));
  cudaDeviceProp prop{};
  HELIOS_CUDA_CHECK(cudaGetDeviceProperties(&prop, idx));
  stats_.index = idx;
  stats_.total_vram = prop.totalGlobalMem;
  size_t f = 0;
  HELIOS_CUDA_CHECK(cudaMemGetInfo(&f, nullptr));
  stats_.free_vram = f;
  char bdf[64] = {};
  if (cudaDeviceGetPCIBusId(bdf, sizeof bdf, idx) == cudaSuccess) {
    stats_.link_speed_gts = 0; stats_.link_width = 0;
    std::string base = std::string("/sys/bus/pci/devices/") + bdf + "/";
    std::ifstream fw(base + "current_link_width"); if (fw) fw >> stats_.link_width;
    std::ifstream fs(base + "current_link_speed");  if (fs) { std::string s; getline(fs, s); stats_.link_speed_gts = atoi(s.c_str()); }
  }
  for (int i = 0; i < N_STREAMS; i++) HELIOS_CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
  for (int i = 0; i < 8; i++) HELIOS_CUDA_CHECK(cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming));
  // carve-out pool: allocate aggressively (bump allocator avoids fragmentation)
  size_t want = std::min(reserve_for_pool, stats_.free_vram - (size_t)(2ull<<20));
  pool_ = nullptr;
  if (want > (1ull<<20)) {
    if (cudaMalloc(&pool_, want) == cudaSuccess) pool_size_ = want;
    else pool_ = nullptr, pool_size_ = 0;
  }
  // L2 persistence limit (hot-expert/KV window support)
  int persist = 0;
  cudaDeviceGetAttribute(&persist, cudaDevAttrMaxPersistingL2CacheSize, idx);
  if (persist > 0) cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, std::min(persist, 64<<20));
  return true;
}

void* Device::alloc(size_t bytes, int align) {
  // Allocations must land on THIS device regardless of the caller thread's current device.
  HELIOS_CUDA_CHECK(cudaSetDevice(stats_.index));
  HELIOS_CUDA_CHECK(cudaSetDevice(stats_.index));
  bytes = (bytes + align - 1) & ~(size_t)(align - 1);
  if (pool_ && pool_off_ + bytes <= pool_size_) { void* p = (char*)pool_ + pool_off_; pool_off_ += bytes; return p; }
  void* p = nullptr;
  int cur = -1; cudaGetDevice(&cur);
  cudaError_t e = cudaMalloc(&p, bytes);
  if (e != cudaSuccess) {
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "[device] cudaMalloc(%.2f MB) failed on device %d (ctx device %d, free %.2f GB): %s\n",
            bytes / 1048576.0, stats_.index, cur, fr / 1073741824.0, cudaGetErrorString(e));
    abort();
  }
  return p;
}

Device::Ring* Device::make_ring(size_t slot_bytes, int n_slots) {
  HELIOS_CUDA_CHECK(cudaSetDevice(stats_.index));
  Ring* r = new Ring();
  r->slot_bytes = slot_bytes;
  r->n_slots = n_slots;
  HELIOS_CUDA_CHECK(cudaMallocHost(&r->base, slot_bytes * n_slots));
  r->done = new cudaEvent_t[n_slots];
  for (int i = 0; i < n_slots; i++) HELIOS_CUDA_CHECK(cudaEventCreateWithFlags(&r->done[i], cudaEventDisableTiming));
  rings_.push_back(r);
  return r;
}

void Device::enqueue_h2d(void* dst, const void* pinned_src, size_t bytes, int slot_idx) {
  HELIOS_CUDA_CHECK(cudaSetDevice(stats_.index));
  Ring* r = nullptr;
  for (auto* rr : rings_) if (((char*)rr->base <= (char*)pinned_src && (char*)pinned_src < (char*)rr->base + rr->slot_bytes * rr->n_slots)) { r = rr; break; }
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream(1)));
  if (r && slot_idx >= 0 && slot_idx < r->n_slots) HELIOS_CUDA_CHECK(cudaEventRecord(r->done[slot_idx], stream(1)));
}

bool Device::slot_done(Ring& r, int slot_idx) const {
  return cudaEventQuery(r.done[slot_idx]) == cudaSuccess;
}

// Measured pinned-memory H2D bandwidth (GB/s). Robust where sysfs link data is unavailable
// (a failed sysfs read used to rank the x16 card as "slowest" and put the slot pool on the x4).
static double measure_h2d_gbps(int dev) {
  cudaSetDevice(dev);
  const size_t bytes = 256ull << 20;
  char* pin = nullptr;
  if (cudaHostAlloc((void**)&pin, bytes, cudaHostAllocDefault) != cudaSuccess) return 0.0;
  memset(pin, 1, bytes);
  void* d = nullptr;
  if (cudaMalloc(&d, bytes) != cudaSuccess) { cudaFreeHost(pin); return 0.0; }
  cudaStream_t st = nullptr;
  cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  cudaMemcpyAsync(d, pin, bytes, cudaMemcpyHostToDevice, st);
  cudaStreamSynchronize(st);
  auto t0 = std::chrono::steady_clock::now();
  const int reps = 4;
  for (int i = 0; i < reps; i++) cudaMemcpyAsync(d, pin, bytes, cudaMemcpyHostToDevice, st);
  cudaStreamSynchronize(st);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  cudaFree(d); cudaFreeHost(pin); cudaStreamDestroy(st);
  return ms > 0 ? (double)bytes * reps / (ms * 1e6) : 0.0;
}

bool Engine::init() {
  // rank devices by PCIe bandwidth: slow link = rank 0 (trunk), fast = rank 1 (streaming)
  int idx[N_GPU];
  HELIOS_CUDA_CHECK(cudaGetDeviceCount(&idx[0])); // reuse var: count
  if (idx[0] < N_GPU) { fprintf(stderr, "need %d GPUs\n", N_GPU); return false; }
  int order[N_GPU]; for (int i = 0; i < N_GPU; i++) order[i] = i;
  double bw[N_GPU];
  for (int i = 0; i < N_GPU; i++) {
    bw[i] = measure_h2d_gbps(i);   // measured: sysfs link data is unreliable on this board
    char bdf[64] = {};
    if (cudaDeviceGetPCIBusId(bdf, sizeof bdf, i) == cudaSuccess)
      fprintf(stderr, "[device] gpu%d (%s) H2D %.2f GB/s\n", i, bdf, bw[i]);
  }
  if (const char* env = getenv("HELIOS_GPU_ORDER")) {   // explicit "trunk,slots" physical indices
    int a = -1, b = -1;
    if (sscanf(env, "%d,%d", &a, &b) == 2 && a != b && a >= 0 && a < N_GPU && b >= 0 && b < N_GPU) {
      devs_[0].init(a, 0);
      devs_[1].init(b, 0);
      cudaSetDevice(a);
      fprintf(stderr, "[device] forced order: trunk=gpu%d slots=gpu%d\n", a, b);
      return true;
    }
  }
  std::sort(order, order + N_GPU, [&](int a, int b){ return bw[a] < bw[b]; });
  // rank0 = slowest (trunk GPU0 role), rank1 = fastest (GPU1 role)
  devs_[0].init(order[0], 0);   // trunk pool sized later by model planner
  devs_[1].init(order[1], 0);
  // P2P probe (PHB consumer boards: usually unavailable; keep host-pinned bounce then)
  int a01 = 0, a10 = 0;
  cudaDeviceCanAccessPeer(&a01, order[0], order[1]);
  cudaDeviceCanAccessPeer(&a10, order[1], order[0]);
  if (a01 && a10) {
    HELIOS_CUDA_CHECK(cudaSetDevice(order[0]));
    if (cudaDeviceEnablePeerAccess(order[1], 0) == cudaSuccess) {
      HELIOS_CUDA_CHECK(cudaSetDevice(order[1]));
      cudaDeviceEnablePeerAccess(order[0], 0);
      devs_[0].stats_.p2p_to_other = devs_[1].stats_.p2p_to_other = true;
    }
  }
  cudaSetDevice(order[0]);
  return true;
}

} // namespace helios