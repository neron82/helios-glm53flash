#pragma once
// Dual-3090 device management: contexts, carve-out pools, pinned DMA rings, streams.
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace helios {

constexpr int N_GPU = 2;
constexpr size_t GiB = 1ull << 30;

struct DevStats {
  int index;
  size_t total_vram = 0;
  size_t free_vram = 0;
  int link_width = 0;       // PCIe negotiated width (from sysfs)
  int link_speed_gts = 0;   // GT/s
  bool p2p_to_other = false;
};

class Device {
public:
  bool init(int idx, size_t reserve_for_pool);
  const DevStats& stats() const { return stats_; }
  cudaStream_t stream(int i) const { return streams_[i]; }      // 0=compute, 1=DMA-in, 2=DMA-out, 3=aux
  cudaEvent_t event(int i) const { return events_[i]; }
  static constexpr int N_STREAMS = 4;

  // VRAM pool: simple bump allocator over a big cudaMalloc; fragmentation-free carve-outs.
  void* alloc(size_t bytes, int align = 256);
  size_t pool_used() const { return pool_off_; }
  size_t pool_free() const { return pool_size_ - pool_off_; }

  // Pinned host staging ring for RAM->GPU expert streaming.
  struct Ring {
    cudaEvent_t* done = nullptr;   // per-slot completion events (n_slots)
    size_t slot_bytes = 0;
    int n_slots = 0;
    int write_slot = 0;        // CPU writes into this slot then enqueues DMA
    int armed_slot = 0;        // DMA scheduled up to here
    char* base = nullptr;      // pinned ring base pointer
  };
  Ring* make_ring(size_t slot_bytes, int n_slots);
  std::vector<Ring*>& rings() { return rings_; }

  // Async DMA copy from pinned ring slot -> device dst, on DMA stream, records event.
  void enqueue_h2d(void* dst, const void* pinned_src, size_t bytes, int slot_idx);
  bool slot_done(Ring& r, int slot_idx) const;

  int phys_idx() const { return stats_.index; }   // physical CUDA device index

  friend class Engine;   // P2P flag writes
private:
  DevStats stats_;
  cudaStream_t streams_[N_STREAMS] = {};
  cudaEvent_t events_[8] = {};
  void* pool_ = nullptr;
  size_t pool_size_ = 0, pool_off_ = 0;
  std::vector<Ring*> rings_;
};

class Engine {
public:
  bool init();            // both devices, P2P probe, uctx
  Device& gpu(int i) { return devs_[i]; }
  static Engine& instance();
private:
  Device devs_[N_GPU];
};

}  // namespace helios