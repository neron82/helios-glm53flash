#pragma once
// Expert slot manager: GPU1 slot pool holding expert slabs streamed from the RAM arena.
// v1: global LRU with optional heat-pinned slots; async slab copies on the DMA stream
// with per-slot done events. Statistics for hit-rate tuning.
#include "core/model.hpp"
#include "core/device.hpp"
#include <unordered_map>
#include <string>
#include <vector>

namespace helios {

class SlotMgr {
public:
  // pool_bytes: VRAM to reserve for expert slabs on GPU1; stride from the model.
  bool init(const Model& m, size_t pool_bytes);
  // Reuse the pool the loader already allocated (Model::slot_pool / n_slots).
  // census_path (optional): persistent per-(layer,expert) request census. Loaded at init, saved
  // periodically and at exit. Ranking the pool by this long-run census instead of intra-run heat is
  // what makes residency reach the coverage curve the skew probe prints: heat resets to 0 on every
  // eviction, so the 60% pinned share used to be filled from whichever phase flooded last.
  bool init_from_pool(const Model& m, void* pool, int n_slots, const char* census_path = nullptr);

  int n_slots() const { return n_slots_; }
  // Diagnostics for a (layer, expert) that should be resident but is not.
  void diag_expert(int layer, int expert) const;
  // Last-resort acquisition: evicts regardless of pinning (and of busy slots not in the current
  // step) so an expert the kernel needs can always be made resident. Returns the slot or -1.
  int force_resident(int layer, int expert);
  void* zero_slab() const { return zero_slab_; }
  void set_zero_slab(void* p) { zero_slab_ = p; }
  size_t stride() const { return stride_; }
  const DevStats& stats() const { return stats_; }

  // Slots are keyed by the TENSOR-NAMESPACE layer index (Model::slab(layer, expert)).
  // Slot holding (layer, expert) or -1 if absent.
  int find(int layer, int expert);
  // Slot index for (layer, expert), streaming the slab from RAM if needed.
  // Returns a slot whose contents are NOT yet guaranteed until sync_copies()/ensure().
  int acquire(int layer, int expert);
  // Wait until all outstanding slab copies finished (events on the aux stream).
  void sync_copies();

  // Slot bookkeeping: mark slots used by the current token/forward (prevents eviction),
  // and touch heat counters for pinned-set statistics.
  // Routing skew probe: per (layer, expert) request counts plus the coverage curve you would get
  // from pinning only the top K keys. Answers "how much residency is even reachable here".
  void note_request(int layer, int expert);
  void print_skew() const;
  // Census rank: entries whose long-run request count is in the top max_pinned_ are pinned, and
  // eviction always prefers the lowest-ranked non-pinned slot.
  bool is_hot(int layer, int expert) const;
  void recompute_hot();
  bool save_census();
  void begin_step();
  void end_step();

  char* slot_ptr(int slot) const { return pool_ + (size_t)slot * stride_; }

  // ---- statistics ----
  struct Stats {
    uint64_t hits = 0, misses = 0, evictions = 0, h2d_bytes = 0, pinned_hits = 0;
    uint64_t lookups = 0, resident_hits = 0;   // residency hit rate across expert requests
    double last_ms = 0;
  };
  Stats& stats_counters() { return st_; }
  void print_stats();

private:
  struct Slot {
    int layer = -1, expert = -1;
    uint32_t heat = 0;
    uint64_t last_used = 0;
    bool pinned = false;
    bool busy = false;             // copy in flight for this step
  };
  int evict_one();
  std::string census_path_;
  std::vector<uint8_t> hot_;     // per (layer,expert): in the census top max_pinned_
  int hot_count_ = 0;
  uint64_t since_rank_ = 0;
  // end_step() runs once per LAYER per step (~43x per token), so any per-N-step schedule there
  // becomes a per-token cost. Rank refreshes and census saves are therefore throttled by wall clock.
  double last_rank_ = -1e9, last_save_ = -1e9;
  void start_copy(int slot, int layer, int expert);

  const Model* m_ = nullptr;
  char* pool_ = nullptr;
  int n_slots_ = 0;
  size_t stride_ = 0;
  std::vector<Slot> slots_;
  mutable std::vector<uint32_t> freq_;    // per (layer, expert) request counts for the skew probe
  int n_layer_ = 0;
  void* zero_slab_ = nullptr;   // all-zero slab used if an expert cannot be made resident
  std::unordered_map<uint64_t, int> map_;   // (layer<<32|expert) -> slot
  std::vector<int> used_;                    // slots used this step
  std::vector<int> pending_;                 // slots with in-flight copies
  uint64_t clock_ = 0;
  int pinned_count_ = 0;
  int max_pinned_ = 0;            // cap on the pinned (static) share of the pool
  DevStats stats_{};
  Stats st_{};
};

}  // namespace helios