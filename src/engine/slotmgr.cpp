#include "engine/slotmgr.hpp"
#include "../cuda/cuda_shim.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace helios {

bool SlotMgr::init(const Model& m, size_t pool_bytes) {
  m_ = &m;
  stride_ = m.slot_stride;
  n_slots_ = (int)(pool_bytes / stride_);
  if (n_slots_ < 16) { fprintf(stderr, "[slots] pool too small\n"); return false; }
  n_layer_ = (int)m.cfg.n_layers + 1;
  freq_.assign((size_t)n_layer_ * 288, 0);
  Device& g1 = Engine::instance().gpu(1);
  cudaSetDevice(g1.phys_idx());
  pool_ = (char*)g1.alloc((size_t)n_slots_ * stride_, 4096);
  if (!pool_) { fprintf(stderr, "[slots] OOM for %d slots (%.1f GB)\n", n_slots_,
                          (double)((size_t)n_slots_ * stride_) / GiB); return false; }
  slots_.assign(n_slots_, Slot{});
  max_pinned_ = n_slots_ * 60 / 100;
  map_.reserve(n_slots_ * 2);
  stats_ = g1.stats();
  printf("[slots] %d slots x %.3f MB = %.2f GB on gpu%d\n", n_slots_,
         stride_ / 1048576.0, (double)((size_t)n_slots_ * stride_) / GiB, g1.phys_idx());
  return true;
}

bool SlotMgr::init_from_pool(const Model& m, void* pool, int n_slots, const char* census_path) {
  m_ = &m;
  stride_ = m.slot_stride;
  pool_ = (char*)pool;
  n_layer_ = (int)m.cfg.n_layers + 1;
  freq_.assign((size_t)n_layer_ * 288, 0);
  n_slots_ = n_slots;
  if (!pool_ || n_slots_ < 16) { fprintf(stderr, "[slots] invalid pre-allocated pool\n"); return false; }
  slots_.assign(n_slots_, Slot{});
  max_pinned_ = n_slots_ * 60 / 100;
  map_.reserve(n_slots_ * 2);
  // Persistent census: rank the pool by long-run request counts, not by intra-run heat. Heat is
  // reset to 0 by every eviction, so under the old rule the pinned 60% was whatever phase flooded
  // last (a prefill chunk issues ~49k requests, a 120-token decode ~41k).
  if (census_path && *census_path) {
    census_path_ = census_path;
    FILE* f = fopen(census_path_.c_str(), "rb");
    if (f) {
      char magic[8]; uint32_t nl = 0, ne = 0;
      if (fread(magic, 1, 8, f) == 8 && !memcmp(magic, "HELIOSC1", 8) &&
          fread(&nl, 4, 1, f) == 1 && fread(&ne, 4, 1, f) == 1 &&
          (size_t)nl * ne == freq_.size()) {
        if (fread(freq_.data(), 4, freq_.size(), f) == freq_.size()) {
          uint64_t tot = 0;
          for (uint32_t c : freq_) tot += c;
          printf("[slots] census loaded: %s (%llu requests over %u x %u)\n",
                 census_path_.c_str(), (unsigned long long)tot, nl, ne);
        }
      } else {
        freq_.assign(freq_.size(), 0);
        fprintf(stderr, "[slots] census %s ignored (format/layer mismatch)\n", census_path_.c_str());
      }
      fclose(f);
    }
  }
  hot_.assign(freq_.size(), 0);
  recompute_hot();
  stats_ = Engine::instance().gpu(1).stats();
  printf("[slots] reusing pool: %d slots x %.3f MB = %.2f GB\n", n_slots_, stride_ / 1048576.0,
         (double)((size_t)n_slots_ * stride_) / GiB);
  return true;
}

bool SlotMgr::is_hot(int layer, int expert) const {
  if (hot_.empty() || layer < 0 || layer >= n_layer_ || expert < 0 || expert >= 288) return false;
  return hot_[(size_t)layer * 288 + expert] != 0;
}

// Mark the top max_pinned_ census entries. Cost is a sort of ~12k entries; called on a coarse
// schedule rather than per step.
void SlotMgr::recompute_hot() {
  if (freq_.empty()) return;
  std::fill(hot_.begin(), hot_.end(), 0);
  std::vector<uint32_t> idx;
  idx.reserve(freq_.size());
  for (size_t i = 0; i < freq_.size(); i++) if (freq_[i]) idx.push_back((uint32_t)i);
  size_t k = (size_t)max_pinned_;
  hot_count_ = 0;
  if (idx.size() > k) {
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](uint32_t a, uint32_t b) { return freq_[a] > freq_[b]; });
    idx.resize(k);
  }
  for (uint32_t i : idx) { hot_[i] = 1; hot_count_++; }
}

bool SlotMgr::save_census() {
  if (census_path_.empty() || freq_.empty()) return false;
  std::string tmp = census_path_ + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  uint32_t nl = (uint32_t)n_layer_, ne = 288;
  bool ok = fwrite("HELIOSC1", 1, 8, f) == 8 && fwrite(&nl, 4, 1, f) == 1 &&
            fwrite(&ne, 4, 1, f) == 1 && fwrite(freq_.data(), 4, freq_.size(), f) == freq_.size();
  fclose(f);
  if (!ok) { remove(tmp.c_str()); return false; }
  return rename(tmp.c_str(), census_path_.c_str()) == 0;
}

void SlotMgr::note_request(int layer, int expert) {
  if (freq_.empty()) return;
  if (layer < 0 || layer >= n_layer_ || expert < 0 || expert >= 288) return;
  uint32_t& c = freq_[(size_t)layer * 288 + expert];
  if (c < 0xffffffffu) c++;
}

int SlotMgr::force_resident(int layer, int expert) {
  int slot = find(layer, expert);
  if (slot >= 0) return slot;
  for (int attempt = 0; attempt < 4; attempt++) {
    int best = -1;
    uint64_t best_t = ~0ull;
    for (int i = 0; i < n_slots_; i++) {
      Slot& s = slots_[i];
      bool in_use = false;
      for (int u : used_) if (u == i) { in_use = true; break; }
      if (in_use) continue;
      if (s.last_used < best_t) { best_t = s.last_used; best = i; }
    }
    if (best < 0) break;
    Slot& s = slots_[best];
    if (s.pinned) { s.pinned = false; if (pinned_count_ > 0) pinned_count_--; }
    s.busy = false;
    slot = acquire(layer, expert);
    if (slot >= 0) return slot;
  }
  return -1;
}

void SlotMgr::diag_expert(int layer, int expert) const {
  int in_slots = 0, in_used = 0, busy = 0, pinned = 0, mapped = 0;
  uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert;
  for (int i = 0; i < n_slots_; i++) {
    const Slot& s = slots_[i];
    if (s.layer == layer && s.expert == expert) in_slots++;
    if (s.busy) busy++;
    if (s.pinned) pinned++;
  }
  for (int i : used_) if (slots_[i].layer == layer && slots_[i].expert == expert) in_used++;
  auto it = map_.find(key);
  if (it != map_.end()) {
    const Slot& s = slots_[it->second];
    mapped = 1;
    fprintf(stderr, "[diag] (L%d,E%d) map->slot %d: layer=%d expert=%d busy=%d pinned=%d\n", layer,
            expert, it->second, s.layer, s.expert, (int)s.busy, (int)s.pinned);
  }
  fprintf(stderr, "[diag] (L%d,E%d) in_slots=%d in_used=%d mapped=%d | pool busy=%d pinned=%d "
                  "used=%zu pending=%zu n_slots=%d\n",
          layer, expert, in_slots, in_used, mapped, busy, pinned, used_.size(), pending_.size(),
          n_slots_);
}

void SlotMgr::print_skew() const {
  if (freq_.empty()) return;
  size_t tot = 0;
  std::vector<uint32_t> f;
  f.reserve(freq_.size());
  for (uint32_t c : freq_) { if (c) { f.push_back(c); tot += c; } }
  if (!tot) return;
  std::sort(f.begin(), f.end(), std::greater<uint32_t>());
  printf("[skew] distinct=%zu requests=%zu | top-K share of all requests:", f.size(), tot);
  for (double frac : {0.10, 0.25, 0.50, 1.00}) {
    size_t k = (size_t)(f.size() * frac);
    if (k == 0) k = 1;
    if (k > f.size()) k = f.size();
    size_t acc = 0;
    for (size_t i = 0; i < k; i++) acc += f[i];
    printf(" %.0f%%:%.0f%%", frac * 100, 100.0 * acc / tot);
  }
  printf("\n");
}

int SlotMgr::find(int layer, int expert) {
  auto it = map_.find(((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert);
  if (it == map_.end()) { st_.lookups++; return -1; }
  st_.lookups++; st_.resident_hits++;
  Slot& s = slots_[it->second];
  s.last_used = ++clock_;
  if (s.heat < 0xffff) s.heat++;
  // Pin only up to a fraction of the pool: the remainder stays a dynamic LRU so the working set
  // can still adapt when routing changes.
  // Admission is two-touch: acquire() leaves the slot unpinned, and a later hit promotes it if the
  // census ranks it hot. That keeps one-off prefill experts from consuming the pinned share.
  if (is_hot(layer, expert)) {
    if (!s.pinned && pinned_count_ < max_pinned_) { s.pinned = true; pinned_count_++; st_.pinned_hits++; }
  } else if (s.pinned) {
    s.pinned = false; pinned_count_--;
  }
  return it->second;
}

int SlotMgr::evict_one() {
  // Free slot first, then lowest heat among occupied (LFU), then anything non-busy.
  for (int i = 0; i < n_slots_; i++)
    if (slots_[i].layer < 0 && !slots_[i].busy) { slots_[i].busy = true; return i; }
  int best = -1;
  uint64_t best_score = ~0ull;
  uint64_t best_t = ~0ull;
  for (int i = 0; i < n_slots_; i++) {
    Slot& s = slots_[i];
    if (s.busy || s.pinned) continue;
    // Rank by the persistent census (0 for never-requested), tie-broken by heat then recency. An
    // evicted slot keeps its census score, so the pool converges on the long-run hot set instead of
    // re-learning every expert from zero.
    uint64_t sc = (s.layer >= 0 && (size_t)s.layer * 288 + s.expert < freq_.size())
                      ? freq_[(size_t)s.layer * 288 + s.expert] : 0ull;
    uint64_t h = s.heat;
    if (sc < best_score || (sc == best_score && h < best_t)) {
      best_score = sc; best_t = h; best = i;
    }
  }
  if (best < 0) {                       // all pinned/busy: drop the least recently used pinned slot
    for (int i = 0; i < n_slots_; i++) {
      Slot& s = slots_[i];
      if (s.busy) continue;
      if (s.last_used < best_t) { best_t = s.last_used; best = i; }
    }
    if (best >= 0) { slots_[best].pinned = false; pinned_count_--; }
  }
  if (best >= 0) {
    Slot& s = slots_[best];
    s.busy = true;
    if (s.layer >= 0) map_.erase(((uint64_t)(uint32_t)s.layer << 32) | (uint32_t)s.expert);
    s.layer = -1; s.expert = -1; s.heat = 0;
    st_.evictions++;
  }
  return best;
}

void SlotMgr::start_copy(int slot, int layer, int expert) {
  const char* src = m_->slab(layer, expert);
  Device& g1 = Engine::instance().gpu(1);
  // Measured: spreading these over two DMA streams changes nothing (70.4 tok/s both ways), so the
  // ~4.1GB/s of prefill-time slot streaming is a memory-subsystem limit (scattered 6MB reads out of
  // a 75GB pinned arena), not a limit on transfers in flight. Keep one stream and one event.
  cudaStream_t dma = g1.stream(1);
  HELIOS_CUDA_CHECK(cudaMemcpyAsync(pool_ + (size_t)slot * stride_, src, stride_,
                                    cudaMemcpyHostToDevice, dma));
  HELIOS_CUDA_CHECK(cudaEventRecord(g1.event(1), dma));
  pending_.push_back(slot);
  st_.h2d_bytes += stride_;
  st_.misses++;
}

int SlotMgr::acquire(int layer, int expert) {
  uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert;
  auto it = map_.find(key);
  if (it != map_.end()) {
    Slot& s = slots_[it->second];
    s.last_used = ++clock_;
    s.heat++;
    used_.push_back(it->second);
    st_.hits++;
    if (s.pinned) st_.pinned_hits++;
    return it->second;
  }
  int slot = evict_one();
  if (slot < 0) {
    // Every candidate was busy or pinned. Slots are marked busy only while their copy is in
    // flight, so drain the copies and retry before giving up; a running MoE kernel has already
    // been synchronised by the caller (the previous layer's y transfer), so reusing its slots here
    // is safe.
    sync_copies();
    slot = evict_one();
  }
  if (slot < 0) {   // everything pinned: drop the least recently used pinned slot
    uint64_t best_t = ~0ull;
    int best = -1;
    for (int i = 0; i < n_slots_; i++) {
      const Slot& s = slots_[i];
      if (s.busy) continue;
      if (s.last_used < best_t) { best_t = s.last_used; best = i; }
    }
    if (best >= 0) {
      slots_[best].pinned = false;
      if (pinned_count_ > 0) pinned_count_--;
      slots_[best].busy = true;
      if (slots_[best].layer >= 0)
        map_.erase(((uint64_t)(uint32_t)slots_[best].layer << 32) | (uint32_t)slots_[best].expert);
      slots_[best].layer = -1; slots_[best].expert = -1; slots_[best].heat = 0;
      st_.evictions++;
      slot = best;
    }
  }
  if (slot < 0) {
    int busy = 0, empty = 0, occ = 0, pin = 0;
    for (const Slot& s : slots_) {
      if (s.busy) busy++; if (s.layer < 0) empty++; else occ++; if (s.pinned) pin++;
    }
    fprintf(stderr, "[slots] pool exhausted: layer=%d expert=%d busy=%d pinned=%d empty=%d "
                    "occupied=%d pending=%zu n_slots=%d\n",
            layer, expert, busy, pin, empty, occ, pending_.size(), n_slots_);
    return -1;
  }
  Slot& s = slots_[slot];
  s.layer = layer; s.expert = expert; s.last_used = ++clock_; s.heat = 1;
  map_[key] = slot;
  used_.push_back(slot);
  start_copy(slot, layer, expert);
  return slot;
}

void SlotMgr::sync_copies() {
  if (pending_.empty()) return;
  Device& g1 = Engine::instance().gpu(1);
  HELIOS_CUDA_CHECK(cudaEventSynchronize(g1.event(1)));
  // Only release slots the caller is not using this step. A slot acquired earlier in the same step
  // must stay reserved until end_step(), otherwise a later acquire (or the retry inside acquire)
  // could evict an expert the kernel is about to read, and find() would then return -1 for it.
  for (int i : pending_) {
    bool in_use = false;
    for (int u : used_) if (u == i) { in_use = true; break; }
    if (!in_use) slots_[i].busy = false;
  }
  pending_.clear();
}

void SlotMgr::begin_step() { used_.clear(); }

void SlotMgr::end_step() {
  for (int i : used_) if (slots_[i].layer >= 0) slots_[i].busy = false;
  used_.clear();
  since_rank_++;
  double now = std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
  if (now - last_rank_ >= 5.0) { last_rank_ = now; recompute_hot(); }
  if (now - last_save_ >= 60.0 && !census_path_.empty()) { last_save_ = now; save_census(); }
}

void SlotMgr::print_stats() {
  double lk = (double)st_.lookups;
  printf("[slots] lookups=%llu resident=%.1f%% streams=%llu evictions=%llu pinned=%d h2d=%.2fGB\n",
         (unsigned long long)st_.lookups, lk > 0 ? 100.0 * st_.resident_hits / lk : 0.0,
         (unsigned long long)st_.misses, (unsigned long long)st_.evictions, pinned_count_,
         st_.h2d_bytes / 1073741824.0);
  if (!hot_.empty()) printf("[slots] census hot set=%d of %zu entries (pin budget %d)\n",
                            hot_count_, hot_.size(), max_pinned_);
  if (save_census()) printf("[slots] census saved to %s\n", census_path_.c_str());
}

}  // namespace helios