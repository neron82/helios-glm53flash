#pragma once
// Cross-request prefix cache policy.
//
// Kept as pure logic, separate from the CUDA mechanism that executes it, because this is where a
// prefix cache actually goes wrong: a resume position past the shared prefix serves tokens the
// caches never saw, and a resume past a snapshot the prompt does not match restores a KDA state that
// belongs to different tokens. Both are usually invisible in the generated text, so the decision is
// tested exhaustively (test_prefix) instead of being inferred from output.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace helios {

// Tokens per indexer pool; a resume must be a multiple of this (see below).
inline constexpr int POOL_TOKENS = 4;

inline int prefix_checkpoint_before(int end) {
  return end > 0 ? (end - 1) / POOL_TOKENS * POOL_TOKENS : 0;
}

// A branch rewrites cache rows after resume. States from that discarded suffix
// must never be restored against the new resident history.
inline int prefix_discard_suffix(std::vector<int>& snap_pos, int resume) {
  int latest = 0;
  for (int& pos : snap_pos) {
    if (pos > resume) pos = -1;
    latest = std::max(latest, pos);
  }
  return latest;
}

struct PrefixPlan {
  int common = 0;        // tokens the prompt shares with the resident history
  int resume = 0;        // position to resume at (0 = recompute the whole prompt)
  int slot = -1;         // snapshot slot to restore, -1 for none
  bool extend = false;   // the resident state already covers the resume point exactly
};

// hist: the token sequence the caches currently hold; pos: how many of those rows are computed.
// snap_pos: captured position per snapshot slot (-1 = empty).
inline PrefixPlan prefix_plan(const std::vector<int32_t>& hist, int pos,
                              const std::vector<int>& prompt, const std::vector<int>& snap_pos,
                              int reuse_limit = std::numeric_limits<int>::max()) {
  PrefixPlan pl;
  const int ps = (int)prompt.size();
  const int lim = std::max(0, std::min({(int)hist.size(), pos, reuse_limit}));
  int L = 0;
  while (L < lim && L < ps && prompt[L] == hist[L]) L++;
  pl.common = L;
  // The caller samples from the prompt's last token, so the last token is always re-run - which also
  // guarantees at least one row through the model even when the whole prompt is already resident.
  if (L >= ps) L = ps - 1;
  if (L < 0) L = 0;
  if (L == pos && L == (int)hist.size()) {          // nothing to restore: the planes already hold it
    pl.extend = true;
    pl.resume = L;
    return pl;
  }
  // Otherwise the KDA state has to be wound back, and the newest snapshot at or below the divergence
  // point is the cheapest: the rows between it and the prompt are recomputed, bounded by the
  // snapshot interval.
  int best = -1;
  for (int i = 0; i < (int)snap_pos.size(); i++)
    if (snap_pos[i] >= 0 && snap_pos[i] <= L && snap_pos[i] % POOL_TOKENS == 0 &&
        (best < 0 || snap_pos[i] > snap_pos[best])) best = i;
  if (best >= 0) {
    pl.slot = best;
    pl.resume = snap_pos[best];
  } else {
    pl.slot = -1;
    pl.resume = 0;                                   // no usable snapshot: recompute from the start
  }
  // Snapshot positions must already be pool-aligned. Rounding a position down
  // without rewinding the captured recurrent state would restore the wrong prefix.
  return pl;
}

}  // namespace helios
