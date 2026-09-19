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
#include <vector>

namespace helios {

// Tokens per indexer pool; a resume must be a multiple of this (see below).
inline constexpr int POOL_TOKENS = 4;

struct PrefixPlan {
  int common = 0;        // tokens the prompt shares with the resident history
  int resume = 0;        // position to resume at (0 = recompute the whole prompt)
  int slot = -1;         // snapshot slot to restore, -1 for none
  bool extend = false;   // the resident state already covers the resume point exactly
};

// hist: the token sequence the caches currently hold; pos: how many of those rows are computed.
// snap_pos: captured position per snapshot slot (-1 = empty).
inline PrefixPlan prefix_plan(const std::vector<int32_t>& hist, int pos,
                              const std::vector<int>& prompt, const std::vector<int>& snap_pos) {
  PrefixPlan pl;
  const int ps = (int)prompt.size();
  const int lim = std::min<int>((int)hist.size(), pos);
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
    if (snap_pos[i] >= 0 && snap_pos[i] <= L && (best < 0 || snap_pos[i] > snap_pos[best])) best = i;
  if (best >= 0) {
    pl.slot = best;
    pl.resume = snap_pos[best];
  } else {
    pl.slot = -1;
    pl.resume = 0;                                   // no usable snapshot: recompute from the start
  }
  // Resumes must land on a pool boundary: the indexer's raw rows are a ring, so a group must never
  // straddle a position jump (within a chunk, and across decode steps, they are contiguous). Rounding
  // down costs at most three tokens of recompute.
  pl.resume -= pl.resume % POOL_TOKENS;
  return pl;
}

}  // namespace helios
