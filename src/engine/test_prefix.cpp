// Unit test for the cross-request prefix-cache policy (Runner::prefix_plan).
//
// The policy is where a prefix cache actually goes wrong: a resume position that is too long serves
// the request tokens the caches never saw, and one that is too short silently recomputes work (or,
// worse, resumes from a snapshot the caller's tokens do not match). Both are invisible in the output
// most of the time, so the decision is tested exhaustively here rather than inferred from text.
//
// No GPU involved: prefix_plan is pure.
#include "engine/prefix.hpp"
#include <cstdio>
#include <cstring>

using helios::prefix_plan;
using helios::PrefixPlan;

static int failures = 0, checks = 0;

static void check(const char* what, bool ok) {
  checks++;
  if (!ok) { failures++; printf("  FAIL %s\n", what); }
}

// Convenience: resident history of `n` tokens counting down (n-1, n-2, ...), and a prompt built from
// an explicit list.
static std::vector<int32_t> hist_n(int n) {
  std::vector<int32_t> h(n);
  for (int i = 0; i < n; i++) h[i] = i;
  return h;
}
static std::vector<int> v(std::initializer_list<int> l) { return std::vector<int>(l); }

int main() {
  // A snapshot ring holding 8192/16384/24576 (a typical mid-prefill state) plus empty slots.
  std::vector<int> snaps = {8192, 16384, 24576, -1, -1};

  printf("[1] empty history: nothing to reuse, state is already initial\n");
  {
    auto pl = prefix_plan({}, 0, v({5, 6, 7}), snaps);
    check("common=0", pl.common == 0);
    check("resume 0", pl.resume == 0);
    check("no slot", pl.slot == -1);
  }

  printf("[2] prompt extends the resident history (multi-turn chat)\n");
  {
    auto h = hist_n(100);
    std::vector<int> p(h.begin(), h.end());
    for (int i = 0; i < 12; i++) p.push_back(1000 + i);      // one more user turn
    auto pl = prefix_plan(h, 100, p, snaps);
    check("common=100", pl.common == 100);
    check("extend", pl.extend);
    check("resume=100", pl.resume == 100);
    check("no restore needed", pl.slot == -1);
  }

  printf("[3] prompt identical to the resident history (regenerate)\n");
  {
    auto h = hist_n(30000);
    std::vector<int> p(h.begin(), h.end());
    auto pl = prefix_plan(h, 30000, p, snaps);
    check("common=30000", pl.common == 30000);
    check("not extend (last token re-run)", !pl.extend);
    check("resume at newest snapshot <= 29999", pl.slot == 2 && pl.resume == 24576);
  }

  printf("[4] divergence with a snapshot at or below it\n");
  {
    auto h = hist_n(30000);
    std::vector<int> p(h.begin(), h.end());
    p[20000] = 999999;                       // edit deep in the history
    auto pl = prefix_plan(h, 30000, p, snaps);
    check("common=20000", pl.common == 20000);
    check("snapshot at 16384", pl.slot == 1 && pl.resume == 16384);
  }

  printf("[5] divergence below every snapshot\n");
  {
    auto h = hist_n(30000);
    std::vector<int> p = h;
    p[100] = 999999;
    auto pl = prefix_plan(h, 30000, p, snaps);
    check("common=100", pl.common == 100);
    check("restart", pl.slot == -1 && pl.resume == 0 && !pl.extend);
  }

  printf("[6] no snapshots at all: only extension reuse remains\n");
  {
    auto h = hist_n(5000);
    std::vector<int> p(h.begin(), h.end());
    p[4000] = 7;
    auto pl = prefix_plan(h, 5000, p, std::vector<int>());
    check("divergence restarts", pl.resume == 0);
    std::vector<int> p2(h.begin(), h.end());
    p2.push_back(4242);
    auto pl2 = prefix_plan(h, 5000, p2, std::vector<int>());
    check("extension still reuses", pl2.extend && pl2.resume == 5000);
  }

  printf("[7] snapshot exactly at the divergence point\n");
  {
    auto h = hist_n(30000);
    std::vector<int> p(h.begin(), h.end());
    p[24576] = 42;
    auto pl = prefix_plan(h, 30000, p, snaps);
    check("common=24576", pl.common == 24576);
    check("resume exactly at 24576", pl.resume == 24576 && pl.slot == 2);
  }

  printf("[8] new prompt is a strict prefix of the history\n");
  {
    auto h = hist_n(30000);
    std::vector<int> p(h.begin(), h.begin() + 1000);
    auto pl = prefix_plan(h, 30000, p, snaps);
    check("common=1000", pl.common == 1000);
    check("last token re-run, history cut to 999", pl.resume == 999 || pl.resume == 0);
    check("never resumes past the prompt", pl.resume <= 999);
  }

  printf("[9] unrelated prompt\n");
  {
    auto h = hist_n(30000);
    auto pl = prefix_plan(h, 30000, v({7, 7, 7}), snaps);
    check("common=0", pl.common == 0);
    check("restart", pl.resume == 0 && pl.slot == -1);
  }

  printf("[10] resident history never exceeds the computed position\n");
  {
    // hist_ can be longer than pos_ only transiently; the policy must bound the match by pos_.
    auto h = hist_n(10000);
    std::vector<int> p(h.begin(), h.end());
    for (int pos : {0, 1, 4096, 8192, 9999, 10000}) {
      auto pl = prefix_plan(h, pos, p, snaps);
      check("resume <= pos", pl.resume <= pos);
      check("common <= pos", pl.common <= pos);
      check("resume < prompt size", pl.resume < (int)p.size());
    }
  }

  printf("[11] a one-token prompt never resumes past it\n");
  {
    auto h = hist_n(5000);
    auto pl = prefix_plan(h, 5000, v({0}), snaps);
    check("resume=0", pl.resume == 0);
  }

  printf("\n%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
  return failures ? 1 : 0;
}
