#include "engine/sampler.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace helios {

void Sampler::reset(uint64_t seed) {
  state_ = seed ? seed : 0x853c49e6748fea9bull;
}

// splitmix64-ish -> uniform [0,1)
float Sampler::next_f() {
  state_ += 0x9E3779B97F4A7C15ull;
  uint64_t z = state_;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z = z ^ (z >> 31);
  return (float)((z >> 40) * (1.0 / 16777216.0));
}

int Sampler::sample(const half* logits, int V, const GenParams& p, const std::vector<int>& recent) {
  if (p.greedy || p.temperature <= 0.f) {
    int best = 0; float bv = -INFINITY;
    for (int i = 0; i < V; i++) {
      float v = __half2float(logits[i]);
      if (v > bv) { bv = v; best = i; }
    }
    return best;
  }
  buf_.resize(V);
  float inv_t = 1.0f / p.temperature;
  for (int i = 0; i < V; i++) buf_[i] = __half2float(logits[i]) * inv_t;

  // repetition penalty over a recent window
  if (p.rep_penalty > 1.0001f && !recent.empty()) {
    int n = (int)recent.size();
    int from = std::max(0, n - p.rep_window);
    for (int k = from; k < n; k++) {
      int id = recent[k];
      if (id < 0 || id >= V) continue;
      buf_[id] = buf_[id] > 0.f ? buf_[id] / p.rep_penalty : buf_[id] * p.rep_penalty;
    }
  }

  // top-k: keep the k largest values (softmax is monotone in logits)
  int k = p.top_k > 0 ? std::min(p.top_k, V) : V;
  idx_.resize(V);
  for (int i = 0; i < V; i++) idx_[i] = i;
  if (k < V) {
    std::partial_sort(idx_.begin(), idx_.begin() + k, idx_.end(),
                      [&](int a, int b) { return buf_[a] > buf_[b]; });
    idx_.resize(k);
  }

  // softmax over the candidate set
  float mx = -INFINITY;
  for (int i : idx_) mx = std::max(mx, buf_[i]);
  double sum = 0;
  std::vector<float> probs(idx_.size());
  for (size_t i = 0; i < idx_.size(); i++) {
    probs[i] = std::exp(buf_[idx_[i]] - mx);
    sum += probs[i];
  }
  // sort candidates by probability descending for top-p
  std::vector<size_t> order(idx_.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });

  float maxp = (float)(probs[order[0]] / sum);
  double acc = 0;
  size_t keep = 0;
  for (size_t i = 0; i < order.size(); i++) {
    float pi = (float)(probs[order[i]] / sum);
    if (p.min_p > 0.f && pi < p.min_p * maxp && i > 0) break;
    if (p.top_p < 1.0f && acc >= p.top_p && i > 0) break;
    acc += pi;
    keep++;
  }
  double r = next_f() * acc;
  double c = 0;
  for (size_t i = 0; i < keep; i++) {
    c += probs[order[i]] / sum;
    if (r <= c) return idx_[order[i]];
  }
  return idx_[order[keep - 1]];
}

}  // namespace helios