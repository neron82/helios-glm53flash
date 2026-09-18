#pragma once
// Sampler: temperature / top-k / top-p / min-p / repetition penalty over fp16 logits.
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_fp16.h>

namespace helios {

struct GenParams {
  int max_tokens = 256;
  float temperature = 0.7f;
  int top_k = 40;
  float top_p = 0.95f;
  float min_p = 0.0f;
  float rep_penalty = 1.0f;
  int rep_window = 256;
  uint64_t seed = 0;
  bool greedy = false;
  std::vector<std::string> stop;   // stop strings
  std::string grammar;             // unused (reserved)
};

class Sampler {
public:
  void reset(uint64_t seed);
  // logits: fp16 host array [vocab]; recent: token ids for repetition penalty
  int sample(const half* logits, int vocab, const GenParams& p, const std::vector<int>& recent);

private:
  uint64_t state_ = 0x853c49e6748fea9bull;
  float next_f();
  std::vector<float> buf_;
  std::vector<int> idx_;
};

}  // namespace helios