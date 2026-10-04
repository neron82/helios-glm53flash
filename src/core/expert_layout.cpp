#include "core/expert_layout.hpp"
#include <stdexcept>

namespace helios {

ExpertLayout expert_layout(const ShardSet& shards, int layer, int experts,
                           int hidden, int intermediate) {
  ExpertLayout layout;
  static const char* proj[] = {"gate_proj", "up_proj", "down_proj"};
  static const char* suffix[] = {".trellis", ".suh", ".svh", ".mul1"};
  for (int e = 0; e < experts; ++e) {
    for (int p = 0; p < 3; ++p) {
      std::string base = "model.language_model.layers." + std::to_string(layer) +
                         ".mlp.experts." + std::to_string(e) + "." + proj[p];
      const TensorInfo* t[4];
      for (int q = 0; q < 4; ++q) {
        t[q] = shards.find(base + suffix[q]);
        if (!t[q]) throw std::runtime_error("missing expert tensor " + base + suffix[q]);
      }
      const int in = p == 2 ? intermediate : hidden;
      const int out = p == 2 ? hidden : intermediate;
      const auto& shape = t[0]->shape;
      if (shape.size() != 3 || shape[0] != in / 16 || shape[1] != out / 16 ||
          shape[2] % 16 || shape[2] < 32 || shape[2] > 128)
        throw std::runtime_error("unsupported expert trellis shape: " + base);
      const int bits = shape[2] / 16;
      if (bits == 7) throw std::runtime_error("7-bit expert kernels are not built: " + base);
      const size_t bytes[4] = {(size_t)in * out * bits / 8, (size_t)in * 2,
                               (size_t)out * 2, 4};
      const Dtype types[] = {Dtype::I16, Dtype::F16, Dtype::F16, Dtype::I32};
      if (t[1]->shape != std::vector<int64_t>{in} ||
          t[2]->shape != std::vector<int64_t>{out} || !t[3]->shape.empty())
        throw std::runtime_error("invalid expert scales/multiplier shape: " + base);
      for (int q = 0; q < 4; ++q) {
        if (t[q]->bytes != bytes[q] || t[q]->dtype != types[q])
          throw std::runtime_error("invalid expert tensor size/dtype: " + base + suffix[q]);
        if (e == 0) layout.piece[4 * p + q] = bytes[q];
        else if (layout.piece[4 * p + q] != bytes[q])
          throw std::runtime_error("mixed expert bit widths within a layer: " + base);
      }
      // The fused CUDA mul1 codebook has this multiplier compiled in.
      uint32_t mul = 0;
      shards.read(*t[3], &mul);
      if (mul != 0x83DCD12Du)
        throw std::runtime_error("unsupported expert mul1 codebook: " + base);
      layout.bits[p] = bits;
    }
  }
  for (int q = 0; q < 12; ++q) {
    layout.off[q] = layout.stride;
    layout.stride += layout.piece[q];
    if (q % 4 == 3) layout.stride = (layout.stride + 63) & ~(size_t)63;
  }
  return layout;
}

}  // namespace helios
