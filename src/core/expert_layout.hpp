#pragma once
#include "core/safetensors.hpp"
#include <array>

namespace helios {

// RAM slabs are compact per layer. GPU slots use the largest trunk slab, so a
// smaller expert never costs extra DMA or RAM merely because another layer is wider.
struct ExpertLayout {
  std::array<size_t, 12> off{}, piece{};
  std::array<int, 3> bits{};
  size_t stride = 0;
};

ExpertLayout expert_layout(const ShardSet& shards, int layer, int experts,
                           int hidden, int intermediate);

}  // namespace helios
