#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace helios {
class ShardSet;
// The optional Python encoder accepts dense vision weights, not EXL3 vision groups.
bool supports_vision_input(const ShardSet& shards);
struct ImageEmbedding {
  int start = 0, rows = 0, hidden = 0;
  std::vector<uint16_t> values;  // fp16 bits, [rows, hidden]
};
bool encode_image_file(const std::string& path, const std::string& model, int device,
                       ImageEmbedding& result, std::string& error);
bool encode_image_url(const std::string& url, const std::string& model, int device,
                      ImageEmbedding& result, std::string& error);
}  // namespace helios
