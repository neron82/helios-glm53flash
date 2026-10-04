#pragma once
#include <cstdint>
#include <list>
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
// Token placeholders are equal even when images differ. Bound token reuse by
// the first changed/added/removed image's position, comparing actual embeddings.
int image_prefix_limit(const std::vector<ImageEmbedding>& previous,
                       const std::vector<ImageEmbedding>& current);

// Per-server cache of encoded images. Keys are exact data URLs; the memory budget
// includes keys and embedding values. Callers serialize access with generation.
class ImageEmbeddingCache {
public:
  explicit ImageEmbeddingCache(size_t budget = 128u * 1024 * 1024) : budget_(budget) {}
  bool get(const std::string& url, ImageEmbedding& image);
  void put(const std::string& url, const ImageEmbedding& image);
  size_t bytes() const { return bytes_; }
private:
  struct Entry { std::string url; ImageEmbedding image; size_t bytes; };
  std::list<Entry> entries_;
  size_t budget_, bytes_ = 0;
};
bool encode_image_file(const std::string& path, const std::string& model, int device,
                       ImageEmbedding& result, std::string& error);
bool encode_image_url(const std::string& url, const std::string& model, int device,
                      ImageEmbedding& result, std::string& error);
}  // namespace helios
