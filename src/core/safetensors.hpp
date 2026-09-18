#pragma once
// Safetensors container parser for EXL3 shards. Header = 8-byte LE length + JSON tensor map.
// Each EXL3 quantized matrix W[out,in] is stored as four tensors:
//   <name>.suh     F16 [in]   per-input-column scales
//   <name>.svh     F16 [out]  per-output-row scales ("out_scales always")
//   <name>.mul1    I32 []     scalar codebook multiplier (bit-packed)
//   <name>.trellis I16 [d0,d1,S] trellis symbols; d0*d1*S*16 = out*in*bits
#include <cstdint>
#include <string>
#include <vector>
#include <span>
#include <filesystem>

namespace helios {

enum class Dtype : uint8_t { F16, BF16, F32, F64, I32, I16, I8, U8, BOOL, UNKNOWN };

struct TensorInfo {
  std::string name;
  Dtype dtype = Dtype::UNKNOWN;
  uint64_t elems = 0;
  uint64_t bytes = 0;
  int shard = -1;
  uint64_t offset = 0;   // byte offset inside shard data region
  std::vector<int64_t> shape;
};

class ShardSet {
public:
  // Scans dir for *.safetensors (and .incomplete during download), parses headers only.
  void load_dir(const std::filesystem::path& dir);

  const TensorInfo* find(std::string_view name) const;
  std::span<const TensorInfo* const> tensors() const { return ptrs_; }
  uint64_t total_bytes() const { return total_; }
  int shard_count() const { return (int)paths_.size(); }
  const std::string& shard_path(int i) const { return paths_[i]; }
  // Opens shard fd lazily; returns fd (mmap done by caller).
  int shard_fd(int i) const;
  // Byte offset where the data region begins for shard i (8 + header_len).
  uint64_t data_start(int i) const { return data_starts_[i]; }
  // pread one tensor's bytes into dst (thread-safe: fds opened lazily with mutex).
  void read(const TensorInfo& t, void* dst) const;

private:
  std::vector<std::string> paths_;
  std::vector<TensorInfo> tensors_;
  std::vector<const TensorInfo*> ptrs_;  // sorted by name for find()
  uint64_t total_ = 0;
  mutable std::vector<int> fds_;
  std::vector<uint64_t> data_starts_;
};

}  // namespace helios