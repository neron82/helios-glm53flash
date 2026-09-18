#include "core/safetensors.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include "../third_party/json.hpp"
#include <set>

namespace helios {
using nlohmann::json;

static Dtype dtype_of(const std::string& s) {
  if (s == "F16") return Dtype::F16;
  if (s == "BF16") return Dtype::BF16;
  if (s == "F32") return Dtype::F32;
  if (s == "F64") return Dtype::F64;
  if (s == "I32") return Dtype::I32;
  if (s == "I16") return Dtype::I16;
  if (s == "I8") return Dtype::I8;
  if (s == "U8") return Dtype::U8;
  if (s == "BOOL") return Dtype::BOOL;
  return Dtype::UNKNOWN;
}
static uint64_t dtype_size(Dtype d) {
  switch (d) {
    case Dtype::F64: return 8;
    case Dtype::F32: case Dtype::I32: return 4;
    case Dtype::F16: case Dtype::BF16: case Dtype::I16: return 2;
    case Dtype::I8: case Dtype::U8: case Dtype::BOOL: return 1;
    default: return 0;
  }
}

void ShardSet::load_dir(const std::filesystem::path& dir) {
  // Authoritative path: model.safetensors.index.json weight_map -> exact file list.
  std::vector<std::string> ordered_paths;
  std::ifstream idx_f(dir / "model.safetensors.index.json");
  if (idx_f) {
    json ix;
    try { idx_f >> ix; } catch (...) { ix = json(); }
    if (ix.contains("weight_map")) {
      std::set<std::string> files;
      for (auto& [k, v] : ix["weight_map"].items()) files.insert(v.get<std::string>());
      ordered_paths.assign(files.begin(), files.end());
      for (auto& p : ordered_paths) p = (dir / p).string();
    }
  }
  if (ordered_paths.empty()) {
    // Fallback: scan dir (non-recursive) for *.safetensors, sorted by name.
    std::set<std::string> files;
    for (auto& de : std::filesystem::directory_iterator(dir)) {
      if (de.is_regular_file() && de.path().extension() == ".safetensors")
        files.insert(de.path().string());
    }
    ordered_paths.assign(files.begin(), files.end());
  }
  for (auto& cp : ordered_paths) {
    std::ifstream f(cp, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open shard " + cp);
    uint64_t hlen = 0;
    f.read((char*)&hlen, 8);
    if (hlen < 2 || hlen > 512u * 1024u * 1024u) throw std::runtime_error("bad header in " + cp);
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    if ((uint64_t)f.gcount() != hlen) throw std::runtime_error("truncated header in " + cp);
    json j;
    try { j = json::parse(hdr); } catch (...) { throw std::runtime_error("bad json header in " + cp); }
    if (!j.is_object()) throw std::runtime_error("bad header json in " + cp);
    int my_shard = (int)paths_.size();
    paths_.push_back(cp);
    data_starts_.push_back(8 + hlen);
    for (auto& [k, v] : j.items()) {
      if (k == "__metadata__") continue;
      TensorInfo ti;
      ti.name = k;
      ti.dtype = dtype_of(v.at("dtype").get<std::string>());
      ti.shape = v.at("shape").get<std::vector<int64_t>>();
      auto offs = v.at("data_offsets").get<std::vector<int64_t>>();
      ti.offset = (uint64_t)offs[0];
      ti.bytes = (uint64_t)(offs[1] - offs[0]);
      ti.elems = ti.bytes / dtype_size(ti.dtype);
      ti.shard = my_shard;
      tensors_.push_back(std::move(ti));
    }
  }

  std::sort(tensors_.begin(), tensors_.end(), [](const TensorInfo& a, const TensorInfo& b) { return a.name < b.name; });
  ptrs_.reserve(tensors_.size());
  for (auto& t : tensors_) { ptrs_.push_back(&t); total_ += t.bytes; }
  std::sort(ptrs_.begin(), ptrs_.end(), [](const TensorInfo* a, const TensorInfo* b) { return a->name < b->name; });
}

const TensorInfo* ShardSet::find(std::string_view name) const {
  auto it = std::lower_bound(ptrs_.begin(), ptrs_.end(), name,
    [](const TensorInfo* t, std::string_view n) { return t->name < n; });
  if (it != ptrs_.end() && (*it)->name == name) return *it;
  return nullptr;
}

int ShardSet::shard_fd(int i) const {
  if (fds_.size() != paths_.size()) fds_.assign(paths_.size(), -1);
  if (fds_[i] < 0) {
    int fd = ::open(paths_[i].c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open shard " + paths_[i]);
    fds_[i] = fd;
  }
  return fds_[i];
}

void ShardSet::read(const TensorInfo& t, void* dst) const {
  int fd = shard_fd(t.shard);
  uint64_t pos = data_start(t.shard) + t.offset;
  size_t done = 0;
  while (done < t.bytes) {
    ssize_t n = ::pread(fd, (char*)dst + done, t.bytes - done, pos + done);
    if (n <= 0) throw std::runtime_error("pread failed for " + t.name);
    done += (size_t)n;
  }
}

}  // namespace helios