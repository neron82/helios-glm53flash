#include "core/vision.hpp"
#include "core/safetensors.hpp"
#include "json.hpp"
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <spawn.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
namespace helios {
bool supports_vision_input(const ShardSet& shards) {
  // Both checkpoints declare vision_config, but only the dense tower is supported.
  // Inspect headers only; do not copy the tower to RAM or reserve GPU memory here.
  if (!shards.find("model.visual.patch_embed.proj.weight") ||
      !shards.find("model.visual.merger.down_proj.weight")) return false;
  for (const auto* tensor : shards.tensors()) {
    if (!tensor->name.starts_with("model.visual.")) continue;
    if (tensor->name.ends_with(".trellis") || tensor->name.ends_with(".suh") ||
        tensor->name.ends_with(".svh") || tensor->name.ends_with(".mul1")) return false;
    if (tensor->dtype != Dtype::BF16 && tensor->dtype != Dtype::F16 &&
        tensor->dtype != Dtype::F32) return false;
  }
  return true;
}
namespace {
struct Temporary {
  std::string directory;
  Temporary() {
    char pattern[] = "/tmp/helios-vision-XXXXXX";
    char* made = mkdtemp(pattern);
    if (!made) throw std::runtime_error("cannot create vision temporary directory");
    directory = made;
  }
  ~Temporary() { std::error_code ec; std::filesystem::remove_all(directory, ec); }
};

bool run_encoder(const std::string& image, const std::string& model, int device,
                 const Temporary& tmp, ImageEmbedding& result, std::string& error) {
  const char* configured = getenv("HELIOS_VISION_PYTHON");
  std::string python = configured && *configured ? configured : "python3";
  const std::string output = tmp.directory + "/embedding.f16";
  const std::string log = tmp.directory + "/encoder.log";
  std::vector<std::string> args = {python, HELIOS_VISION_SCRIPT, "--model", model,
                                  "--image", image, "--output", output,
                                  "--device", std::to_string(device)};
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
  pid_t pid = 0;
  int launch = posix_spawnp(&pid, python.c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (launch) { error = "cannot launch vision Python interpreter"; return false; }
  int status = 0;
  pid_t waited;
  do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
  if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ifstream input(log);
    std::string detail((std::istreambuf_iterator<char>(input)), {});
    if (detail.size() > 2000) detail = detail.substr(detail.size() - 2000);
    error = "vision encoder failed: " + detail;
    return false;
  }
  nlohmann::json meta;
  std::ifstream(output + ".json") >> meta;
  result.rows = meta.at("rows").get<int>();
  result.hidden = meta.at("hidden").get<int>();
  if (result.rows <= 0 || result.rows > 8000 || result.hidden != 4096)
    throw std::runtime_error("vision encoder returned invalid embedding dimensions");
  const size_t bytes = (size_t)result.rows * result.hidden * 2;
  if (std::filesystem::file_size(output) != bytes)
    throw std::runtime_error("vision encoder returned incomplete embeddings");
  result.values.resize(bytes / 2);
  std::ifstream input(output, std::ios::binary);
  input.read((char*)result.values.data(), bytes);
  if (!input) throw std::runtime_error("cannot read vision embeddings");
  return true;
}
}  // namespace

bool encode_image_file(const std::string& path, const std::string& model, int device,
                       ImageEmbedding& result, std::string& error) {
  try { Temporary tmp; return run_encoder(path, model, device, tmp, result, error); }
  catch (const std::exception& e) { error = e.what(); return false; }
}

bool encode_image_url(const std::string& url, const std::string& model, int device,
                      ImageEmbedding& result, std::string& error) {
  try {
    const size_t comma = url.find(',');
    if (!url.starts_with("data:image/") || comma == std::string::npos ||
        url.substr(0, comma).find(";base64") == std::string::npos) {
      error = "image_url must be a base64 data:image/... URL"; return false;
    }
    if (url.size() - comma > 24u * 1024 * 1024) {
      error = "encoded image exceeds 24 MiB"; return false;
    }
    std::string decoded;
    uint32_t bits = 0; int count = 0; bool padding = false;
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = comma + 1; i < url.size(); ++i) {
      char c = url[i];
      if (c == '=') { padding = true; continue; }
      size_t value = alphabet.find(c);
      if (padding || value == std::string::npos) { error = "invalid image base64"; return false; }
      bits = (bits << 6) | (uint32_t)value; count += 6;
      if (count >= 8) { count -= 8; decoded.push_back((char)(bits >> count)); }
    }
    if (decoded.empty()) { error = "empty image"; return false; }
    Temporary tmp;
    std::string image = tmp.directory + "/image";
    { std::ofstream input(image, std::ios::binary); input.write(decoded.data(), decoded.size()); }
    return run_encoder(image, model, device, tmp, result, error);
  } catch (const std::exception& e) { error = e.what(); return false; }
}
}  // namespace helios
