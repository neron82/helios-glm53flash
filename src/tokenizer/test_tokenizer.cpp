// Compare helios tokenizer against HF transformers ground truth (test/tokenizer_vectors.json).
#include "tokenizer/tokenizer.hpp"
#include "json.hpp"
#include <cstdio>
#include <fstream>

using namespace helios;
using nlohmann::json;

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string dir = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/models/glm53flash";
  std::string vec = argc > 2 ? argv[2] : "test/tokenizer_vectors.json";
  Tokenizer tk;
  if (!tk.load(dir)) return 1;
  std::ifstream f(vec);
  if (!f) { printf("no vectors at %s\n", vec.c_str()); return 1; }
  json j = json::parse(f);
  int fails = 0, total = 0;
  const auto& texts = j.at("texts");
  const auto& ids = j.at("ids");
  for (size_t i = 0; i < texts.size(); i++) {
    std::string t = texts[i].get<std::string>();
    std::vector<int> want = ids[i].get<std::vector<int>>();
    std::vector<int> got = tk.encode(t);
    total++;
    if (got != want) {
      fails++;
      printf("FAIL [%zu] %s\n  want:", i, json(t).dump().c_str());
      for (int x : want) printf(" %d", x);
      printf("\n  got :");
      for (int x : got) printf(" %d", x);
      printf("\n  dec : %s\n", tk.decode(got).c_str());
    }
  }
  if (j.contains("chat_prompt")) {
    std::string want = j.at("chat_prompt").get<std::string>();
    std::vector<ChatMsg> msgs = {{"system", "You are helpful."}, {"user", "Hello world!"}};
    std::string got = tk.apply_chat_template(msgs, true);
    total++;
    if (got != want) { fails++; printf("FAIL chat template\n  want: %s\n  got : %s\n", want.c_str(), got.c_str()); }
    std::vector<int> want_ids = j.at("chat_ids").get<std::vector<int>>();
    std::vector<int> got_ids = tk.encode(got);
    total++;
    if (got_ids != want_ids) {
      fails++;
      printf("FAIL chat ids\n  want:");
      for (int x : want_ids) printf(" %d", x);
      printf("\n  got :");
      for (int x : got_ids) printf(" %d", x);
      printf("\n");
    }
  }
  printf("TOKENIZER: %d/%d pass%s\n", total - fails, total, fails ? " — FAIL" : " (ALL PASS)");
  return fails ? 1 : 0;
}