# Third-party notices

This engine is MIT licensed (see [`LICENSE`](LICENSE)). It vendors and ports third-party code, and
those components keep their own licenses and copyright notices — the MIT terms require those
notices to travel with the code, which is what this file collects.

| component | where | license | notice |
|---|---|---|---|
| EXL3 quantization core — a port of [exllamav3](https://github.com/turboderp-org/exllamav3) | `src/cuda/quant/` | MIT, © 2025 Turboderp | [`third_party/LICENSE-exllamav3`](third_party/LICENSE-exllamav3), and the notice at the top of [`src/cuda/quant/README_PORT.md`](src/cuda/quant/README_PORT.md) |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | `third_party/httplib.h` | MIT, © 2017 yhirose | [`third_party/LICENSE-httplib`](third_party/LICENSE-httplib) |
| [nlohmann/json](https://github.com/nlohmann/json) | `third_party/json.hpp` | MIT, © 2013-2025 Niels Lohmann | [`third_party/LICENSE.MIT`](third_party/LICENSE.MIT) |

## The EXL3 port

This one is worth stating plainly, because it is a derivative work rather than a dependency.
`src/cuda/quant/` is a C++20/CUDA port of `exllamav3/exllamav3_ext/`: the **kernel bodies are
preserved byte-exact**, and only the torch-dependent launcher wrappers, autotuning and
graph-capture infrastructure were replaced with raw-pointer C++ APIs and explicit stream
parameters. `src/cuda/quant/README_PORT.md` carries the file-by-file mapping.

Because byte-exact kernel bodies are a substantial portion of the upstream work, turboderp's MIT
notice is reproduced in three places: `third_party/LICENSE-exllamav3`, the header of
`src/cuda/quant/README_PORT.md`, and here.

## Model weights

No model weights are distributed with this repository, and the MIT license does not cover them. The
EXL3 checkpoint this engine loads is a separate download under its own terms; see the model's
repository for those.
