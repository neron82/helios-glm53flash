# Mixed-quant validation — 2026-10-04

Both checkpoints pass layout validation across every expert in all 43 routed
layers, full RAM and GPU loading, 256 sampled RAM pieces and 507 device tensor
comparisons (zero mismatches). CPU regressions cover malformed layouts, different
projection widths, prefix policy, UTF-8 and the chat protocol. CUDA reconstruction,
GEMM, aux and attention parity pass. Both tokenizer suites pass 28/28 and each
checkpoint's local Jinja chat template matches all four rendered cases.

The larger checkpoint passes arithmetic, translation, SSE, red/blue/red image
changes, non-square images, ordered two-image input, OCR of HELLO 42, and text
following images. The original checkpoint passes arithmetic, translation and SSE;
its quantized vision tower is explicitly outside the optional BF16 frontend's
scope. See the recorded generation and OCR responses in `results/`.

The initial eight-cell sweep passed with exact 4096/8192 prompt counts and
256/512 output counts, no prefix reuse, no persisted census, and no degraded MoE
resolution. Prefill remains within 3.3% of the original rate; decode is
22.7–25.8% slower, consistent with the larger slabs and smaller resident set.
This satisfies the publication condition. Initial measurements are retained in
`results/prepush-2026-10-04.json`; the final checkout sweep is recorded
separately in `results/final-2026-10-04.json`, with its build revision and raw logs.

GPU text benchmarks exclude startup and vision preprocessing. They validate the
requested grid, not the larger quant's full context window or all visual tasks.

## Final fresh-checkout sweep

Final sweep on the fresh checkout, build `7703158`: **three repeats per cell,
medians below (24 successful runs)**. Each run loads a fresh model process with
identical exact token-ID prompts, `--cap 16384 --chunk 8192 --prefix-snap-mb 0`,
greedy sampling, and no saved expert census. `--ignore-eos` forces the requested
output count for timing; normal EOS-respecting text and image requests were
validated separately. Load/pin time and vision preprocessing are excluded.
Decode counts 255 or 511 forward steps for 256 or 512 emitted tokens because
prefill already supplies the first token's logits.

| Prefill | Generated | Original prefill tok/s | Original decode tok/s | Larger prefill tok/s | Larger decode tok/s |
|---|---|---|---|---|---|
| 4096 | 256 | 313.0 | 16.18 | 302.5 | 12.31 |
| 4096 | 512 | 312.8 | 16.16 | 302.6 | 12.42 |
| 8192 | 256 | 348.1 | 16.02 | 344.6 | 12.45 |
| 8192 | 512 | 348.1 | 15.95 | 344.4 | 12.33 |

Across this grid the larger quant's prefill rate is within **3.4%** of the
original, while decode is **22.3–23.9% slower**. Its compact
arena contains 25.4% more expert data and the same GPU pool fits 2034 slots rather
than 3051. All runs produced the exact requested counts with no degraded MoE
resolution or tensor mismatches. These medians confirm the initial acceptance
sweep's comparable prefill and expected decode slowdown.

The fresh build passed all six CTest suites, aux/attention parity, both tokenizer
suites (28/28) and both local Jinja chat-template comparisons (4/4). Live checks
passed arithmetic, translation, text SSE, red/blue/red image changes, non-square
images, ordered multiple images, OCR, text after images and streamed image
responses. Unsupported URLs and malformed base64 return HTTP 400. Both quant
loaders report zero mismatches in 256 RAM samples and 507 device tensors.

[Final raw measurements](results/final-2026-10-04.json),
[full logs and exact prompts](results/final-2026-10-04-logs.tar.gz),
[initial acceptance sweep](results/prepush-2026-10-04.json),
[fresh text/image responses](results/fresh-generation-2026-10-04.json), and
[image streaming/error checks](results/fresh-image-stream-errors-2026-10-04.json).
The original logs also remain at `/tmp/helios-quant-final/` on the reference host.

```bash
~/shared-venv-gpu/bin/python bench/quant_sweep.py --repeats 3 --output /tmp/helios-quant-sweep
```
