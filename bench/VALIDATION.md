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
`results/prepush-2026-10-04.json`; the final checkout sweep will be recorded
separately, with its build revision and raw logs.

GPU text benchmarks exclude startup and vision preprocessing. They validate the
requested grid, not the larger quant's full context window or all visual tasks.
