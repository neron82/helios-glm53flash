#!/usr/bin/env python3
"""Add opt-in row-byte fingerprints around the first recurrence and layer boundaries.
Instrumentation only: no arithmetic or buffers change; used for both numerical arms.
"""
from pathlib import Path
p=Path('src/engine/runner.cpp'); s=p.read_text()
helper = r'''
// Diagnostic only: hashes exact device bytes by row at prefill / first decode.
static void det_plane(const char* name, int layer, int pos, const void* ptr,
                      size_t row_bytes, int rows, cudaStream_t stream) {
  if (!getenv("HELIOS_DET_PLANES") || pos > 4096) return;
  HELIOS_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<unsigned char> bytes(row_bytes * rows);
  HELIOS_CUDA_CHECK(cudaMemcpy(bytes.data(), ptr, bytes.size(), cudaMemcpyDeviceToHost));
  fprintf(stderr, "[det-plane] pos=%d layer=%d name=%s row_bytes=%zu hashes=", pos, layer, name, row_bytes);
  for (int row=0; row<rows; ++row) {
    uint64_t h=14695981039346656037ull;
    for(size_t j=0;j<row_bytes;++j) h=(h ^ bytes[row*row_bytes+j])*1099511628211ull;
    fprintf(stderr, "%s%016llx", row?",":"", (unsigned long long)h);
  }
  fprintf(stderr, "\n");
}
'''
s=s.replace('// ---------------------------------------------------------------- init',helper+'\n// ---------------------------------------------------------------- init',1)
s=s.replace('  // 5) delta rule recurrence', '  det_plane("conv_out", L.index, pos, w0.conv_out_bf16, 24576*2, n, s);\n  det_plane("gate", L.index, pos, w0.kda_g, 8192*4, n, s);\n  det_plane("beta", L.index, pos, w0.beta_bf16, 64*2, n, s);\n  det_plane("rec_before", L.index, pos, c_->kda_rec(kda_ord), 128*4, 64*128, s);\n  // 5) delta rule recurrence',1)
s=s.replace('  if (const char* dd = getenv("HELIOS_DUMP_KDA"))', '  det_plane("rec_output", L.index, pos, w0.kda_rec_bf16, 8192*2, n, s);\n  det_plane("rec_after", L.index, pos, c_->kda_rec(kda_ord), 128*4, 64*128, s);\n  if (const char* dd = getenv("HELIOS_DUMP_KDA"))',1)
s=s.replace('  cudaStream_t s = s0_stream();\n  {\n    // attention site', '  cudaStream_t s = s0_stream();\n  det_plane("layer_input", L.index, pos, streams, 4*4096*4, n, s);\n  {\n    // attention site',1)
s=s.replace('    DBGSYNC(s, "attn layer");', '    det_plane("attn_output", L.index, pos, w0.attn_out16, 4096*2, n, s);\n    DBGSYNC(s, "attn layer");',1)
s=s.replace('    DBGSYNC(s, "ffn layer");', '    det_plane("ffn_output", L.index, pos, w0.ffn_out16, 4096*2, n, s);\n    DBGSYNC(s, "ffn layer");',1)
p.write_text(s)
