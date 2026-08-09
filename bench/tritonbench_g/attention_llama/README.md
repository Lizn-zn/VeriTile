# attention_llama

- Source file: `attention_llama.py` (upstream `data/TritonBench_G_v1/attention_llama.py`)
- Corpus: TritonBench-G v1
- Size: 174 lines, 1 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported. Every other `tl.*` form it uses is in the DSL
  surface, but it needs: `tl.float8e5`.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md) for the
  ranked unlock levers.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
