# fp4_to_bf16

- Source file: `fp4_to_bf16.py` (upstream `data/TritonBench_G_v1/fp4_to_bf16.py`)
- Corpus: TritonBench-G v1
- Size: 214 lines, 2 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported. Every other `tl.*` form it uses is in the DSL
  surface, but it needs: `tl.interleave`.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md) for the
  ranked unlock levers.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
