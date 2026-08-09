# int_scaled_matmul

- Source file: `int_scaled_matmul.py` (upstream `data/TritonBench_G_v1/int_scaled_matmul.py`)
- Corpus: TritonBench-G v1
- Size: 304 lines, 2 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported. Every other `tl.*` form it uses is in the DSL
  surface, but it needs: `tl.broadcast_to`.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md) for the
  ranked unlock levers.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
