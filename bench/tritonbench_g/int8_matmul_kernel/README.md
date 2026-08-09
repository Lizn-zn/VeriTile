# int8_matmul_kernel

- Source file: `int8_matmul_kernel.py` (upstream `data/TritonBench_G_v1/int8_matmul_kernel.py`)
- Corpus: TritonBench-G v1
- Size: 271 lines, 1 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported. Every other `tl.*` form it uses is in the DSL
  surface, but it needs: `tl.static_assert`.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md) for the
  ranked unlock levers.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
