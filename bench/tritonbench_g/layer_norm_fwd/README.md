# layer_norm_fwd

- Source file: `layer_norm_fwd.py` (upstream `data/TritonBench_G_v1/layer_norm_fwd.py`)
- Corpus: TritonBench-G v1
- Size: 218 lines, 1 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported. Every other `tl.*` form it uses is in the DSL
  surface, but it needs: `tl.rand`.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md) for the
  ranked unlock levers.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
