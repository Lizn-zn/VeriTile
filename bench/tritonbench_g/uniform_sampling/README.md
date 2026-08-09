# uniform_sampling

- Source file: `uniform_sampling.py` (upstream `data/TritonBench_G_v1/uniform_sampling.py`)
- Corpus: TritonBench-G v1
- Size: 226 lines, 2 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported. Every other `tl.*` form it uses is in the DSL
  surface, but it needs: `tl.philox`, `tl.static_assert`, `tl.uint_to_uniform_float`.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md) for the
  ranked unlock levers.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
