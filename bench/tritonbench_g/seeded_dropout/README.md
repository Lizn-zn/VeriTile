# seeded_dropout

- Source file: `seeded_dropout.py` (upstream `data/TritonBench_G_v1/seeded_dropout.py`)
- Corpus: TritonBench-G v1
- Size: 60 lines, 1 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported. Every other `tl.*` form it uses is in the DSL
  surface, but it needs: `tl.rand`.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md) for the
  ranked unlock levers.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
