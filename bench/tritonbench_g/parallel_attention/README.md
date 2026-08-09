# parallel_attention

- Source file: `parallel_attention.py` (upstream `data/TritonBench_G_v1/parallel_attention.py`)
- Corpus: TritonBench-G v1
- Size: 481 lines, 4 `@triton.jit` kernel(s)
- Status: READY — not imported yet, but every `tl.*` form it uses is already in the
  DSL surface, so the port is a scheduling question, not a capability one.
  Import the `.py` **with** its `.lean` port in one change:
  `bench/audit_tritonbench_g.sh` enforces `py_count == lean_count`.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
