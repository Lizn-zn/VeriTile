# matmul_dequantize

- Source file: `matmul_dequantize.py` (upstream `data/TritonBench_G_v1/matmul_dequantize.py`)
- Corpus: TritonBench-G v1
- Size: 358 lines, 3 `@triton.jit` kernel(s)
- Status: READY — not imported yet, but every `tl.*` form it uses is already in the
  DSL surface, so the port is a scheduling question, not a capability one.
  Import the `.py` **with** its `.lean` port in one change:
  `bench/audit_tritonbench_g.sh` enforces `py_count == lean_count`.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
