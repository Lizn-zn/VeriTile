# isfinite_kernel

- Source file: `isfinite_kernel.py` (upstream `data/TritonBench_G_v1/isfinite_kernel.py`)
- Corpus: TritonBench-G v1
- Size: 262 lines, 2 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported, on two independent grounds.
  **Syntax:** the compute is `_isfinited(x)` / `_finitef(x)`, imported under an
  alias from `triton.language.extra.cuda.libdevice`; the DSL binds `erf`,
  `llrint` and `pow` from libdevice, not these.
  **Semantics (the binding one):** VeriTile's arithmetic is over `ℝ`, which has
  no infinities and no NaN, so a faithful `isfinite` is vacuously true at every
  lane. A port would compile and its headline would be provable while saying
  nothing. This needs an IEEE special-value layer, not a libdevice binding.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md).

Structurally this kernel is the same FlagGems pointwise-codegen template as
`relu_strided_buffer` (a `one_tile_per_cta` constexpr branch, `tl.make_block_ptr`
with `boundary_check`, an inlined `@triton.jit` helper, and a grid-stride else
branch over `tl.num_programs(0)`), so the whole proof architecture is already in
place — only the specification is not expressible.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
