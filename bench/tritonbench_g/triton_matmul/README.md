# triton_matmul

- Source file: `triton_matmul.py` (upstream `data/TritonBench_G_v1/triton_matmul.py`)
- Corpus: TritonBench-G v1
- Size: 133 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `TritonMatmul.lean`, main theorems
  `triton_matmul_f16_closed_form_correct` / `triton_matmul_f8_closed_form_correct`
  (exec closed forms) and `triton_matmul_f16_io_correctness` /
  `triton_matmul_f8_io_correctness` (`⊨[R]` streaming io faces),
  all dimension-general, 0 `sorry`.

The upstream kernel is the Triton-tutorial L2-grouped tiled GEMM `C = A × B`
with two twists over the `matmul_triton_autotune` twin: out-of-range A-row /
B-column indices are **clamped** to `0` (`tl.where(offs < M, offs, 0)` +
`tl.max_contiguous`/`tl.multiple_of` hints) instead of `% M`-wrapped, and the
epilogue downcast branches on the **output buffer's compile-time dtype** —
`float8e4nv` when the host allocates fp8, `float16` otherwise.

Both epilogue arms are proven, through one od-generic proof stack: every
active output cell holds `od(Σ_{k<K} A[i,k]·B[k,j])` (exec closed form), and
on the streaming `⊨[R]` face the output window reads back as `od`-typed cells
holding `R.round od (Σ A·B)` for every rounding model `R` — the fp8 arm is
**the corpus's first fp8 matmul face**. Side conditions: the
`K = BLOCK_SIZE_K · numKBlocks` trip-count presentation (template precedent),
`scn = 1` + `BN ≤ scm` (store-lane injectivity), clean-input `hundef` on the
exact face.

Translation-surface blocker (registered in `proof_blockers.md`): the
constexpr epilogue split into two Lean surfaces (`matmul_tma` precedent);
`tl.cdiv(K, BLOCK_SIZE_K)` supplied as the antiquoted `numKBlocks`; the
K-loop counter spelled `kk`. The `triton.jit(launch_metadata=...)` hook and
the host's per-dtype config table are the trusted boundary.
