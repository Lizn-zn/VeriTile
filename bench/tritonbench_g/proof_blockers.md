# TritonBench-G Proof Blockers

These kernels currently have faithful-looking surface translations but do not
yet have a real `ComputeCorrect.Realizes` theorem. They must not be counted as
complete until the blocker below is discharged or linked to a GitHub issue.

## `mean_reduction.py`

- File: `mean_reduction/MeanReduction.lean`
- Blocker: correctness needs a full row-wise sum invariant across
  `for off in range(0, N, BLOCK_N)`.
- Current local spec now describes the full row mean via `meanSpec` as a
  `Fin N` sum over the original row, matching all `off` iterations together,
  but the theorem is still a placeholder until the loop invariant is proved.

## `embedding_triton_kernel.py`

- File: `embedding_triton_kernel/EmbeddingTritonKernel.lean`
- Blocker: correctness needs a `forRange` invariant across
  `range(0, BLOCK_N, BLOCK_NN)` and a two-dimensional write map for all
  `(sequence, dmodel)` lanes written by the loop.
- Current local spec now includes both the per-iteration
  `BLOCK_NN × BLOCK_DMODEL` target (`outOffset2D` / `embeddingSpec2D`) and the
  full `BLOCK_N × BLOCK_DMODEL` post-loop target (`outOffsetFull` /
  `embeddingSpecFull`), but the theorem is still a placeholder until the loop
  invariant is proved.

## `diag_ssm_triton.py`

- File: `diag_ssm_triton/DiagSsmTriton.lean`
- Blocker: correctness needs a recurrence invariant across `tl.for t in length`
  for the state update `s = s * Lambda + x`, with a spec for every stored time
  step, not only the first update.
- Current local spec now includes the recurrence target via `diagSsmStateAfter`
  and `diagSsmForwardSpec`, but the theorem is still a placeholder until the
  loop invariant is proved.
