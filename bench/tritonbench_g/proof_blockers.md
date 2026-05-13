# TritonBench-G Proof Blockers

These kernels currently have faithful-looking surface translations but do not
yet have a real `ComputeCorrect.Realizes` theorem. They must not be counted as
complete until the algorithm-layer blocker below is discharged or linked to a
GitHub issue. The compute-facing wrappers for these files already reduce to
the named algorithm postconditions.

## `mean_reduction.py`

- File: `mean_reduction/MeanReduction.lean`
- Blocker: correctness needs a full row-wise sum invariant across
  `for off in range(0, N, BLOCK_N)`.
- Current local spec now describes the full row mean via `meanSpec` as a
  `Fin N` sum over the original row, matching all `off` iterations together,
  and the intended `_mean` loop accumulator via `meanLanePrefix` /
  `meanAccumulatorSpec`. The pure lane-step fact is now proved as
  `meanLanePrefix_step` and lifted to the accumulator tile by
  `meanAccumulatorSpec_step`; the bridge from the final accumulator to the
  row-wise sum is proved by `meanFromAccumulatorSpec_eq_meanSpec`. The theorem
  target is fixed as `mean_dim_kernel_correct_target`, but the theorem is still
  a placeholder until this accumulator invariant is connected to the actual
  loop body with `forRange_inv`. The compute-facing wrapper is discharged once
  that algorithm-layer postcondition is available via
  `mean_dim_kernel_compute_correct_of_algorithm`.

## `embedding_triton_kernel.py`

- File: `embedding_triton_kernel/EmbeddingTritonKernel.lean`
- Blocker: correctness needs a `forRange` invariant across
  `range(0, BLOCK_N, BLOCK_NN)` and a two-dimensional write map for all
  `(sequence, dmodel)` lanes written by the loop.
- Current local spec now includes both the per-iteration
  `BLOCK_NN × BLOCK_DMODEL` target (`outOffset2D` / `embeddingSpec2D`) and the
  full `BLOCK_N × BLOCK_DMODEL` post-loop target (`outOffsetFull` /
  `embeddingSpecFull`), with `embeddingPrefixActive` identifying the lanes
  already written after a prefix of chunks. The chunk-to-full bridges are
  `outOffset2D_eq_full`, `embeddingSpec2D_eq_full`, and
  `storeActive2D_iff_full`; the theorem target is fixed as
  `embedding_kernel_correct_target`, but the theorem is still a placeholder
  until the per-chunk write invariant is instantiated with `forRange_inv` under
  the full `outOffsetFull` injectivity assumption. The compute-facing wrapper
  is discharged once that algorithm-layer postcondition is available via
  `embedding_kernel_compute_correct_of_algorithm`.

## `diag_ssm_triton.py`

- File: `diag_ssm_triton/DiagSsmTriton.lean`
- Blocker: correctness needs a recurrence invariant across `tl.for t in length`
  for the state update `s = s * Lambda + x`, with a spec for every stored time
  step, not only the first update.
- Current local spec now includes the recurrence target via `diagSsmStateAfter`
  / `diagSsmForwardSpec`, state-register targets via `diagSsmStateTile` and
  `diagSsmMaskedStateTile`, plus the full `(time, column)` write target via
  `diagSsmForwardOutOffset` / `diagSsmForwardSpecAt`, with
  `diagSsmForwardSpecAt_eq_stateTile` bridging stores to the recurrence state.
  The theorem target is fixed as
  `diag_ssm_forward_kernel_correct_target`, but the theorem is still a
  placeholder until the recurrence invariant is instantiated with `forLoop_inv`.
  The theorem surface now requires full `diagSsmForwardOutOffset` injectivity
  over `(time, column)` indices. The compute-facing wrapper is discharged once
  that algorithm-layer postcondition is available via
  `diag_ssm_forward_kernel_compute_correct_of_algorithm`.
