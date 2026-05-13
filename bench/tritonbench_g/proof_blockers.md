# TritonBench-G Proof Blockers

These kernels currently have faithful-looking surface translations but do not
yet have a real `ComputeCorrect.Realizes` theorem. They must not be counted as
complete until the algorithm-layer blocker below is discharged or linked to a
GitHub issue. The public `*_correct` surfaces no longer contain `True` /
`trivial` placeholders; they expose the remaining algorithm-layer obligation as
an explicit hypothesis and reduce it to the named algorithm postconditions.

## `mean_reduction.py`

- File: `mean_reduction/MeanReduction.lean`
- Blocker: correctness needs a full row-wise sum invariant across
  `for off in range(0, N, BLOCK_N)`.
- Current local spec now describes the full row mean via `meanSpec` as a
  `Fin N` sum over the original row, matching all `off` iterations together,
  and the intended `_mean` loop accumulator via `meanLanePrefix` /
  `meanAccumulatorSpec`. Because the loop body masks inactive rows, the actual
  loop invariant must use `meanMaskedAccumulatorSpec`; its zero, step, and
  active-row projection lemmas are proved locally. The per-iteration masked
  load tile is captured by `meanChunkLoadSpec`, with
  `meanMaskedAccumulatorSpec_step_add` matching the actual `_mean += a` update.
  The `forRange_inv` predicate is now named `meanLoopInvariant`, with
  `meanLoopInvariant_init_of_zero_reg` covering the zero-initialized entry and
  `meanLoopInvariant_step_of_accumulator_update` reducing the loop step to the
  concrete register update produced by the body. The final active-row readout
  bridge is proved by
  `meanFromMaskedAccumulatorSpec_eq_meanSpec`; the register/reduction-facing
  bridge is now named `meanLoopInvariant_register_reduceSum_to_meanSpec`, and
  it connects the `_mean` tile register plus `tl.sum(axis=1)` readout to the
  final `meanSpec`. The pure lane-step fact is proved as `meanLanePrefix_step`
  and lifted to the accumulator tile by `meanAccumulatorSpec_step`; the bridge
  from the final accumulator to the row-wise sum is proved by
  `meanFromAccumulatorSpec_eq_meanSpec`. The theorem target is fixed as
  `mean_dim_kernel_correct_target`, and the public theorem exposes the
  remaining algorithm-layer postcondition as an explicit hypothesis until this
  accumulator invariant is connected to the actual loop body with
  `forRange_inv`. The compute-facing wrapper is discharged once that
  algorithm-layer postcondition is supplied via
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
  `storeActive2D_iff_full`. The `forRange_inv` predicate is now named
  `embeddingLoopInvariant`, with `embeddingLoopInvariant_zero` covering the
  vacuous prefix and `embeddingLoopInvariant_to_alg_post` bridging a completed
  prefix to `embedding_kernel_alg_post`. The chunk step is factored as
  `embeddingLoopInvariant_step_of_chunk_write`, which combines old-prefix
  preservation with the current `BLOCK_NN × BLOCK_DMODEL` write map to produce
  the next prefix invariant. The theorem target is fixed as
  `embedding_kernel_correct_target`, and the public theorem exposes the
  remaining algorithm-layer postcondition as an explicit hypothesis until that
  chunk step is instantiated with the actual loop body via
  `forRange_inv` under the full `outOffsetFull` injectivity assumption. The
  compute-facing wrapper is discharged once that algorithm-layer postcondition
  is supplied via
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
  The `forLoop_inv` predicate is now named `diagSsmForwardLoopInvariant`, with
  `diagSsmForwardLoopInvariant_zero` covering the initial loaded state,
  `diagSsmForwardLoopInvariant_step_of_time_write` factoring one time-step
  register/write update, and `diagSsmForwardLoopInvariant_to_alg_post`
  bridging a completed loop to `diag_ssm_forward_kernel_alg_post`. The theorem
  target is fixed as
  `diag_ssm_forward_kernel_correct_target`, and the public theorem exposes the
  remaining algorithm-layer postcondition as an explicit hypothesis until that
  recurrence invariant is instantiated with the actual loop body via
  `forLoop_inv`. The theorem surface now requires full
  `diagSsmForwardOutOffset` injectivity over `(time, column)` indices. The
  compute-facing wrapper is discharged once that algorithm-layer postcondition
  is supplied via
  `diag_ssm_forward_kernel_compute_correct_of_algorithm`.
