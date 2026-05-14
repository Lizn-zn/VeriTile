# TritonBench-G Proof Blockers

These kernels currently have faithful-looking surface translations but do not
yet have a real `ComputeCorrect.Realizes` theorem. They must not be counted as
complete until the algorithm-layer blocker below is discharged or linked to a
GitHub issue. The public `*_correct` surfaces no longer contain `True` /
`trivial` placeholders; they expose the remaining algorithm-layer obligation as
an explicit hypothesis and reduce it to the named algorithm postconditions.

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
  prefix to `embedding_kernel_alg_post`. The variant
  `embeddingLoopInvariant_to_alg_post_of_final` now consumes the actual
  `forRange_inv` conclusion `BLOCK_N ≤ final`, so the final readout bridge no
  longer requires the loop offset to stop definitionally at exactly `BLOCK_N`.
  The old-prefix/current-chunk disjointness lemmas
  `embeddingPrefixIndex_ne_currentChunk` and
  `embeddingOldPrefix_outOffset_ne_currentChunk` are available for deriving
  the preservation side of the chunk step from the concrete `tl.store`.
  `embeddingChunkLane_lt_of_aligned_start` captures the pointwise aligned-chunk
  boundary condition needed to map every current `BLOCK_NN` lane into the full
  `BLOCK_N` write map, and `embeddingChunkLaneBound_of_aligned_start` packages
  it in the `∀ lane` shape consumed by the current-chunk scatter step; the
  actual wrapper uses the simple `BLOCK_NN = 1` case.
  The chunk step is factored as `embeddingLoopInvariant_step_of_chunk_write`,
  which combines old-prefix preservation with the current
  `BLOCK_NN × BLOCK_DMODEL` write map to produce the next prefix invariant.
  `embeddingCurrentChunkScatter_write` provides the current-chunk masked scatter
  readback fact needed for the concrete `tl.store`, and
  `embeddingCurrentChunkNoCollision_of_full_injective` derives its no-collision
  premise from full `outOffsetFull` injectivity plus chunk-bound coverage.
  `embeddingCurrentChunkScatter_preserve_old` covers old-prefix preservation for
  the current-chunk scatter using the generic
  `BlockState.scatter_prop_masked_preserves_other_offset` helper.
  `embeddingLoopInvariant_step_of_current_chunk_scatter` packages these pieces
  into the next-prefix invariant for the concrete current-chunk scatter state.
  `embeddingPreLoop`, `embeddingLoopBody`, and `embeddingProjectedBody` now name
  the concrete DSL-expanded algorithm body, with `embedding_kernel_toAlg_body`
  proving that this split is exactly the current `toAlgKernel.body`. The named
  body also records the macro lowering detail that direct `region + offset`
  memory expressions become `MemAccess.region` accesses in the algorithm layer.
  `embeddingPreLoop_step_regs` proves the concrete pre-loop register facts for
  `start_n`, `offs_nn`, and `offs_d`, plus preservation of the input and
  weight reads needed by the first loop iteration.
  `seqLaneIndex_eval_add`, `outOffset2D_eval_add`,
  `storeActive2D_eval_iff`, and `embeddingLoopStoreValue_eval_eq_spec2D`
  normalize the offset and value shapes produced by the DSL expansion to the
  local `outOffset2D` / `storeActive2D` / `embeddingSpec2D` vocabulary.
  `embeddingLoopBody_step_current_chunk_write` now consumes the concrete
  `embeddingLoopBody` execution and proves the current chunk's masked scatter
  readback against `embeddingSpec2D`. `embeddingLoopBody_step_preserve_old`
  proves the same concrete body preserves all previously written prefix lanes.
  `embeddingLoopInvariant_step_of_concrete_body` combines those two facts into
  the per-iteration invariant step for the actual loop body.
  Remaining store-side work is matching the DSL-expanded store to that scatter
  shape. The theorem target is fixed as
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
  `forLoop_inv`. The old-time/current-time disjointness lemmas
  `diagSsmForwardIndex_ne_currentTime` and
  `diagSsmForwardOutOffset_ne_currentTime` are available for deriving the
  preservation side of the time-step write from the concrete `tl.store`.
  `diagSsmMaskedStateTile_succ` exposes the register-update shape needed to
  match `s = s * Lambda + x` on active lanes. The current-time simp lemmas
  `diagSsmForwardOutOffset_currentTime`,
  `diagSsmForwardSpecAt_currentTime`, and
  `diagSsmForwardActive_currentTime` expose the store address, expected value,
  and active predicate for the loop body's current `t` lane.
  `diagSsmForwardCurrentTimeScatter_write` gives the current-time masked
  scatter readback fact needed for the loop body's `tl.store`, and
  `diagSsmForwardCurrentTimeNoCollision_of_out_injective` derives its lane
  no-collision premise from full `diagSsmForwardOutOffset` injectivity. The
  generic semantic helper `BlockState.scatter_prop_masked_preserves_other_offset`
  and the local theorem `diagSsmForwardCurrentTimeScatter_preserve_old` cover
  old-time preservation for the current-time scatter.
  `diagSsmForwardLoopInvariant_step_of_current_time_scatter` packages the
  updated state register plus current-time scatter into the next loop invariant.
  The remaining store-side work is matching the concrete DSL-expanded loop body
  to that register/scatter shape. The theorem surface now requires full
  `diagSsmForwardOutOffset` injectivity over `(time, column)` indices. The
  compute-facing wrapper is discharged once that algorithm-layer postcondition
  is supplied via
  `diag_ssm_forward_kernel_compute_correct_of_algorithm`.
