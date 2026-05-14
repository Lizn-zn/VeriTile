# TritonBench-G Proof Blockers

These kernels currently have faithful-looking surface translations but do not
yet have a real `ComputeCorrect.Realizes` theorem. They must not be counted as
complete until the algorithm-layer blocker below is discharged or linked to a
GitHub issue. The public `*_correct` surfaces no longer contain `True` /
`trivial` placeholders; they expose the remaining algorithm-layer obligation as
an explicit hypothesis and reduce it to the named algorithm postconditions.

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
