# fused_recurrent_delta

- Source file: `fused_recurrent_delta.py`
- Corpus: TritonBench-G v1
- Status: DONE — DSL port (`FusedRecurrentDelta.lean`) of **both** kernels
  (`fused_recurrent_fwd_kernel` + `fused_recurrent_bwd_kernel`), genuine
  standalone delta-rule recurrence spec (`deltaState`), and
  `ComputeCorrect.Realizes` summary
  (`fused_recurrent_delta_output_summary_general`) proven sorry-free.

The forward kernel is the flash-linear-attention **delta rule** recurrent
scan: per `(i_v, i_k, i_bh)` program and time step it reads out the state
(`v_minus = Σ h·k`), stores the delta `v_new = v − v_minus` **in place into
`v`**, applies the rank-1 update `h += k ⊗ (β ⊙ v_new)` (with `β` either a
per-`(b,h,t)` scalar or a headwise row, `IS_HEADWISE_BETA`), and emits
`o = Σ h·q·scale` from the post-update state — plus optional `h0` seed and
`ht` final-state store. The backward kernel runs a reverse-time `d_h` scan
(`dk`/`dv`/`dbeta`, optional `dh0`), a `tl.debug_barrier()`, then a
forward-time recomputation for the in-place `dk` correction and `dq`.

The delta-rule transition `(I − β k kᵀ)` has no geometric closed form, so the
spec is the explicit recurrence `deltaState` defined by recursion over the
input regions (the `reversed_cumsum_scalar` carry-fold precedent). Every
stored output of both kernels is a clause of the bundled dimension-general
main theorem — each forward face realized against `deltaState` /
`vNewClosed` / `outputClosed`, each backward face realized against its
genuine per-step gradient formula over the materialized carried-state
buffers. The cross-step folds threading `h` / `d_h` are the documented
trusted loop boundary (`fused_recurrent_hgrn` / `fused_rwkv6_kernel`
precedent), with the in-place-`v` overwrite consequence stated plainly in
the module docstring. Honest side conditions: `BK ≤ K`, `BV ≤ V`.
