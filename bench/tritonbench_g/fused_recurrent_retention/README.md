# fused_recurrent_retention

- Source file: `fused_recurrent_retention.py`
- Corpus: TritonBench-G v1
- Status: DONE — DSL port (`FusedRecurrentRetention.lean`), genuine decayed
  outer-product closed-form specs, and `ComputeCorrect.Realizes` summary
  (`fused_recurrent_retention_output_summary_general`) proven sorry-free,
  covering **both** launched kernels (forward and autograd backward).

`fused_recurrent_retention_fwd_kernel` is the flash-linear-attention fused
recurrent retention scan: each program `(i_v, i_k, i_bh)` carries a `[BV, BK]`
state `h` across the `0..T` loop with the scalar per-head decay
`b_b = 1 − 2^(−5 − i_h)` (`tl.math.exp2`), updating `h = b_b·h + k_t ⊗ v_t`
and emitting `o_t = Σ_k h·(scale·q_t)` from the **post-update** state, with
optional `initial_state` seed and `final_state` store. The backward kernel
(exercised via `loss.backward()`) runs a forward-time `dq` scan and — after a
verbatim-transcribed `tl.debug_barrier()` — a reverse-time `dk`/`dv` scan
carrying the gradient state `d_h += scale·q_t ⊗ do_t; d_h *= b_b` with
pointer `-=` decrements.

Both full surfaces lower to the algorithm layer. The genuine closed forms are
`stateClosed(m) = seed·b_b^m + Σ_{t<m} k_t⊗v_t·b_b^(m−1−t)` (its `outClosed`
reduction is the classic `o_t = Σ_{s≤t} b_b^(t−s)·(scale·q_t·k_s)·v_s`, see
`outClosed_as_decayed_dots`) and
`dStateClosed(t) = Σ_{t≤u<T} scale·q_u⊗do_u·b_b^(u−t)` with `dq`/`dk`/`dv` as
its reductions — all over input regions only. The cross-step `range(0, T)`
folds are the trusted boundary: each loop-body face is realized against the
closed forms under self-propagating carry invariants
(`HPrev = stateClosed(m)` / `DHPrev = b_b·dStateClosed(m+1)`), following the
`fused_rwkv6_kernel` / `fused_recurrent_hgrn` packaging. Step slices read the
per-time rows unmasked (in-bounds regime; the partial tail-block `other=0`
lanes and the host `o.sum(0)`/`dq.sum(0)`/`dk.sum(0)`/`dv.sum(0)` cross-block
reductions are the documented trusted boundary).
