import VeriTile.Triton

/-!
# `fused_recurrent_delta` — genuine delta-rule recurrence correctness

`fused_recurrent_fwd_kernel` is the flash-linear-attention **delta rule**
forward recurrent scan. Program `(i_v, i_k, i_bh)` carries a `[BV, BK]` state
tile `h` across the `0..T` time loop. With `q_t = q[t] · scale`, per step `t`:

* `v_minus[j_v] = Σ_{j_k} h[j_v,j_k] · k_t[j_k]`         (state readout)
* `v_new[j_v]   = v_t[j_v] − v_minus[j_v]`               (the delta; stored
                                                          **in place** into `v`)
* `h[j_v,j_k]  += k_t[j_k] · (β_t[j_v] · v_new[j_v])`    (rank-1 state update)
* `o_t[j_v]     = Σ_{j_k} h[j_v,j_k] · q_t[j_k]`         (readout of the
                                                          *post-update* state)

with an optional `h0` initial-state seed and an optional `ht` final-state
store. `β` is either a per-`(b,h,t)` scalar or a headwise `[.., T, V]` row
(`IS_HEADWISE_BETA`). The companion `fused_recurrent_bwd_kernel` runs a
reverse-time gradient scan producing `dk`/`dv`/`dbeta`, a `tl.debug_barrier()`,
then a forward-time recomputation producing the `dk` correction and `dq`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies (forward and backward). The host launch (grid shape,
`NK = 1` assertion, autograd plumbing, the `dq/dk/dv/dbeta` `.sum(0)`
reductions on the host) is the *trusted boundary*, not a proof obligation
here. Because the program ids are universally quantified, each per-program
statement covers every program of the grid.

## Genuine recurrence spec (NOT a self-reference)

The delta-rule state has no geometric-decay closed form (its transition is
`(I − β_t k_t k_tᵀ)`-shaped), so the specification is the **explicit
standalone recurrence** `deltaState` — a Lean function defined by recursion on
the step count over the *input* regions `k, v, beta, h0`:

```
deltaState 0       = h0-seed (or 0)
deltaState (m+1)   = deltaState m
                     + k_m ⊗ (β_m ⊙ (v_m − (deltaState m) · k_m))
```

This is never a read-back of the kernel's own output (the
`reversed_cumsum_scalar` carry-fold face is the precedent for
recurrence-valued specs). The per-step faces below are each realized against
`deltaState` / its derived forms `vNewClosed` / `outputClosed`.

## Proof architecture

The headline is a flat 11-way conjunction; its `refine` has eleven independent
goals and there is **no implication chain between them** — in particular no
clause discharges another clause's carry hypothesis. Each line below is one
conjunct, in order:

```
fused_recurrent_delta_output_summary_general          ← TOP THEOREM
  1  fused_recurrent_delta_fwd_surface_toAlgorithm_supported     fwd surface lowers
  2  fused_recurrent_delta_bwd_surface_toAlgorithm_supported     bwd surface lowers
  3  fused_recurrent_delta_vnew_step_closed_form         v ← v_m − h·k_m  (vNewClosed m)
  4  fused_recurrent_delta_state_step_closed_form        HOut ← deltaState(m+1)
  5  fused_recurrent_delta_output_step_closed_form       o ← outputClosed m
  6  fused_recurrent_delta_bwd_dk_step_slice_compute_correct              dk row (loop 1)
  7  fused_recurrent_delta_bwd_dv_step_slice_compute_correct              dv row (loop 1)
  8  fused_recurrent_delta_bwd_dbeta_step_slice_headwise_compute_correct  dbeta row
  9  fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta_compute_correct dbeta scalar
 10  fused_recurrent_delta_bwd_dk_correction_step_sequenced               dk fixup (loop 2)
 11  fused_recurrent_delta_bwd_dq_step_slice_compute_correct              dq row (loop 2)

supporting algebra (used INSIDE clauses 3–5, not extra conjuncts):
  deltaState / deltaState_succ            the standalone recurrence
  stateStepSpec_eq_deltaState_succ        one body advances it, given `hPrev`
  dkCorrClosed                            loop-1 `dk`/`dv` values, sequenced into
                                          clause 10 (via the slice face
                                          `..._dk_correction_step_slice_compute_correct`)
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. `tl.debug_barrier()` is transcribed in place; it is an
intra-program fence and erases to a no-op at the algorithm layer. All four
`tl.constexpr` flags (`USE_INITIAL_STATE`, `STORE_FINAL_STATE`,
`IS_HEADWISE_BETA`, and the backward `USE_DH0`/`USE_DHT`) are modeled and flow
through the surfaces verbatim as `Bool` parameters.

The cross-step folds threading `h` (forward), `d_h` (backward loop 1) and the
recomputed `h` (backward loop 2) are the **trusted loop boundary**, exactly as
in `fused_recurrent_hgrn` / `fused_rwkv6_kernel`: the carried state is
presented to each step face as a materialized buffer (`HPrev`/`HOut`/`DHPrev`),
and under the *assumed* carry invariant `HPrev = deltaState(m)` the forward
loop-body faces realize `vNewClosed(m)`, `deltaState(m+1)` and `outputClosed(m)`
(theorem `stateStepSpec_eq_deltaState_succ`, algebraic core `deltaState_succ`).
No clause discharges another clause's carry invariant, and there is no proven
chaining from one step's output buffer to the next step's input buffer. Two boundary consequences are stated plainly:

* the forward kernel **overwrites `v` in place** (row `t` is rewritten to
  `v_new` at step `t`, after being read); every face's recurrence hypothesis
  and conclusion are evaluated in that face's *own* initial state `s`, and the
  cross-step reconciliation of `deltaState`'s `v`-row references with the
  pristine input rows (which the kernel never re-reads after overwriting) is
  part of the trusted fold;
* the backward kernel reads the *updated* `v` (the stored `v_new` deltas) —
  its faces are genuine over the backward launch's actual input regions, and
  the backward per-step gradient formulas are realized against the
  materialized carried buffers (`DHPrev`, loop-2 `HPrev`) with the reverse /
  forward folds trusted, following the `fused_recurrent_hgrn` backward
  precedent.

Output-offset injectivity side conditions (`BK ≤ K`, `BV ≤ V` for the
`[BV,BK]`/`[BK,BV]` state tiles) are honest structural hypotheses; the Python
regression (`K = BK = 16`, `V = 32`, `BV ∈ {8, 32}`) satisfies them.

**The load masks — a disclosed gap, scoped by `hFullK` / `hFullV`.** Every
Python load in both kernels is guarded (`mask_bk` / `mask_bv`, `other = 0`); the
step slices in this file load *unmasked* (their masked *stores* are transcribed
faithfully). For a partial tile Python reads `0` where these slices read live
memory, so the headline pins the full-tile regime with the two explicit
hypotheses `hFullK` / `hFullV` rather than pretending to cover partial tiles.
Why the masks are not simply put back: every face here observes a value produced
by a `tl.sum` over the *other* axis, so masking the loads turns each executed
lane into an `if mask then some … else some 0` carrier inside a `WithBot ℝ`
reduction, whose lane index is a `TileShape.insertAxisIndex`-projected index that
`simp`/`rw` cannot normalize (the generic `insertAxisIndex_succ` simp lemma does
not fire on the `Fin 2` literal `1`); `Semantics/MaskedReduction.lean` carries
carrier bridges only for two-factor 1-D shapes, not for these 2-D multi-factor
ones. Closing the gap needs that library-level bridge first — the sibling port
`fused_rwkv6_kernel` shows the shape of the fix where a face is *elementwise*
(its state-update slice is now fully masked and needs no full-tile hypothesis),
and every face in this file is a reduction face.

## DSL-forced transcription notes (all mechanical, documented per
`review_criteria.md`)

* Python `do` (grad-output pointer) is spelled `do_` (`do` is a Lean keyword);
  the Python loop variables `_` are spelled `_i`.
* DSL pointers are `ℕ`-offsets and do not support `ptr − literal`, so the
  scalar-beta pointer initializations `… + T - 1` (the non-headwise
  `p_beta`/`p_dbeta` forms) are transcribed with the trailing subtraction
  parenthesized into the `ℕ` domain (`… + ($(T) - $(1))`); the groupings
  agree whenever `T ≥ 1`, and for `T = 0` the pointer is never dereferenced
  (the `range(T)` loop is empty). The `(T - 1) * K` / `(T - 1) * V` offsets
  are already parenthesized in the Python and transcribe verbatim.
* The Python conditional increments `p_beta += V if IS_HEADWISE_BETA else 1`
  (and the `p_dbeta` analogue / the conditional `p_beta`/`p_dbeta`/`b_beta`/
  `d_beta` definitions) are transcribed as `if IS_HEADWISE_BETA { … } else
  { … }` DSL gates — the whitelisted constexpr-branch form.

Translation-surface blocker: the Python conditional-expression pointer
increments (`p_beta += V if IS_HEADWISE_BETA else 1` and the `p_dbeta`
analogue) and conditional definitions are spelled as `if IS_HEADWISE_BETA
{ … } else { … }` constexpr gates, one statement per arm — this adds one
`if` to the control-flow count and repeats the `p_beta`/`p_dbeta` lhs once
per arm relative to the Python ternary form. `do` is spelled `do_` (Lean
keyword) and `_` loop variables are spelled `_i`; the scalar-beta `… + T - 1`
pointer initializations are parenthesized into the ℕ domain
(`… + ($(T) - $(1))`). All arms transcribe the Python verbatim otherwise.
-/

namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `fused_recurrent_delta_output_summary_general` —
shape-general, but scoped to eight single-step **slices** (one of which writes the
internal carry register `HOut`, not a Python tensor), not to the launched
kernels. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `fused_recurrent_delta.py`'s
`fused_recurrent_fwd_kernel`.

All three `tl.constexpr` flags are `Bool` parameters gating the verbatim
branches; the `IS_HEADWISE_BETA` branches keep the Python `p_beta`/`b_beta`
shapes (`[BV]` row vs scalar). See the module docstring for the three
mechanical transcription notes (`_i` loop variable, `ℕ`-domain grouping of
trailing `- 1`, `if`-gate form of the conditional `p_beta` increment). -/
def fused_recurrent_delta_fwd_surface
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  if IS_HEADWISE_BETA {
    p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  } else {
    p_beta = beta + i_bh * $(T)
  }
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    _v_minus = tl.sum(h * b_k[None, :], axis=1)
    b_v -= _v_minus
    if IS_HEADWISE_BETA {
      b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    } else {
      b_beta = tl.load(p_beta).to(tl.float32)
    }
    tl.store(p_v, (b_v).to(p_v.dtype.element_ty), mask=mask_bv)
    b_v *= b_beta
    h += b_k[None, :] * b_v[:, None]
    _o = h * b_q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    if IS_HEADWISE_BETA {
      p_beta += $(V)
    } else {
      p_beta += $(1)
    }
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
}

/-- The full delta-rule forward surface lowers to the algorithm layer, for
every flag combination. -/
theorem fused_recurrent_delta_fwd_surface_toAlgorithm_supported
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) :
    ∃ alg, (fused_recurrent_delta_fwd_surface q k v beta o h0 ht s_qk_h s_vo_h
      B H T K V BK BV scale USE_INITIAL_STATE STORE_FINAL_STATE
      IS_HEADWISE_BETA).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_delta_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Faithful transcription of `fused_recurrent_delta.py`'s
`fused_recurrent_bwd_kernel`: the reverse-time `d_h` scan storing
`dk`/`dv`/`dbeta`, the optional `dh0` store, the `tl.debug_barrier()` fence,
then the forward-time recomputation of `h` storing the `dk` correction and
`dq`. `do` is spelled `do_`; see the module docstring for the other two
mechanical notes. -/
def fused_recurrent_delta_bwd_surface
    (q k v beta dht dh0 do_ dq dk dv dbeta h0 : RegionName)
    (s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  mask_bk = i_k * $(BK) + tl.arange(0, $(BK)) < $(K)
  mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(V)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  if IS_HEADWISE_BETA {
    p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  } else {
    p_beta = beta + i_bh * $(T) + ($(T) - $(1))
  }
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  if IS_HEADWISE_BETA {
    p_dbeta = dbeta + (i_bh + i_k * $(B) * $(H) + i_v * $(B) * $(H) * $(NK)) * $(s_vo_h) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  } else {
    p_dbeta = dbeta + (i_bh + i_v * $(B) * $(H)) * $(T) + ($(T) - $(1))
  }
  d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_DHT {
    p_ht = dht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
  }
  for _i in range($(T)) {
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * $(scale)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA {
      b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    } else {
      b_beta = tl.load(p_beta).to(tl.float32)
    }
    d_h += b_q[:, None] * b_do[None, :]
    d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
    d_v = tl.sum(d_h * b_k[:, None], axis=0)
    if IS_HEADWISE_BETA {
      d_beta = d_v * b_v
    } else {
      d_beta = tl.sum(d_v * b_v)
    }
    d_v = d_v * b_beta
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
    if IS_HEADWISE_BETA {
      tl.store(p_dbeta, (d_beta).to(p_dbeta.dtype.element_ty), mask=mask_bv)
    } else {
      tl.store(p_dbeta, (d_beta).to(p_dbeta.dtype.element_ty))
    }
    d_h -= b_k[:, None] * d_v[None, :]
    p_do -= $(V)
    p_q -= $(K)
    p_k -= $(K)
    p_v -= $(V)
    p_dk -= $(K)
    p_dv -= $(V)
    if IS_HEADWISE_BETA {
      p_dbeta -= $(V)
    } else {
      p_dbeta -= $(1)
    }
    if IS_HEADWISE_BETA {
      p_beta -= $(V)
    } else {
      p_beta -= $(1)
    }
  }
  if USE_DH0 {
    p_dh0 = dh0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    tl.store(p_dh0, (d_h).to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])
  }
  tl.debug_barrier()
  h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  if IS_HEADWISE_BETA {
    p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  } else {
    p_beta = beta + i_bh * $(T)
  }
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  if USE_INITIAL_STATE {
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for i in range($(0), $(T), $(1)) {
    d_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)
    d_v = tl.load(p_dv, mask=mask_bv, other=0).to(tl.float32)
    d_k -= tl.sum(d_v[None, :] * h, axis=1)
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    if IS_HEADWISE_BETA {
      b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    } else {
      b_beta = tl.load(p_beta).to(tl.float32)
    }
    b_v *= b_beta
    h += b_k[:, None] * b_v[None, :]
    _d_q = h * b_do[None, :]
    d_q = tl.sum(_d_q, axis=1) * $(scale)
    tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
    p_k += $(K)
    p_do += $(V)
    p_v += $(V)
    p_dk += $(K)
    p_dv += $(V)
    p_dq += $(K)
    if IS_HEADWISE_BETA {
      p_beta += $(V)
    } else {
      p_beta += $(1)
    }
  }
}

/-- The full delta-rule backward surface — both reverse and forward loops,
the `dh0` branch, and the `tl.debug_barrier()` fence (which erases to a no-op
at the algorithm layer) — lowers to the algorithm layer, for every flag
combination. -/
theorem fused_recurrent_delta_bwd_surface_toAlgorithm_supported
    (q k v beta dht dh0 do_ dq dk dv dbeta h0 : RegionName)
    (s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT : Bool) :
    ∃ alg, (fused_recurrent_delta_bwd_surface q k v beta dht dh0 do_ dq dk dv
      dbeta h0 s_qk_h s_vo_h NK B H T K V BK BV scale USE_INITIAL_STATE
      IS_HEADWISE_BETA USE_DH0 USE_DHT).toAlgorithm? = Except.ok alg := by
  simp [fused_recurrent_delta_bwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Index / offset helpers

`s.pids 0 = i_v`, `s.pids 1 = i_k`, `s.pids 2 = i_bh`. The forward state tile
is `[BV, BK]` (`idx.1` the value axis `j_v`, `idx.2.1` the key axis `j_k`);
the backward carried tiles (`d_h` and the recomputed `h`) are `[BK, BV]`
(`idx.1 = j_k`, `idx.2.1 = j_v`). Both share the flattened row-major state
memory layout `i_bh·K·V + j_k·V + j_v` of `h0`/`ht`. -/

def vIndex (s : BlockState) (BV : Nat) (jv : Fin BV) : Nat :=
  s.pids 0 * BV + jv.val

def kIndex (s : BlockState) (BK : Nat) (jk : Fin BK) : Nat :=
  s.pids 1 * BK + jk.val

/-- Flattened `[BV, BK]` state / `h0` / `ht` address
(row-major `i_bh·K·V + j_k·V + j_v`). -/
def stateOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.2.1 * V + vIndex s BV idx.1

/-- The same flattened state address for the backward `[BK, BV]` tile
orientation (`d_h`, loop-2 `h`). -/
def dhOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + kIndex s BK idx.1 * V + vIndex s BV idx.2.1

def activeV (s : BlockState) (V BV : Nat) (jv : Fin BV) : Prop :=
  vIndex s BV jv < V

instance activeVDecidable (s : BlockState) (V BV : Nat) (jv : Fin BV) :
    Decidable (activeV s V BV jv) := by
  unfold activeV; infer_instance

def activeK (s : BlockState) (K BK : Nat) (jk : Fin BK) : Prop :=
  kIndex s BK jk < K

instance activeKDecidable (s : BlockState) (K BK : Nat) (jk : Fin BK) :
    Decidable (activeK s K BK jk) := by
  unfold activeK; infer_instance

/-! ## Genuine input readers (over the *input* regions)

`kVal/vVal` read the kernel's exact block-pointer layouts at time row `t`
through the shared `[T, K]` k-layout (`q`, `k`) and `[T, V]` v-layout (`v`,
`do`, headwise `beta`). `betaVal` selects the headwise v-layout row or the
per-`(b,h,t)` scalar according to `IS_HEADWISE_BETA`. -/

/-- Element `R[i_bh][t, j_k]` of the shared `[T, K]` **k-layout** at time row
`t`, key channel `j_k` (offset `i_bh·s_qk_h + (i_k·BK + j_k) + t·K`). Both `q`
and `k` block pointers use this layout. -/
noncomputable def kVal (s : BlockState) (r : RegionName)
    (s_qk_h K BK : Nat) (t : Nat) (jk : Fin BK) : ℝ :=
  s.readMem r (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)

/-- `q[t][j_k]·scale` — the kernel multiplies the loaded `q` row by `scale`. -/
noncomputable def qVal (s : BlockState) (q : RegionName)
    (s_qk_h K BK : Nat) (scale : ℝ) (t : Nat) (jk : Fin BK) : ℝ :=
  scale * kVal s q s_qk_h K BK t jk

/-- Element `R[i_bh][t, j_v]` of the `[T, V]` **v-layout** at time row `t`,
value channel `j_v` (offset `i_bh·s_vo_h + (i_v·BV + j_v) + t·V`). The `v`,
`do` and headwise-`beta` block pointers use this layout. -/
noncomputable def vVal (s : BlockState) (r : RegionName)
    (s_vo_h V BV : Nat) (t : Nat) (jv : Fin BV) : ℝ :=
  s.readMem r (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)

/-- `β_t[j_v]`: the headwise v-layout row when `IS_HEADWISE_BETA`, else the
per-`(b,h,t)` scalar `beta[i_bh·T + t]` (broadcast over `j_v`). -/
noncomputable def betaVal (s : BlockState) (beta : RegionName)
    (IS_HEADWISE_BETA : Bool) (s_vo_h T V BV : Nat)
    (t : Nat) (jv : Fin BV) : ℝ :=
  if IS_HEADWISE_BETA then
    s.readMem beta (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)
  else
    s.readMem beta (s.pids 2 * T + t)

noncomputable def h0Val (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (idx : TileIndex [BV, BK]) : ℝ :=
  s.readMem h0 (stateOffset s K V BK BV idx)

/-- Seeded initial state `h^(0)`: `h0` if `USE_INITIAL_STATE` else `0`. -/
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : ℝ :=
  if USE_INITIAL_STATE then h0Val s h0 K V BK BV idx else 0

/-! ## Genuine standalone recurrence spec

`deltaState m` is the delta-rule state after processing time steps
`0, …, m−1`, defined by **recursion on `m`** over the *input* regions
`k, v, beta, h0` — never a read-back of the kernel's own output. The
delta-rule transition has no geometric closed form, so the recurrence itself
is the specification (`reversed_cumsum_scalar` carry-fold precedent). -/

/-- **Genuine delta-rule state recurrence.** Tile element `idx = (j_v, j_k)`:

```
deltaState 0       = stateSeed                    (h0 or 0)
deltaState (m+1)   = deltaState m
                     + k_m[j_k] · (β_m[j_v] · (v_m[j_v] − Σ_{j'_k} deltaState m [j_v,j'_k] · k_m[j'_k]))
```

A standalone specification over the input regions `k, v, beta, h0`. -/
noncomputable def deltaState (s : BlockState) (k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) :
    Nat → TileIndex [BV, BK] → ℝ
  | 0, idx => stateSeed s h0 USE_INITIAL_STATE K V BK BV idx
  | m + 1, idx =>
    deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
        s_qk_h s_vo_h T K V BK BV m idx
      + kVal s k s_qk_h K BK m idx.2.1 *
          (betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV m idx.1 *
            (vVal s v s_vo_h V BV m idx.1
              - ∑ jk : Fin BK,
                  deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
                      s_qk_h s_vo_h T K V BK BV m (idx.1, jk, PUnit.unit)
                    * kVal s k s_qk_h K BK m jk))

/-- The state readout `v_minus` at step `m`, lane `j_v`:
`Σ_{j_k} deltaState(m)[j_v,j_k] · k_m[j_k]`. -/
noncomputable def vMinusClosed (s : BlockState) (k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) (m : Nat) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
        s_qk_h s_vo_h T K V BK BV m (jv, jk, PUnit.unit)
      * kVal s k s_qk_h K BK m jk

/-- **Genuine closed form of the in-place `v` writeback at step `m`**:
the delta `v_new = v_m − v_minus`. -/
noncomputable def vNewClosed (s : BlockState) (k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) (m : Nat) (jv : Fin BV) : ℝ :=
  vVal s v s_vo_h V BV m jv
    - vMinusClosed s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
        s_qk_h s_vo_h T K V BK BV m jv

/-- **The delta-rule carry-fold recurrence, unrolled one step**:
`deltaState(m+1) = deltaState(m) + k_m ⊗ (β_m ⊙ v_new(m))`. This is the exact
closed-form counterpart of the Python loop body
`h += b_k[None,:] * (b_beta * (b_v - v_minus))[:,None]`. -/
theorem deltaState_succ (s : BlockState) (k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) (m : Nat) (idx : TileIndex [BV, BK]) :
    deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
        s_qk_h s_vo_h T K V BK BV (m + 1) idx
      = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV m idx
        + kVal s k s_qk_h K BK m idx.2.1 *
            (betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV m idx.1 *
              vNewClosed s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
                s_qk_h s_vo_h T K V BK BV m idx.1) := by
  simp only [deltaState, vNewClosed, vMinusClosed]

/-- **Genuine closed form of output row `m`, lane `j_v`** — the readout of the
*post-update* state:
`o_m[j_v] = Σ_{j_k} deltaState(m+1)[j_v,j_k] · scale·q_m[j_k]`. -/
noncomputable def outputClosed (s : BlockState) (q k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ) (m : Nat) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
        s_qk_h s_vo_h T K V BK BV (m + 1) (jv, jk, PUnit.unit)
      * qVal s q s_qk_h K BK scale m jk

/-! ## Row offsets of the per-step stores -/

/-- The `v`-layout row address at time `t`, lane `j_v` — shared by the `v`
in-place writeback and the `v`/`do`/headwise-`beta` loads. -/
def vRowOffset (s : BlockState) (t s_vo_h V BV : Nat) (jv : Fin BV) : Nat :=
  s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V

/-- The masked output address at lane `j_v` for time row `t` — the kernel's
exact `o + (i_bh + i_k·B·H)·s_vo_h + i_v·BV + j_v + t·V` layout (also the
`dv` store layout of the backward kernel). -/
def outOffset (s : BlockState) (t s_vo_h B H V BV : Nat) (jv : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV + jv.val + t * V

/-! ## `v_new` writeback step slice (the in-place delta store)

One loop body's `v` writeback, isolated from the cross-step loop: with the
materialized pre-update state tile `HPrev`, it loads the time-`t` rows of `k`
and `v`, forms `v_minus = Σ_{j_k} HPrev·k_t` and masked-stores the delta
`v_t − v_minus` back into `v` at row `t` (the kernel's in-place update; the
row is read before it is written). -/

def fused_recurrent_delta_vnew_step_slice
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  _v_minus = tl.sum(prev * b_k[None, :], axis=1)
  b_v -= _v_minus
  tl.store(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    (b_v).to(v.dtype.element_ty), mask=mask_bv)
}

/-- The per-lane spec of one `v_new` writeback:
`v_t[j_v] − Σ_{j_k} HPrev[j_v,j_k] · k_t[j_k]`. -/
noncomputable def vnewStepSpec (s : BlockState) (HPrev k v : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (jv : Fin BV) : ℝ :=
  s.readMem v (vRowOffset s t s_vo_h V BV jv)
    - ∑ jk : Fin BK,
        s.readMem HPrev (stateOffset s K V BK BV (jv, jk, PUnit.unit))
          * kVal s k s_qk_h K BK t jk

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_vnew_step_slice_correct
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => vRowOffset s t s_vo_h V BV jv)) :
    ∀ jv : Fin BV,
      let outAddr := vRowOffset s t s_vo_h V BV jv
      (exec (fused_recurrent_delta_vnew_step_slice HPrev k v
            t s_qk_h s_vo_h K V BK BV) s).map (·.readMem v outAddr)
        = some (if activeV s V BV jv then
            vnewStepSpec s HPrev k v t s_qk_h s_vo_h K V BK BV jv
          else s.readMem v outAddr) := by
  intro jv
  simp [exec, fused_recurrent_delta_vnew_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComparableDType.lt, stateOffset, vRowOffset, activeV,
        vIndex, kIndex, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx => s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx =>
      s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V)
        - ∑ jk : Fin BK,
            s.readMem HPrev
                (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V +
                  (s.pids 0 * BV + idx.1.val)) *
              s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : vRowOffset s t s_vo_h V BV a = vRowOffset s t s_vo_h V BV b := by
      simpa [offsetFn, vRowOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem v (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem v (offsetFn (jv, PUnit.unit)) =
    if activeV s V BV jv then
      vnewStepSpec s HPrev k v t s_qk_h s_vo_h K V BK BV jv
    else s.readMem v (offsetFn (jv, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jv, PUnit.unit)]
  by_cases hjv : s.pids 0 * BV + jv.val < V
  · simp only [P, hjv, if_true, activeV, vIndex]
    simp only [valueFn, vnewStepSpec, stateOffset, kVal, vVal, vRowOffset,
      kIndex, vIndex]
  · simp only [P, activeV, vIndex, hjv, if_false, BlockState.setReg_readMem]

theorem fused_recurrent_delta_vnew_step_slice_compute_correct
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => vRowOffset s t s_vo_h V BV jv)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_vnew_step_slice HPrev k v
        t s_qk_h s_vo_h K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (v, vRowOffset s t s_vo_h V BV jv)))
      (expected := fun jv : Fin BV =>
        vnewStepSpec s HPrev k v t s_qk_h s_vo_h K V BK BV jv) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_delta_vnew_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jv hActive
  have h := fused_recurrent_delta_vnew_step_slice_correct HPrev k v
    t s_qk_h s_vo_h K V BK BV s hOutInj jv
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## State-update step slice (the delta-rule carry-fold body)

One loop body's state update, isolated from the cross-step loop: with the
materialized pre-update state tile `HPrev`, it loads the time-`t` rows of
`k`, `v` and `beta`, recomputes `v_minus`, the delta `b_v − v_minus`, the
beta scaling, and stores `HPrev + k_t ⊗ (β_t ⊙ v_new)` into the state buffer
`HOut`. The two `IS_HEADWISE_BETA` compile-time specializations differ in the
`b_beta` load shape (`[BV]` row vs scalar), exactly as in the surface. -/

def fused_recurrent_delta_state_step_slice_headwise
    (HPrev k v beta HOut : RegionName) (t s_qk_h s_vo_h T K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  _v_minus = tl.sum(prev * b_k[None, :], axis=1)
  b_v -= _v_minus
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_v *= b_beta
  acc = prev + b_k[None, :] * b_v[:, None]
  tl.store(HOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(HOut.dtype.element_ty))
}

def fused_recurrent_delta_state_step_slice_scalarbeta
    (HPrev k v beta HOut : RegionName) (t s_qk_h s_vo_h T K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  _v_minus = tl.sum(prev * b_k[None, :], axis=1)
  b_v -= _v_minus
  b_beta = tl.load(beta + i_bh * $(T) + $(t))
  b_v *= b_beta
  acc = prev + b_k[None, :] * b_v[:, None]
  tl.store(HOut + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]),
    (acc).to(HOut.dtype.element_ty))
}

/-- The arithmetic spec of one state-update body (both beta modes, selected
by `IS_HEADWISE_BETA` through `betaVal`):
`HPrev + k_t ⊗ (β_t ⊙ (v_t − Σ_{j_k} HPrev·k_t))`. -/
noncomputable def stateStepSpec (s : BlockState) (HPrev k v beta : RegionName)
    (IS_HEADWISE_BETA : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat)
    (idx : TileIndex [BV, BK]) : ℝ :=
  s.readMem HPrev (stateOffset s K V BK BV idx)
    + kVal s k s_qk_h K BK t idx.2.1 *
        (betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV t idx.1 *
          (vVal s v s_vo_h V BV t idx.1
            - ∑ jk : Fin BK,
                s.readMem HPrev (stateOffset s K V BK BV (idx.1, jk, PUnit.unit))
                  * kVal s k s_qk_h K BK t jk))

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_state_step_slice_headwise_correct
    (HPrev k v beta HOut : RegionName)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s K V BK BV idx)) :
    ∀ idx : TileIndex [BV, BK],
      let outAddr := stateOffset s K V BK BV idx
      (exec (fused_recurrent_delta_state_step_slice_headwise HPrev k v beta HOut
            t s_qk_h s_vo_h T K V BK BV) s).map (·.readMem HOut outAddr)
        = some (stateStepSpec s HPrev k v beta Bool.true
            t s_qk_h s_vo_h T K V BK BV idx) := by
  intro idx
  simp [exec, fused_recurrent_delta_state_step_slice_headwise, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComparableDType.lt, stateOffset, vRowOffset,
        vIndex, kIndex, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV, BK] → Nat :=
    fun idx => s.pids 2 * K * V +
      (s.pids 1 * BK + idx.2.1.val) * V + (s.pids 0 * BV + idx.1.val)
  let valueFn : TileIndex [BV, BK] → ℝ :=
    fun idx =>
      s.readMem HPrev
          (s.pids 2 * K * V + (s.pids 1 * BK + idx.2.1.val) * V +
            (s.pids 0 * BV + idx.1.val))
        + s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.2.1.val + t * K) *
            ((s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V)
              - ∑ jk : Fin BK,
                  s.readMem HPrev
                      (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V +
                        (s.pids 0 * BV + idx.1.val)) *
                    s.readMem k
                      (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)) *
              s.readMem beta
                (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, stateOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem HOut (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BV, BK])).readMem HOut (offsetFn idx) =
    stateStepSpec s HPrev k v beta Bool.true t s_qk_h s_vo_h T K V BK BV idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp only [valueFn, stateStepSpec, stateOffset, betaVal, kVal, vVal,
    kIndex, vIndex, if_true, ite_true]
  ring

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_state_step_slice_scalarbeta_correct
    (HPrev k v beta HOut : RegionName)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s K V BK BV idx)) :
    ∀ idx : TileIndex [BV, BK],
      let outAddr := stateOffset s K V BK BV idx
      (exec (fused_recurrent_delta_state_step_slice_scalarbeta HPrev k v beta HOut
            t s_qk_h s_vo_h T K V BK BV) s).map (·.readMem HOut outAddr)
        = some (stateStepSpec s HPrev k v beta Bool.false
            t s_qk_h s_vo_h T K V BK BV idx) := by
  intro idx
  simp [exec, fused_recurrent_delta_state_step_slice_scalarbeta, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComparableDType.lt, stateOffset, vRowOffset,
        vIndex, kIndex, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV, BK] → Nat :=
    fun idx => s.pids 2 * K * V +
      (s.pids 1 * BK + idx.2.1.val) * V + (s.pids 0 * BV + idx.1.val)
  let valueFn : TileIndex [BV, BK] → ℝ :=
    fun idx =>
      s.readMem HPrev
          (s.pids 2 * K * V + (s.pids 1 * BK + idx.2.1.val) * V +
            (s.pids 0 * BV + idx.1.val))
        + s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.2.1.val + t * K) *
            ((s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V)
              - ∑ jk : Fin BK,
                  s.readMem HPrev
                      (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V +
                        (s.pids 0 * BV + idx.1.val)) *
                    s.readMem k
                      (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)) *
              s.readMem beta (s.pids 2 * T + t))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, stateOffset, kIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem HOut (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BV, BK])).readMem HOut (offsetFn idx) =
    stateStepSpec s HPrev k v beta Bool.false t s_qk_h s_vo_h T K V BK BV idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp only [valueFn, stateStepSpec, stateOffset, betaVal, kVal, vVal,
    kIndex, vIndex, Bool.false_eq_true, ite_false, if_false]
  ring

theorem fused_recurrent_delta_state_step_slice_headwise_compute_correct
    (HPrev k v beta HOut : RegionName)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s K V BK BV idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_state_step_slice_headwise HPrev k v beta
        HOut t s_qk_h s_vo_h T K V BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (HOut, stateOffset s K V BK BV idx))
      (expected := fun idx =>
        stateStepSpec s HPrev k v beta Bool.true t s_qk_h s_vo_h T K V BK BV idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_delta_state_step_slice_headwise,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := fused_recurrent_delta_state_step_slice_headwise_correct HPrev k v beta
    HOut t s_qk_h s_vo_h T K V BK BV s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem fused_recurrent_delta_state_step_slice_scalarbeta_compute_correct
    (HPrev k v beta HOut : RegionName)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s K V BK BV idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_state_step_slice_scalarbeta HPrev k v beta
        HOut t s_qk_h s_vo_h T K V BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (HOut, stateOffset s K V BK BV idx))
      (expected := fun idx =>
        stateStepSpec s HPrev k v beta Bool.false t s_qk_h s_vo_h T K V BK BV idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_delta_state_step_slice_scalarbeta,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := fused_recurrent_delta_state_step_slice_scalarbeta_correct HPrev k v beta
    HOut t s_qk_h s_vo_h T K V BK BV s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Output-readout step slice (the per-step `o_t` store)

One loop body's output store, isolated from the cross-step loop. The delta
rule reads out the **post-update** state, so this face consumes a materialized
post-update tile `HNext`: it loads `HNext` and the scaled `q_t` row, reduces
over the key axis and masked-stores the `[BV]` row into `o` at time row `t`.

`HNext` holding `deltaState(m+1)` is an *assumption* wherever this face is used
(the headline's `hNext`). The state-update face proves `deltaState(m+1)` about
the state *after executing* its own slice; nothing in this file sequences that
post-state into this face's pre-state. -/

def fused_recurrent_delta_output_step_slice
    (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  prev = tl.load(HNext + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[None, :]) * $(V) + (i_v * $(BV) + offs_v[:, None]))
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K)) * $(scale)
  _o = prev * b_q[None, :]
  _o = tl.sum(_o, axis=1)
  tl.store(o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (_o).to(o.dtype.element_ty), mask=mask_bv)
}

/-- The per-lane spec of one output store, reading the *post-update*
materialized state `HNext`:
`o_t[j_v] = Σ_{j_k} HNext[j_v,j_k] · scale·q_t[j_k]`. -/
noncomputable def outputStepSpec (s : BlockState) (HNext q : RegionName)
    (t s_qk_h K V BK BV : Nat) (scale : ℝ) (jv : Fin BV) : ℝ :=
  ∑ jk : Fin BK,
    s.readMem HNext (stateOffset s K V BK BV (jv, jk, PUnit.unit))
      * qVal s q s_qk_h K BK scale t jk

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_output_step_slice_correct
    (HNext q o : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outOffset s t s_vo_h B H V BV jv)) :
    ∀ jv : Fin BV,
      let outAddr := outOffset s t s_vo_h B H V BV jv
      (exec (fused_recurrent_delta_output_step_slice HNext q o
            t s_qk_h s_vo_h B H K V BK BV scale) s).map (·.readMem o outAddr)
        = some (if activeV s V BV jv then
            outputStepSpec s HNext q t s_qk_h K V BK BV scale jv
          else s.readMem o outAddr) := by
  intro jv
  simp [exec, fused_recurrent_delta_output_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, stateOffset, outOffset, activeV, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx => (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV +
      idx.1.val + t * V
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx =>
      ∑ jk : Fin BK,
        s.readMem HNext
            (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V +
              (s.pids 0 * BV + idx.1.val)) *
          (s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K) * scale)
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : outOffset s t s_vo_h B H V BV a = outOffset s t s_vo_h B H V BV b := by
      simpa [offsetFn, outOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem o (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem o (offsetFn (jv, PUnit.unit)) =
    if activeV s V BV jv then
      outputStepSpec s HNext q t s_qk_h K V BK BV scale jv
    else s.readMem o (offsetFn (jv, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jv, PUnit.unit)]
  by_cases hjv : s.pids 0 * BV + jv.val < V
  · simp only [P, hjv, if_true, activeV, vIndex]
    simp only [valueFn, outputStepSpec, stateOffset, kVal, qVal, kIndex, vIndex]
    apply Finset.sum_congr rfl
    intro jk _
    ring
  · simp only [P, activeV, vIndex, hjv, if_false, BlockState.setReg_readMem]

theorem fused_recurrent_delta_output_step_slice_compute_correct
    (HNext q o : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outOffset s t s_vo_h B H V BV jv)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_output_step_slice HNext q o
        t s_qk_h s_vo_h B H K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (o, outOffset s t s_vo_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        outputStepSpec s HNext q t s_qk_h K V BK BV scale jv) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_delta_output_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jv hActive
  have h := fused_recurrent_delta_output_step_slice_correct HNext q o
    t s_qk_h s_vo_h B H K V BK BV scale s hOutInj jv
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Genuine closed-form step realizations (the forward carry-fold)

The cross-step fold over `range(0, T)` threading `h` is the trusted boundary;
each *step face* is realized against the genuine recurrence `deltaState` /
its derived forms over the input regions `q, k, v, beta, h0`. -/

/-- **`v_new` step realizes the genuine delta.** If `HPrev` holds
`deltaState(m)`, the per-lane `v` writeback value equals `vNewClosed(m)`. -/
theorem vnewStepSpec_eq_vNewClosed
    (s : BlockState) (HPrev k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h T K V BK BV : Nat)
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem HPrev (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV m idx)
    (jv : Fin BV) :
    vnewStepSpec s HPrev k v m s_qk_h s_vo_h K V BK BV jv
      = vNewClosed s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV m jv := by
  unfold vnewStepSpec vNewClosed vMinusClosed vVal vRowOffset
  congr 1
  apply Finset.sum_congr rfl
  intro jk _
  rw [hPrev]

/-- **State carry-fold step (genuine).** If `HPrev` holds `deltaState(m)`,
then one loop body — `stateStepSpec`, i.e.
`HPrev + k_m ⊗ (β_m ⊙ (v_m − HPrev·k_m))` — produces exactly the genuine
`(m+1)`-step folded state `deltaState(m+1)`. -/
theorem stateStepSpec_eq_deltaState_succ
    (s : BlockState) (HPrev k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h T K V BK BV : Nat)
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem HPrev (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV m idx)
    (idx : TileIndex [BV, BK]) :
    stateStepSpec s HPrev k v beta IS_HEADWISE_BETA m s_qk_h s_vo_h T K V BK BV idx
      = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV (m + 1) idx := by
  have hsum :
      (∑ jk : Fin BK,
          s.readMem HPrev (stateOffset s K V BK BV (idx.1, jk, PUnit.unit))
            * kVal s k s_qk_h K BK m jk)
        = ∑ jk : Fin BK,
            deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
                s_qk_h s_vo_h T K V BK BV m (idx.1, jk, PUnit.unit)
              * kVal s k s_qk_h K BK m jk := by
    apply Finset.sum_congr rfl
    intro jk _
    rw [hPrev]
  rw [deltaState_succ]
  unfold stateStepSpec vNewClosed vMinusClosed
  rw [hPrev idx, hsum]

/-- **Output step realizes the genuine closed form.** If `HNext` holds the
post-update state `deltaState(m+1)`, the output reduction equals
`outputClosed(m)`. -/
theorem outputStepSpec_eq_outputClosed
    (s : BlockState) (HNext q k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ)
    (hNext : ∀ idx : TileIndex [BV, BK],
      s.readMem HNext (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV (m + 1) idx)
    (jv : Fin BV) :
    outputStepSpec s HNext q m s_qk_h K V BK BV scale jv
      = outputClosed s q k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV scale m jv := by
  unfold outputStepSpec outputClosed
  apply Finset.sum_congr rfl
  intro jk _
  rw [hNext]

/-- **Genuine `v_new` writeback.** One `v` writeback body, with the
materialized pre-update state `HPrev = deltaState(m)`, realizes the genuine
delta `vNewClosed(m)` (masked) into `v` at time row `m` — the kernel's
in-place update. -/
theorem fused_recurrent_delta_vnew_step_closed_form
    (HPrev k v beta h0 : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h T K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => vRowOffset s m s_vo_h V BV jv))
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem HPrev (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV m idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_vnew_step_slice HPrev k v
        m s_qk_h s_vo_h K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (v, vRowOffset s m s_vo_h V BV jv)))
      (expected := fun jv : Fin BV =>
        vNewClosed s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV m jv) := by
  have h := fused_recurrent_delta_vnew_step_slice_compute_correct HPrev k v
    m s_qk_h s_vo_h K V BK BV s hOutInj
  have hcong : (fun jv : Fin BV =>
      vnewStepSpec s HPrev k v m s_qk_h s_vo_h K V BK BV jv)
      = (fun jv : Fin BV =>
        vNewClosed s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV m jv) := by
    funext jv
    exact vnewStepSpec_eq_vNewClosed s HPrev k v beta h0 IS_HEADWISE_BETA
      USE_INITIAL_STATE m s_qk_h s_vo_h T K V BK BV hPrev jv
  rwa [hcong] at h

/-- **Genuine state carry-fold step.** One state-update body — the
`IS_HEADWISE_BETA` specialization selected by the flag, exactly as the
constexpr selects the compiled kernel — with the materialized pre-update
state `HPrev = deltaState(m)`, realizes the genuine `(m+1)`-step folded state
`deltaState(m+1)` into the state buffer. -/
theorem fused_recurrent_delta_state_step_closed_form
    (HPrev k v beta h0 HOut : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h T K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BV, BK] => stateOffset s K V BK BV idx))
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem HPrev (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV m idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_state_step_slice_headwise HPrev k v beta HOut
            m s_qk_h s_vo_h T K V BK BV
        else
          fused_recurrent_delta_state_step_slice_scalarbeta HPrev k v beta HOut
            m s_qk_h s_vo_h T K V BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (HOut, stateOffset s K V BK BV idx))
      (expected := fun idx =>
        deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV (m + 1) idx) := by
  have hcong : (fun idx : TileIndex [BV, BK] =>
      stateStepSpec s HPrev k v beta IS_HEADWISE_BETA m s_qk_h s_vo_h T K V BK BV idx)
      = (fun idx : TileIndex [BV, BK] =>
        deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV (m + 1) idx) := by
    funext idx
    exact stateStepSpec_eq_deltaState_succ s HPrev k v beta h0 IS_HEADWISE_BETA
      USE_INITIAL_STATE m s_qk_h s_vo_h T K V BK BV hPrev idx
  cases IS_HEADWISE_BETA
  · -- IS_HEADWISE_BETA = false
    have h := fused_recurrent_delta_state_step_slice_scalarbeta_compute_correct
      HPrev k v beta HOut m s_qk_h s_vo_h T K V BK BV s hOutInj
    rw [hcong] at h
    simpa using h
  · -- IS_HEADWISE_BETA = true
    have h := fused_recurrent_delta_state_step_slice_headwise_compute_correct
      HPrev k v beta HOut m s_qk_h s_vo_h T K V BK BV s hOutInj
    rw [hcong] at h
    simpa using h

/-- **Genuine output step.** One output body, with the materialized
post-update state `HNext = deltaState(m+1)`, realizes the genuine
`outputClosed(m)` (the key-axis readout of the post-update state) into the
output row. -/
theorem fused_recurrent_delta_output_step_closed_form
    (HNext q k v beta h0 o : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE : Bool)
    (m s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outOffset s m s_vo_h B H V BV jv))
    (hNext : ∀ idx : TileIndex [BV, BK],
      s.readMem HNext (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV (m + 1) idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_output_step_slice HNext q o
        m s_qk_h s_vo_h B H K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (o, outOffset s m s_vo_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        outputClosed s q k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV scale m jv) := by
  have h := fused_recurrent_delta_output_step_slice_compute_correct HNext q o
    m s_qk_h s_vo_h B H K V BK BV scale s hOutInj
  have hcong : (fun jv : Fin BV =>
      outputStepSpec s HNext q m s_qk_h K V BK BV scale jv)
      = (fun jv : Fin BV =>
        outputClosed s q k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV scale m jv) := by
    funext jv
    exact outputStepSpec_eq_outputClosed s HNext q k v beta h0 IS_HEADWISE_BETA
      USE_INITIAL_STATE m s_qk_h s_vo_h T K V BK BV scale hNext jv
  rwa [hcong] at h

/-! ## Backward per-step gradient faces (loop 1 / loop 2 bodies)

The backward kernel carries `d_h : [BK, BV]` through the reverse-time loop
and a recomputed `h : [BK, BV]` through the forward-time loop; both folds are
the trusted boundary (`fused_recurrent_hgrn` backward precedent), with the
carried state presented as materialized buffers `DHPrev` / `HPrev`. Each face
below realizes the genuine per-step gradient formula over those buffers and
the backward launch's input rows. NOTE: by the time the backward kernel runs,
region `v` holds the forward kernel's stored deltas `v_new` — the `vVal v`
reads below are the backward kernel's actual `b_v` inputs. -/

/-- The `dk`/`dq` store row address at time `t`, lane `j_k` — the kernel's
`(i_bh + i_v·B·H)·s_qk_h + i_k·BK + j_k + t·K` layout. -/
def dkRowOffset (s : BlockState) (t s_qk_h B H K BK : Nat) (jk : Fin BK) : Nat :=
  (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK + jk.val + t * K

/-- The headwise `dbeta` store row address at time `t`, lane `j_v` — the
kernel's `(i_bh + i_k·B·H + i_v·B·H·NK)·s_vo_h + j_v + t·V` layout (a plain
`tl.arange(0, BV)`, with no `i_v·BV` term, exactly as in the Python). -/
def dbetaRowOffset (s : BlockState) (t s_vo_h B H NK V BV : Nat)
    (jv : Fin BV) : Nat :=
  (s.pids 2 + s.pids 1 * B * H + s.pids 0 * B * H * NK) * s_vo_h + jv.val + t * V

/-- The scalar `dbeta` store address at time `t` — the kernel's
`(i_bh + i_v·B·H)·T + t` layout. -/
def dbetaScalarOffset (s : BlockState) (t T B H : Nat) : Nat :=
  (s.pids 2 + s.pids 0 * B * H) * T + t

def fused_recurrent_delta_bwd_dk_step_slice_headwise
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K)) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}

def fused_recurrent_delta_bwd_dk_step_slice_scalarbeta
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K)) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_beta = tl.load(beta + i_bh * $(T) + $(t))
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}

/-- The genuine per-lane `dk` step formula (loop 1, both beta modes):
`dk_t[j_k] = Σ_{j_v} (DHPrev + scale·q_t ⊗ do_t)[j_k,j_v] · (v_t[j_v]·β_t[j_v])`. -/
noncomputable def dkStepSpec (s : BlockState) (DHPrev q do_ v beta : RegionName)
    (IS_HEADWISE_BETA : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ)
    (jk : Fin BK) : ℝ :=
  ∑ jv : Fin BV,
    (s.readMem DHPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))
        + qVal s q s_qk_h K BK scale t jk * vVal s do_ s_vo_h V BV t jv)
      * (vVal s v s_vo_h V BV t jv *
          betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV t jv)

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dk_step_slice_headwise_correct
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk)) :
    ∀ jk : Fin BK,
      let outAddr := dkRowOffset s t s_qk_h B H K BK jk
      (exec (fused_recurrent_delta_bwd_dk_step_slice_headwise DHPrev q do_ v beta dk
            t s_qk_h s_vo_h B H T K V BK BV scale) s).map (·.readMem dk outAddr)
        = some (if activeK s K BK jk then
            dkStepSpec s DHPrev q do_ v beta Bool.true
              t s_qk_h s_vo_h T K V BK BV scale jk
          else s.readMem dk outAddr) := by
  intro jk
  simp [exec, fused_recurrent_delta_bwd_dk_step_slice_headwise, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, dhOffset, dkRowOffset, activeK, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK] → Nat :=
    fun idx => (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK +
      idx.1.val + t * K
  let valueFn : TileIndex [BK] → ℝ :=
    fun idx =>
      ∑ jv : Fin BV,
        (s.readMem DHPrev
            (s.pids 2 * K * V + (s.pids 1 * BK + idx.1.val) * V +
              (s.pids 0 * BV + jv.val))
          + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.1.val + t * K) *
              scale *
              s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)) *
        (s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V) *
          s.readMem beta (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V))
  let P : TileIndex [BK] → Prop := fun idx => s.pids 1 * BK + idx.1.val < K
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : dkRowOffset s t s_qk_h B H K BK a = dkRowOffset s t s_qk_h B H K BK b := by
      simpa [offsetFn, dkRowOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dk (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BK])).readMem dk (offsetFn (jk, PUnit.unit)) =
    if activeK s K BK jk then
      dkStepSpec s DHPrev q do_ v beta Bool.true t s_qk_h s_vo_h T K V BK BV scale jk
    else s.readMem dk (offsetFn (jk, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jk, PUnit.unit)]
  by_cases hjk : s.pids 1 * BK + jk.val < K
  · simp only [P, hjk, if_true, activeK, kIndex]
    simp only [valueFn, dkStepSpec, dhOffset, betaVal, kVal, qVal, vVal,
      kIndex, vIndex, if_true, ite_true]
    apply Finset.sum_congr rfl
    intro jv _
    ring
  · simp only [P, activeK, kIndex, hjk, if_false, BlockState.setReg_readMem]

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dk_step_slice_scalarbeta_correct
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk)) :
    ∀ jk : Fin BK,
      let outAddr := dkRowOffset s t s_qk_h B H K BK jk
      (exec (fused_recurrent_delta_bwd_dk_step_slice_scalarbeta DHPrev q do_ v beta dk
            t s_qk_h s_vo_h B H T K V BK BV scale) s).map (·.readMem dk outAddr)
        = some (if activeK s K BK jk then
            dkStepSpec s DHPrev q do_ v beta Bool.false
              t s_qk_h s_vo_h T K V BK BV scale jk
          else s.readMem dk outAddr) := by
  intro jk
  simp [exec, fused_recurrent_delta_bwd_dk_step_slice_scalarbeta, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, dhOffset, dkRowOffset, activeK, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK] → Nat :=
    fun idx => (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK +
      idx.1.val + t * K
  let valueFn : TileIndex [BK] → ℝ :=
    fun idx =>
      ∑ jv : Fin BV,
        (s.readMem DHPrev
            (s.pids 2 * K * V + (s.pids 1 * BK + idx.1.val) * V +
              (s.pids 0 * BV + jv.val))
          + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.1.val + t * K) *
              scale *
              s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)) *
        (s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V) *
          s.readMem beta (s.pids 2 * T + t))
  let P : TileIndex [BK] → Prop := fun idx => s.pids 1 * BK + idx.1.val < K
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : dkRowOffset s t s_qk_h B H K BK a = dkRowOffset s t s_qk_h B H K BK b := by
      simpa [offsetFn, dkRowOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dk (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BK])).readMem dk (offsetFn (jk, PUnit.unit)) =
    if activeK s K BK jk then
      dkStepSpec s DHPrev q do_ v beta Bool.false t s_qk_h s_vo_h T K V BK BV scale jk
    else s.readMem dk (offsetFn (jk, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jk, PUnit.unit)]
  by_cases hjk : s.pids 1 * BK + jk.val < K
  · simp only [P, hjk, if_true, activeK, kIndex]
    simp only [valueFn, dkStepSpec, dhOffset, betaVal, kVal, qVal, vVal,
      kIndex, vIndex, Bool.false_eq_true, ite_false, if_false]
    apply Finset.sum_congr rfl
    intro jv _
    ring
  · simp only [P, activeK, kIndex, hjk, if_false, BlockState.setReg_readMem]

/-- **Genuine `dk` step (loop 1).** The `IS_HEADWISE_BETA` specialization
selected by the flag realizes `dkStepSpec` (masked) into `dk` at time row `t`. -/
theorem fused_recurrent_delta_bwd_dk_step_slice_compute_correct
    (DHPrev q do_ v beta dk : RegionName) (IS_HEADWISE_BETA : Bool)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dk_step_slice_headwise DHPrev q do_ v beta dk
            t s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dk_step_slice_scalarbeta DHPrev q do_ v beta dk
            t s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s K BK jk)
        (fun jk => (dk, dkRowOffset s t s_qk_h B H K BK jk)))
      (expected := fun jk : Fin BK =>
        dkStepSpec s DHPrev q do_ v beta IS_HEADWISE_BETA
          t s_qk_h s_vo_h T K V BK BV scale jk) := by
  cases IS_HEADWISE_BETA
  · -- IS_HEADWISE_BETA = false
    simp only [Bool.false_eq_true, if_false]
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [fused_recurrent_delta_bwd_dk_step_slice_scalarbeta,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro jk hActive
    have h := fused_recurrent_delta_bwd_dk_step_slice_scalarbeta_correct DHPrev q
      do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale s hOutInj jk
    rw [hExec] at h
    simpa [hActive] using Option.some.inj h
  · -- IS_HEADWISE_BETA = true
    simp only [if_true]
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [fused_recurrent_delta_bwd_dk_step_slice_headwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro jk hActive
    have h := fused_recurrent_delta_bwd_dk_step_slice_headwise_correct DHPrev q
      do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale s hOutInj jk
    rw [hExec] at h
    simpa [hActive] using Option.some.inj h

def fused_recurrent_delta_bwd_dv_step_slice_headwise
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K)) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_v = d_v * b_beta
  tl.store(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (d_v).to(dv.dtype.element_ty), mask=mask_bv)
}

def fused_recurrent_delta_bwd_dv_step_slice_scalarbeta
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K)) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_beta = tl.load(beta + i_bh * $(T) + $(t))
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_v = d_v * b_beta
  tl.store(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (d_v).to(dv.dtype.element_ty), mask=mask_bv)
}

/-- The genuine per-lane `dv` step formula (loop 1, both beta modes):
`dv_t[j_v] = (Σ_{j_k} (DHPrev + scale·q_t ⊗ do_t)[j_k,j_v] · k_t[j_k]) · β_t[j_v]`. -/
noncomputable def dvStepSpec (s : BlockState) (DHPrev q do_ k beta : RegionName)
    (IS_HEADWISE_BETA : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ)
    (jv : Fin BV) : ℝ :=
  (∑ jk : Fin BK,
      (s.readMem DHPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))
          + qVal s q s_qk_h K BK scale t jk * vVal s do_ s_vo_h V BV t jv)
        * kVal s k s_qk_h K BK t jk)
    * betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV t jv

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dv_step_slice_headwise_correct
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outOffset s t s_vo_h B H V BV jv)) :
    ∀ jv : Fin BV,
      let outAddr := outOffset s t s_vo_h B H V BV jv
      (exec (fused_recurrent_delta_bwd_dv_step_slice_headwise DHPrev q do_ k beta dv
            t s_qk_h s_vo_h B H T K V BK BV scale) s).map (·.readMem dv outAddr)
        = some (if activeV s V BV jv then
            dvStepSpec s DHPrev q do_ k beta Bool.true
              t s_qk_h s_vo_h T K V BK BV scale jv
          else s.readMem dv outAddr) := by
  intro jv
  simp [exec, fused_recurrent_delta_bwd_dv_step_slice_headwise, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, dhOffset, outOffset, activeV, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx => (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV +
      idx.1.val + t * V
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx =>
      (∑ jk : Fin BK,
        (s.readMem DHPrev
            (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V +
              (s.pids 0 * BV + idx.1.val))
          + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K) *
              scale *
              s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V)) *
        s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)) *
      s.readMem beta (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V)
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : outOffset s t s_vo_h B H V BV a = outOffset s t s_vo_h B H V BV b := by
      simpa [offsetFn, outOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dv (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem dv (offsetFn (jv, PUnit.unit)) =
    if activeV s V BV jv then
      dvStepSpec s DHPrev q do_ k beta Bool.true t s_qk_h s_vo_h T K V BK BV scale jv
    else s.readMem dv (offsetFn (jv, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jv, PUnit.unit)]
  by_cases hjv : s.pids 0 * BV + jv.val < V
  · simp only [P, hjv, if_true, activeV, vIndex]
    simp only [valueFn, dvStepSpec, dhOffset, betaVal, kVal, qVal, vVal,
      kIndex, vIndex, if_true, ite_true]
    congr 1
    apply Finset.sum_congr rfl
    intro jk _
    ring
  · simp only [P, activeV, vIndex, hjv, if_false, BlockState.setReg_readMem]

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dv_step_slice_scalarbeta_correct
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outOffset s t s_vo_h B H V BV jv)) :
    ∀ jv : Fin BV,
      let outAddr := outOffset s t s_vo_h B H V BV jv
      (exec (fused_recurrent_delta_bwd_dv_step_slice_scalarbeta DHPrev q do_ k beta dv
            t s_qk_h s_vo_h B H T K V BK BV scale) s).map (·.readMem dv outAddr)
        = some (if activeV s V BV jv then
            dvStepSpec s DHPrev q do_ k beta Bool.false
              t s_qk_h s_vo_h T K V BK BV scale jv
          else s.readMem dv outAddr) := by
  intro jv
  simp [exec, fused_recurrent_delta_bwd_dv_step_slice_scalarbeta, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, dhOffset, outOffset, activeV, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx => (s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV +
      idx.1.val + t * V
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx =>
      (∑ jk : Fin BK,
        (s.readMem DHPrev
            (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V +
              (s.pids 0 * BV + idx.1.val))
          + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K) *
              scale *
              s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V)) *
        s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)) *
      s.readMem beta (s.pids 2 * T + t)
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : outOffset s t s_vo_h B H V BV a = outOffset s t s_vo_h B H V BV b := by
      simpa [offsetFn, outOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dv (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem dv (offsetFn (jv, PUnit.unit)) =
    if activeV s V BV jv then
      dvStepSpec s DHPrev q do_ k beta Bool.false t s_qk_h s_vo_h T K V BK BV scale jv
    else s.readMem dv (offsetFn (jv, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jv, PUnit.unit)]
  by_cases hjv : s.pids 0 * BV + jv.val < V
  · simp only [P, hjv, if_true, activeV, vIndex]
    simp only [valueFn, dvStepSpec, dhOffset, betaVal, kVal, qVal, vVal,
      kIndex, vIndex, Bool.false_eq_true, ite_false, if_false]
    congr 1
    apply Finset.sum_congr rfl
    intro jk _
    ring
  · simp only [P, activeV, vIndex, hjv, if_false, BlockState.setReg_readMem]

/-- **Genuine `dv` step (loop 1).** The `IS_HEADWISE_BETA` specialization
selected by the flag realizes `dvStepSpec` (masked) into `dv` at time row `t`. -/
theorem fused_recurrent_delta_bwd_dv_step_slice_compute_correct
    (DHPrev q do_ k beta dv : RegionName) (IS_HEADWISE_BETA : Bool)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => outOffset s t s_vo_h B H V BV jv)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dv_step_slice_headwise DHPrev q do_ k beta dv
            t s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dv_step_slice_scalarbeta DHPrev q do_ k beta dv
            t s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (dv, outOffset s t s_vo_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        dvStepSpec s DHPrev q do_ k beta IS_HEADWISE_BETA
          t s_qk_h s_vo_h T K V BK BV scale jv) := by
  cases IS_HEADWISE_BETA
  · -- IS_HEADWISE_BETA = false
    simp only [Bool.false_eq_true, if_false]
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [fused_recurrent_delta_bwd_dv_step_slice_scalarbeta,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro jv hActive
    have h := fused_recurrent_delta_bwd_dv_step_slice_scalarbeta_correct DHPrev q
      do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale s hOutInj jv
    rw [hExec] at h
    simpa [hActive] using Option.some.inj h
  · -- IS_HEADWISE_BETA = true
    simp only [if_true]
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [fused_recurrent_delta_bwd_dv_step_slice_headwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro jv hActive
    have h := fused_recurrent_delta_bwd_dv_step_slice_headwise_correct DHPrev q
      do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale s hOutInj jv
    rw [hExec] at h
    simpa [hActive] using Option.some.inj h

def fused_recurrent_delta_bwd_dbeta_step_slice_headwise
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K)) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_beta = d_v * b_v
  tl.store(dbeta + (i_bh + i_k * $(B) * $(H) + i_v * $(B) * $(H) * $(NK)) * $(s_vo_h) +
    offs_v + $(t) * $(V), (d_beta).to(dbeta.dtype.element_ty), mask=mask_bv)
}

def fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K)) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_beta = tl.sum(d_v * b_v)
  tl.store(dbeta + (i_bh + i_v * $(B) * $(H)) * $(T) + $(t),
    (d_beta).to(dbeta.dtype.element_ty))
}

/-- The genuine per-lane headwise `dbeta` step formula (loop 1):
`dbeta_t[j_v] = (Σ_{j_k} (DHPrev + scale·q_t ⊗ do_t)[j_k,j_v] · k_t[j_k]) · v_t[j_v]`
(the *pre-beta* `d_v`, as in the Python). -/
noncomputable def dbetaStepSpec (s : BlockState) (DHPrev q do_ k v : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) (jv : Fin BV) : ℝ :=
  (∑ jk : Fin BK,
      (s.readMem DHPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))
          + qVal s q s_qk_h K BK scale t jk * vVal s do_ s_vo_h V BV t jv)
        * kVal s k s_qk_h K BK t jk)
    * vVal s v s_vo_h V BV t jv

/-- The genuine scalar `dbeta` step formula (loop 1):
`dbeta_t = Σ_{j_v} d_v[j_v] · v_t[j_v]` (the full `tl.sum` reduction). -/
noncomputable def dbetaScalarStepSpec (s : BlockState) (DHPrev q do_ k v : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) : ℝ :=
  ∑ jv : Fin BV,
    dbetaStepSpec s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale jv

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dbeta_step_slice_headwise_correct
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => dbetaRowOffset s t s_vo_h B H NK V BV jv)) :
    ∀ jv : Fin BV,
      let outAddr := dbetaRowOffset s t s_vo_h B H NK V BV jv
      (exec (fused_recurrent_delta_bwd_dbeta_step_slice_headwise DHPrev q do_ k v dbeta
            t s_qk_h s_vo_h NK B H T K V BK BV scale) s).map (·.readMem dbeta outAddr)
        = some (if activeV s V BV jv then
            dbetaStepSpec s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale jv
          else s.readMem dbeta outAddr) := by
  intro jv
  simp [exec, fused_recurrent_delta_bwd_dbeta_step_slice_headwise, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, dhOffset, dbetaRowOffset, activeV, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BV] → Nat :=
    fun idx => (s.pids 2 + s.pids 1 * B * H + s.pids 0 * B * H * NK) * s_vo_h +
      idx.1.val + t * V
  let valueFn : TileIndex [BV] → ℝ :=
    fun idx =>
      (∑ jk : Fin BK,
        (s.readMem DHPrev
            (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V +
              (s.pids 0 * BV + idx.1.val))
          + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K) *
              scale *
              s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V)) *
        s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)) *
      s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + idx.1.val + t * V)
  let P : TileIndex [BV] → Prop := fun idx => s.pids 0 * BV + idx.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : dbetaRowOffset s t s_vo_h B H NK V BV a
        = dbetaRowOffset s t s_vo_h B H NK V BV b := by
      simpa [offsetFn, dbetaRowOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dbeta (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BV])).readMem dbeta (offsetFn (jv, PUnit.unit)) =
    if activeV s V BV jv then
      dbetaStepSpec s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale jv
    else s.readMem dbeta (offsetFn (jv, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jv, PUnit.unit)]
  by_cases hjv : s.pids 0 * BV + jv.val < V
  · simp only [P, hjv, if_true, activeV, vIndex]
    simp only [valueFn, dbetaStepSpec, dhOffset, kVal, qVal, vVal,
      kIndex, vIndex]
    congr 1
    apply Finset.sum_congr rfl
    intro jk _
    ring
  · simp only [P, activeV, vIndex, hjv, if_false, BlockState.setReg_readMem]

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta_correct
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState) :
    (exec (fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta DHPrev q do_ k v dbeta
          t s_qk_h s_vo_h B H T K V BK BV scale) s).map
        (·.readMem dbeta (dbetaScalarOffset s t T B H))
      = some (dbetaScalarStepSpec s DHPrev q do_ k v
          t s_qk_h s_vo_h K V BK BV scale) := by
  simp [exec, fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, dhOffset, dbetaScalarOffset, vIndex, kIndex,
        TileShape.dropInsertedIndex, BlockState.writeMem_readMem]
  -- The remaining `insertAxisIndex` lane atoms are `rfl`-reducible; `change`
  -- normalizes them by definitional equality.
  change (∑ jv : Fin BV,
      (∑ jk : Fin BK,
          (s.readMem DHPrev
              (s.pids 2 * K * V + (s.pids 1 * BK + jk.val) * V +
                (s.pids 0 * BV + jv.val))
            + s.readMem q (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K) *
                scale *
                s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)) *
          s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + jk.val + t * K)) *
      s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V))
    = dbetaScalarStepSpec s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale
  simp only [dbetaScalarStepSpec, dbetaStepSpec, dhOffset, kVal, qVal, vVal,
    kIndex, vIndex]
  apply Finset.sum_congr rfl
  intro jv _
  congr 1
  apply Finset.sum_congr rfl
  intro jk _
  ring

/-- **Genuine headwise `dbeta` step (loop 1)** — realizes `dbetaStepSpec`
(masked) into the headwise `dbeta` row at time `t`. -/
theorem fused_recurrent_delta_bwd_dbeta_step_slice_headwise_compute_correct
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jv : Fin BV => dbetaRowOffset s t s_vo_h B H NK V BV jv)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_bwd_dbeta_step_slice_headwise DHPrev q do_ k v
        dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (dbeta, dbetaRowOffset s t s_vo_h B H NK V BV jv)))
      (expected := fun jv : Fin BV =>
        dbetaStepSpec s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale jv) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_delta_bwd_dbeta_step_slice_headwise,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jv hActive
  have h := fused_recurrent_delta_bwd_dbeta_step_slice_headwise_correct DHPrev q
    do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale s hOutInj jv
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Genuine scalar `dbeta` step (loop 1)** — realizes the full-reduction
`dbetaScalarStepSpec` into the scalar `dbeta` cell at time `t`. -/
theorem fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta_compute_correct
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta DHPrev q do_ k v
        dbeta t s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := fun _ : PUnit => some (dbeta, dbetaScalarOffset s t T B H))
      (expected := fun _ =>
        dbetaScalarStepSpec s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  have h := fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta_correct DHPrev q
    do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale s
  rw [hExec] at h
  exact Option.some.inj h

/-! ### Loop-2 faces: the `dk` correction and `dq` -/

def fused_recurrent_delta_bwd_dk_correction_step_slice
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  d_k = tl.load(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K))
  d_v = tl.load(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V))
  h = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  d_k -= tl.sum(d_v[None, :] * h, axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}

/-- The genuine per-lane loop-2 `dk` correction formula: the loop-1 `dk` row
minus the readout of the recomputed *pre-update* state,
`dk_t[j_k] − Σ_{j_v} dv_t[j_v] · HPrev[j_k,j_v]`. Both the `dk` and `dv` rows
are the loop-1 stores (this face rewrites `dk` at the same address it reads,
after the read — the final memory value of the `dk` row). -/
noncomputable def dkCorrStepSpec (s : BlockState) (HPrev dv dk : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) (jk : Fin BK) : ℝ :=
  s.readMem dk (dkRowOffset s t s_qk_h B H K BK jk)
    - ∑ jv : Fin BV,
        s.readMem dv (outOffset s t s_vo_h B H V BV jv)
          * s.readMem HPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dk_correction_step_slice_correct
    (HPrev dv dk : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk)) :
    ∀ jk : Fin BK,
      let outAddr := dkRowOffset s t s_qk_h B H K BK jk
      (exec (fused_recurrent_delta_bwd_dk_correction_step_slice HPrev dv dk
            t s_qk_h s_vo_h B H K V BK BV) s).map (·.readMem dk outAddr)
        = some (if activeK s K BK jk then
            dkCorrStepSpec s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV jk
          else s.readMem dk outAddr) := by
  intro jk
  simp [exec, fused_recurrent_delta_bwd_dk_correction_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComparableDType.lt, dhOffset, dkRowOffset, outOffset,
        activeK, vIndex, kIndex, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK] → Nat :=
    fun idx => (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK +
      idx.1.val + t * K
  let valueFn : TileIndex [BK] → ℝ :=
    fun idx =>
      s.readMem dk ((s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK +
          idx.1.val + t * K)
        - ∑ jv : Fin BV,
            s.readMem dv ((s.pids 2 + s.pids 1 * B * H) * s_vo_h + s.pids 0 * BV +
                jv.val + t * V) *
              s.readMem HPrev
                (s.pids 2 * K * V + (s.pids 1 * BK + idx.1.val) * V +
                  (s.pids 0 * BV + jv.val))
  let P : TileIndex [BK] → Prop := fun idx => s.pids 1 * BK + idx.1.val < K
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : dkRowOffset s t s_qk_h B H K BK a = dkRowOffset s t s_qk_h B H K BK b := by
      simpa [offsetFn, dkRowOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dk (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BK])).readMem dk (offsetFn (jk, PUnit.unit)) =
    if activeK s K BK jk then
      dkCorrStepSpec s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV jk
    else s.readMem dk (offsetFn (jk, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jk, PUnit.unit)]
  by_cases hjk : s.pids 1 * BK + jk.val < K
  · simp only [P, hjk, if_true, activeK, kIndex]
    simp only [valueFn, dkCorrStepSpec, dhOffset, dkRowOffset, outOffset,
      kIndex, vIndex]
  · simp only [P, activeK, kIndex, hjk, if_false, BlockState.setReg_readMem]

/-- **Genuine loop-2 `dk` correction step** — realizes `dkCorrStepSpec`
(masked) into `dk` at time row `t` (the in-place fixup; the row is read
before it is written). -/
theorem fused_recurrent_delta_bwd_dk_correction_step_slice_compute_correct
    (HPrev dv dk : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_bwd_dk_correction_step_slice HPrev dv dk
        t s_qk_h s_vo_h B H K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s K BK jk)
        (fun jk => (dk, dkRowOffset s t s_qk_h B H K BK jk)))
      (expected := fun jk : Fin BK =>
        dkCorrStepSpec s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV jk) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_delta_bwd_dk_correction_step_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro jk hActive
  have h := fused_recurrent_delta_bwd_dk_correction_step_slice_correct HPrev dv dk
    t s_qk_h s_vo_h B H K V BK BV s hOutInj jk
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ### Sequencing the loop-2 `dk` correction onto the loop-1 stores

`dkCorrStepSpec` above is stated *relative to whatever the `dk`/`dv` rows hold*
when the correction body runs. In the Python kernel those rows hold exactly what
loop 1 stored: after the `tl.debug_barrier()` the pointers `p_dk`/`p_dv` are
re-initialized to row `0` of the same `dk`/`dv` layouts loop 1 wrote (loop 1
walked them backwards from row `T-1`), so at time row `t` loop 2 reloads loop 1's
`d_k`/`d_v` cells — `dkRowOffset s t` and `outOffset s t` in both loops. The
face below states that sequencing explicitly instead of leaving the two reads
unconstrained: given loop 1's stored values, the final memory value of the `dk`
row is a formula over the backward inputs and the two materialized carries. -/

/-- **The sequenced loop-2 `dk` correction**: loop 1's `dk` row minus the
recomputed-state readout of loop 1's `dv` row,
`dkStepSpec_t[j_k] − Σ_{j_v} dvStepSpec_t[j_v] · HRec[j_k,j_v]`. A formula over
the backward-launch inputs and the two materialized carries (`DHPrev`, `HRec`) —
no unconstrained read of the kernel's own output rows. -/
noncomputable def dkCorrClosed (s : BlockState)
    (DHPrev HRec q do_ k v beta : RegionName) (IS_HEADWISE_BETA : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ) (jk : Fin BK) : ℝ :=
  dkStepSpec s DHPrev q do_ v beta IS_HEADWISE_BETA
      t s_qk_h s_vo_h T K V BK BV scale jk
    - ∑ jv : Fin BV,
        dvStepSpec s DHPrev q do_ k beta IS_HEADWISE_BETA
            t s_qk_h s_vo_h T K V BK BV scale jv
          * s.readMem HRec (dhOffset s K V BK BV (jk, jv, PUnit.unit))

/-- **Genuine sequenced loop-2 `dk` correction step.** Under the two
loop-1-stored-value hypotheses (Python's two-loop structure: loop 2 reloads the
rows loop 1 wrote — see the section note), the correction body realizes
`dkCorrClosed`, i.e. the *final* memory value of the `dk` row expressed through
the loop-1 `dk`/`dv` step formulas rather than through unconstrained reads. -/
theorem fused_recurrent_delta_bwd_dk_correction_step_sequenced
    (DHPrev HRec q do_ k v beta dk dv : RegionName) (IS_HEADWISE_BETA : Bool)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk))
    -- forced by loop 1's `tl.store(p_dk, d_k, mask=mask_bk)` at this very row
    (hDk : ∀ jk : Fin BK,
      s.readMem dk (dkRowOffset s t s_qk_h B H K BK jk)
        = dkStepSpec s DHPrev q do_ v beta IS_HEADWISE_BETA
            t s_qk_h s_vo_h T K V BK BV scale jk)
    -- forced by loop 1's `tl.store(p_dv, d_v, mask=mask_bv)` at this very row
    (hDv : ∀ jv : Fin BV,
      s.readMem dv (outOffset s t s_vo_h B H V BV jv)
        = dvStepSpec s DHPrev q do_ k beta IS_HEADWISE_BETA
            t s_qk_h s_vo_h T K V BK BV scale jv) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_bwd_dk_correction_step_slice HRec dv dk
        t s_qk_h s_vo_h B H K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s K BK jk)
        (fun jk => (dk, dkRowOffset s t s_qk_h B H K BK jk)))
      (expected := fun jk : Fin BK =>
        dkCorrClosed s DHPrev HRec q do_ k v beta IS_HEADWISE_BETA
          t s_qk_h s_vo_h T K V BK BV scale jk) := by
  have h := fused_recurrent_delta_bwd_dk_correction_step_slice_compute_correct
    HRec dv dk t s_qk_h s_vo_h B H K V BK BV s hOutInj
  have hcong : (fun jk : Fin BK =>
      dkCorrStepSpec s HRec dv dk t s_qk_h s_vo_h B H K V BK BV jk)
      = (fun jk : Fin BK =>
        dkCorrClosed s DHPrev HRec q do_ k v beta IS_HEADWISE_BETA
          t s_qk_h s_vo_h T K V BK BV scale jk) := by
    funext jk
    unfold dkCorrStepSpec dkCorrClosed
    rw [hDk jk]
    congr 1
    apply Finset.sum_congr rfl
    intro jv _
    rw [hDv jv]
  rwa [hcong] at h

def fused_recurrent_delta_bwd_dq_step_slice_headwise
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_v *= b_beta
  h = prev + b_k[:, None] * b_v[None, :]
  _d_q = h * b_do[None, :]
  d_q = tl.sum(_d_q, axis=1) * $(scale)
  tl.store(dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_q).to(dq.dtype.element_ty), mask=mask_bk)
}

def fused_recurrent_delta_bwd_dq_step_slice_scalarbeta
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]))
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K))
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V))
  b_beta = tl.load(beta + i_bh * $(T) + $(t))
  b_v *= b_beta
  h = prev + b_k[:, None] * b_v[None, :]
  _d_q = h * b_do[None, :]
  d_q = tl.sum(_d_q, axis=1) * $(scale)
  tl.store(dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_q).to(dq.dtype.element_ty), mask=mask_bk)
}

/-- The genuine per-lane `dq` step formula (loop 2, both beta modes), reading
the recomputed *post-update* state built from the materialized `HPrev`:
`dq_t[j_k] = (Σ_{j_v} (HPrev[j_k,j_v] + k_t[j_k]·(v_t[j_v]·β_t[j_v])) · do_t[j_v]) · scale`
(recall `v` here holds the forward pass's stored deltas `v_new`). -/
noncomputable def dqStepSpec (s : BlockState) (HPrev k v beta do_ : RegionName)
    (IS_HEADWISE_BETA : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ)
    (jk : Fin BK) : ℝ :=
  (∑ jv : Fin BV,
      (s.readMem HPrev (dhOffset s K V BK BV (jk, jv, PUnit.unit))
          + kVal s k s_qk_h K BK t jk *
              (vVal s v s_vo_h V BV t jv *
                betaVal s beta IS_HEADWISE_BETA s_vo_h T V BV t jv))
        * vVal s do_ s_vo_h V BV t jv)
    * scale

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dq_step_slice_headwise_correct
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk)) :
    ∀ jk : Fin BK,
      let outAddr := dkRowOffset s t s_qk_h B H K BK jk
      (exec (fused_recurrent_delta_bwd_dq_step_slice_headwise HPrev k v beta do_ dq
            t s_qk_h s_vo_h B H T K V BK BV scale) s).map (·.readMem dq outAddr)
        = some (if activeK s K BK jk then
            dqStepSpec s HPrev k v beta do_ Bool.true
              t s_qk_h s_vo_h T K V BK BV scale jk
          else s.readMem dq outAddr) := by
  intro jk
  simp [exec, fused_recurrent_delta_bwd_dq_step_slice_headwise, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, dhOffset, dkRowOffset, activeK, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK] → Nat :=
    fun idx => (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK +
      idx.1.val + t * K
  let valueFn : TileIndex [BK] → ℝ :=
    fun idx =>
      (∑ jv : Fin BV,
        (s.readMem HPrev
            (s.pids 2 * K * V + (s.pids 1 * BK + idx.1.val) * V +
              (s.pids 0 * BV + jv.val))
          + s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.1.val + t * K) *
              (s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V) *
                s.readMem beta (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V))) *
        s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)) * scale
  let P : TileIndex [BK] → Prop := fun idx => s.pids 1 * BK + idx.1.val < K
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : dkRowOffset s t s_qk_h B H K BK a = dkRowOffset s t s_qk_h B H K BK b := by
      simpa [offsetFn, dkRowOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dq (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BK])).readMem dq (offsetFn (jk, PUnit.unit)) =
    if activeK s K BK jk then
      dqStepSpec s HPrev k v beta do_ Bool.true t s_qk_h s_vo_h T K V BK BV scale jk
    else s.readMem dq (offsetFn (jk, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jk, PUnit.unit)]
  by_cases hjk : s.pids 1 * BK + jk.val < K
  · simp only [P, hjk, if_true, activeK, kIndex]
    simp only [valueFn, dqStepSpec, dhOffset, betaVal, kVal, vVal,
      kIndex, vIndex, if_true, ite_true]
  · simp only [P, activeK, kIndex, hjk, if_false, BlockState.setReg_readMem]

set_option maxHeartbeats 1000000 in
theorem fused_recurrent_delta_bwd_dq_step_slice_scalarbeta_correct
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk)) :
    ∀ jk : Fin BK,
      let outAddr := dkRowOffset s t s_qk_h B H K BK jk
      (exec (fused_recurrent_delta_bwd_dq_step_slice_scalarbeta HPrev k v beta do_ dq
            t s_qk_h s_vo_h B H T K V BK BV scale) s).map (·.readMem dq outAddr)
        = some (if activeK s K BK jk then
            dqStepSpec s HPrev k v beta do_ Bool.false
              t s_qk_h s_vo_h T K V BK BV scale jk
          else s.readMem dq outAddr) := by
  intro jk
  simp [exec, fused_recurrent_delta_bwd_dq_step_slice_scalarbeta, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, dhOffset, dkRowOffset, activeK, vIndex, kIndex,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK] → Nat :=
    fun idx => (s.pids 2 + s.pids 0 * B * H) * s_qk_h + s.pids 1 * BK +
      idx.1.val + t * K
  let valueFn : TileIndex [BK] → ℝ :=
    fun idx =>
      (∑ jv : Fin BV,
        (s.readMem HPrev
            (s.pids 2 * K * V + (s.pids 1 * BK + idx.1.val) * V +
              (s.pids 0 * BV + jv.val))
          + s.readMem k (s.pids 2 * s_qk_h + s.pids 1 * BK + idx.1.val + t * K) *
              (s.readMem v (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V) *
                s.readMem beta (s.pids 2 * T + t))) *
        s.readMem do_ (s.pids 2 * s_vo_h + s.pids 0 * BV + jv.val + t * V)) * scale
  let P : TileIndex [BK] → Prop := fun idx => s.pids 1 * BK + idx.1.val < K
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have : dkRowOffset s t s_qk_h B H K BK a = dkRowOffset s t s_qk_h B H K BK b := by
      simpa [offsetFn, dkRowOffset] using hab
    obtain rfl : a = b := hOutInj this
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem dq (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BK])).readMem dq (offsetFn (jk, PUnit.unit)) =
    if activeK s K BK jk then
      dqStepSpec s HPrev k v beta do_ Bool.false t s_qk_h s_vo_h T K V BK BV scale jk
    else s.readMem dq (offsetFn (jk, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (jk, PUnit.unit)]
  by_cases hjk : s.pids 1 * BK + jk.val < K
  · simp only [P, hjk, if_true, activeK, kIndex]
    simp only [valueFn, dqStepSpec, dhOffset, betaVal, kVal, vVal,
      kIndex, vIndex, Bool.false_eq_true, ite_false, if_false]
  · simp only [P, activeK, kIndex, hjk, if_false, BlockState.setReg_readMem]

/-- **Genuine `dq` step (loop 2).** The `IS_HEADWISE_BETA` specialization
selected by the flag realizes `dqStepSpec` (masked) into `dq` at time row `t`. -/
theorem fused_recurrent_delta_bwd_dq_step_slice_compute_correct
    (HPrev k v beta do_ dq : RegionName) (IS_HEADWISE_BETA : Bool)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dq_step_slice_headwise HPrev k v beta do_ dq
            t s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dq_step_slice_scalarbeta HPrev k v beta do_ dq
            t s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s K BK jk)
        (fun jk => (dq, dkRowOffset s t s_qk_h B H K BK jk)))
      (expected := fun jk : Fin BK =>
        dqStepSpec s HPrev k v beta do_ IS_HEADWISE_BETA
          t s_qk_h s_vo_h T K V BK BV scale jk) := by
  cases IS_HEADWISE_BETA
  · -- IS_HEADWISE_BETA = false
    simp only [Bool.false_eq_true, if_false]
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [fused_recurrent_delta_bwd_dq_step_slice_scalarbeta,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro jk hActive
    have h := fused_recurrent_delta_bwd_dq_step_slice_scalarbeta_correct HPrev k v
      beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale s hOutInj jk
    rw [hExec] at h
    simpa [hActive] using Option.some.inj h
  · -- IS_HEADWISE_BETA = true
    simp only [if_true]
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [fused_recurrent_delta_bwd_dq_step_slice_headwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro jk hActive
    have h := fused_recurrent_delta_bwd_dq_step_slice_headwise_correct HPrev k v
      beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale s hOutInj jk
    rw [hExec] at h
    simpa [hActive] using Option.some.inj h

/-! ## Offset-injectivity side conditions (dimension-general)

The flattened state address `i_bh·K·V + j_k·V + j_v` is injective on the
`[BV, BK]` (and `[BK, BV]`) tile whenever the tile fits the logical extents
(`BV ≤ V`, `BK ≤ K`): `j_v` occupies the low `V`-digit and `j_k·V` the next, a
faithful mixed-radix encoding. All per-time row addresses are injective in
their lane unconditionally (the lane is an additive term). The Python
regression (`K = BK = 16`, `V = 32`, `BV ∈ {8, 32}`) satisfies them. -/

/-- **Dimension-general** state / `h0` / `ht` address injectivity on the
forward `[BV, BK]` tile, given the tile fits (`BV ≤ V`, `BK ≤ K`). -/
theorem fused_recurrent_delta_state_offset_injective_general
    (s : BlockState) (K V BK BV : Nat) (hBV : BV ≤ V) (hBK : BK ≤ K) :
    Function.Injective (fun idx : TileIndex [BV, BK] => stateOffset s K V BK BV idx) := by
  rintro ⟨⟨av, hav⟩, ⟨ak, hak⟩, _⟩ ⟨⟨bv, hbv⟩, ⟨bk, hbk⟩, _⟩ h
  simp [stateOffset, vIndex, kIndex] at h
  have havV : av < V := lt_of_lt_of_le hav hBV
  have hbvV : bv < V := lt_of_lt_of_le hbv hBV
  have hVpos : 0 < V := lt_of_le_of_lt (Nat.zero_le _) havV
  have hcore : (s.pids 1 * BK + ak) * V + av = (s.pids 1 * BK + bk) * V + bv := by
    omega
  have hv : av = bv := by
    have ha : ((s.pids 1 * BK + ak) * V + av) % V = av % V := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
    have hb : ((s.pids 1 * BK + bk) * V + bv) % V = bv % V := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
    have hmod : ((s.pids 1 * BK + ak) * V + av) % V
        = ((s.pids 1 * BK + bk) * V + bv) % V := by rw [hcore]
    rwa [ha, hb, Nat.mod_eq_of_lt havV, Nat.mod_eq_of_lt hbvV] at hmod
  have hk : ak = bk := by
    subst hv
    have hmul : (s.pids 1 * BK + ak) * V = (s.pids 1 * BK + bk) * V := by omega
    have := Nat.eq_of_mul_eq_mul_right hVpos hmul
    omega
  subst hv; subst hk; rfl

/-- **Dimension-general** state address injectivity on the backward `[BK, BV]`
tile, given the tile fits (`BV ≤ V`, `BK ≤ K`). -/
theorem fused_recurrent_delta_dh_offset_injective_general
    (s : BlockState) (K V BK BV : Nat) (hBV : BV ≤ V) (hBK : BK ≤ K) :
    Function.Injective (fun idx : TileIndex [BK, BV] => dhOffset s K V BK BV idx) := by
  rintro ⟨⟨ak, hak⟩, ⟨av, hav⟩, _⟩ ⟨⟨bk, hbk⟩, ⟨bv, hbv⟩, _⟩ h
  simp [dhOffset, vIndex, kIndex] at h
  have havV : av < V := lt_of_lt_of_le hav hBV
  have hbvV : bv < V := lt_of_lt_of_le hbv hBV
  have hVpos : 0 < V := lt_of_le_of_lt (Nat.zero_le _) havV
  have hcore : (s.pids 1 * BK + ak) * V + av = (s.pids 1 * BK + bk) * V + bv := by
    omega
  have hv : av = bv := by
    have ha : ((s.pids 1 * BK + ak) * V + av) % V = av % V := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
    have hb : ((s.pids 1 * BK + bk) * V + bv) % V = bv % V := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left]
    have hmod : ((s.pids 1 * BK + ak) * V + av) % V
        = ((s.pids 1 * BK + bk) * V + bv) % V := by rw [hcore]
    rwa [ha, hb, Nat.mod_eq_of_lt havV, Nat.mod_eq_of_lt hbvV] at hmod
  have hk : ak = bk := by
    subst hv
    have hmul : (s.pids 1 * BK + ak) * V = (s.pids 1 * BK + bk) * V := by omega
    have := Nat.eq_of_mul_eq_mul_right hVpos hmul
    omega
  subst hv; subst hk; rfl

/-- **Dimension-general** `v`-row address injectivity (unconditional: the lane
is an additive term). -/
theorem fused_recurrent_delta_v_row_offset_injective_general
    (s : BlockState) (t s_vo_h V BV : Nat) :
    Function.Injective (fun jv : Fin BV => vRowOffset s t s_vo_h V BV jv) := by
  intro a b h
  apply Fin.ext
  simp [vRowOffset] at h
  omega

/-- **Dimension-general** `o`/`dv` row address injectivity (unconditional). -/
theorem fused_recurrent_delta_out_offset_injective_general
    (s : BlockState) (t s_vo_h B H V BV : Nat) :
    Function.Injective (fun jv : Fin BV => outOffset s t s_vo_h B H V BV jv) := by
  intro a b h
  apply Fin.ext
  simp [outOffset] at h
  omega

/-- **Dimension-general** `dk`/`dq` row address injectivity (unconditional). -/
theorem fused_recurrent_delta_dk_row_offset_injective_general
    (s : BlockState) (t s_qk_h B H K BK : Nat) :
    Function.Injective (fun jk : Fin BK => dkRowOffset s t s_qk_h B H K BK jk) := by
  intro a b h
  apply Fin.ext
  simp [dkRowOffset] at h
  omega

/-- **Dimension-general** headwise `dbeta` row address injectivity
(unconditional). -/
theorem fused_recurrent_delta_dbeta_row_offset_injective_general
    (s : BlockState) (t s_vo_h B H NK V BV : Nat) :
    Function.Injective (fun jv : Fin BV => dbetaRowOffset s t s_vo_h B H NK V BV jv) := by
  intro a b h
  apply Fin.ext
  simp [dbetaRowOffset] at h
  omega

/-! ### ════════ ★ MAIN THEOREM ★ ════════

**SCOPE — this is a claim about eight hand-cut single-step slices, not about the
launched kernels.** The launched surfaces appear only in clauses 1 and 2, which
say nothing more than "they lower to the algorithm layer". Clause 4 writes `HOut`,
a **fiction region** modelling the carried state register `h`, not a Python
tensor. Neither cross-step fold is modeled, and the `STORE_FINAL_STATE` writeback
has no face.

**Genuine, shape-general delta-rule step summary.** Parameterized
over the symbolic head strides `s_qk_h s_vo_h`, grid extents `NK B H T`,
key/value extents `K V`, tile sizes `BK BV`, the real `scale`, the step index
`m`, the final time `T`, and **all** compile-time flags (`IS_HEADWISE_BETA`,
`USE_INITIAL_STATE`, `STORE_FINAL_STATE`, `USE_DH0`, `USE_DHT`). It bundles
every stored output of both kernels, each realized against the genuine
standalone recurrence `deltaState` / the genuine per-step gradient formulas —
never a read-back of the kernel's own output:

1. the full **forward surface** lowers to the algorithm layer (all flags);
2. the full **backward surface** (both loops, `tl.debug_barrier()`, all
   flags) lowers to the algorithm layer;
3. one forward **`v_new` writeback** realizes the genuine delta
   `vNewClosed(m)` (masked), given the carry invariant
   `HPrev = deltaState(m)`;
4. one forward **state-update** body (the `IS_HEADWISE_BETA` specialization
   selected by the flag) realizes `deltaState(m+1)` — the delta-rule
   carry-fold — given `HPrev = deltaState(m)`;
5. one forward **output** body realizes `outputClosed(m)` (the key-axis
   readout of the *post-update* state), given `HNext = deltaState(m+1)`;
6. one backward loop-1 **`dk`** body (flag-selected) realizes the genuine
   `dkStepSpec` over the materialized reverse-carry `DHPrev` and input rows;
7. one backward loop-1 **`dv`** body (flag-selected) realizes `dvStepSpec`;
8. one backward loop-1 headwise **`dbeta`** body realizes `dbetaStepSpec`;
9. one backward loop-1 scalar **`dbeta`** body realizes the full-reduction
   `dbetaScalarStepSpec` (scalar cell);
10. one backward loop-2 **`dk` correction** body, *sequenced onto the loop-1
    stores*: under the two clause-local hypotheses that the `dk`/`dv` rows still
    hold clause 6's and clause 7's values (which is what Python's two-loop
    structure does — loop 2 re-walks the same `dk`/`dv` addresses forward after
    the `tl.debug_barrier()`), the in-place fixup realizes `dkCorrClosed`
    = `dkStepSpec − Σ_{j_v} dvStepSpec · HRec`, the final memory value of the
    `dk` row expressed through the loop-1 formulas rather than through
    unconstrained reads of the kernel's own output rows;
11. one backward loop-2 **`dq`** body (flag-selected) realizes `dqStepSpec`.

Clause 10 is why clause 6 and clause 10 are consistent rather than two
unrelated values for one `dk` cell: they are `Realizes` facts about *different*
slices run from the same initial state (loop 1's body and loop 2's body), and
clause 10 now names the loop-1 value it starts from instead of reading `dk` at
its own write address with nothing said about it.

The `STORE_FINAL_STATE` writeback has **no** conjunct: its face was a masked
memcpy (load address `HFinal + i_bh·K·V + offs_k·V + offs_v`, store address
`ht + …` — identical under the same mask) whose entire content was the
assumption `HFinal = deltaState(T)`, so it has been deleted rather than
presented as a result about `ht`.

Side conditions. Structural: `BV ≤ V`, `BK ≤ K` (the state tile fits the logical
extents, giving state-address injectivity); all row-address injectivities are
unconditional. **Scope-fixing:** `hFullK` / `hFullV` pin the **full-tile
regime** — every step slice in this file loads *unmasked* where every Python
load carries `mask_bk` / `mask_bv` with `other = 0`, so outside that regime the
slices are not the kernel Python runs. They are not used by the proofs; they
scope the statements (see the module docstring for why the masks cannot yet be
put back). **Load-bearing:** `hPrev` and `hNext` are *assumptions* and they
carry the entire forward recurrence; every clause mentioning `deltaState`,
`vNewClosed` or `outputClosed` holds only under them.

**They are not self-propagating, and clause 4 does not discharge `hNext`.**
Clause 4 concludes about the state *after executing* the state-step slice, while
`hNext` constrains region `HOut` in the *initial* state `s` of the output slice;
sequencing one slice's post-state into the next slice's pre-state is not modeled.
(`HOut` is at least now the same region in both clauses — previously clause 4
wrote `HOut` while clause 5 read a separate `HNext` with nothing relating them, so
`deltaState(m+1)` was proved into a region nothing read and independently assumed
for the region that was read.) The cross-step folds threading `h`, `d_h` and the
recomputed `h` are not modeled (carried state presented as materialized buffers;
see the module docstring, including the in-place-`v` consequence).

By the time the backward kernel runs, region `v` holds the forward pass's stored
deltas `v_new` — the backward faces are genuine over the backward launch's actual
inputs. -/
specification fused_recurrent_delta_output_summary_general
    (q k v beta o h0 ht dht dh0 do_ dq dk dv dbeta : RegionName)
    (HPrev HOut DHPrev HRec : RegionName)
    (IS_HEADWISE_BETA USE_INITIAL_STATE STORE_FINAL_STATE USE_DH0 USE_DHT : Bool)
    (m s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) (s : BlockState)
    (hBV : BV ≤ V) (hBK : BK ≤ K)
    -- **the full-tile regime**, forced by the step slices loading *unmasked*
    -- where every Python load carries `mask_bk` / `mask_bv` with `other = 0`;
    -- outside this regime the slices are not the kernel Python runs (see the
    -- module docstring for why masking them is currently blocked)
    (hFullK : ∀ jk : Fin BK, activeK s K BK jk)
    (hFullV : ∀ jv : Fin BV, activeV s V BV jv)
    (hPrev : ∀ idx : TileIndex [BV, BK],
      s.readMem HPrev (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV m idx)
    -- the post-update carry buffer is the SAME region clause 4 stores into;
    -- this hypothesis is NOT discharged by clause 4 (see the docstring)
    (hNext : ∀ idx : TileIndex [BV, BK],
      s.readMem HOut (stateOffset s K V BK BV idx)
        = deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
            s_qk_h s_vo_h T K V BK BV (m + 1) idx) :
    -- (1) the full forward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_delta_fwd_surface q k v beta o h0 ht s_qk_h s_vo_h
      B H T K V BK BV scale USE_INITIAL_STATE STORE_FINAL_STATE
      IS_HEADWISE_BETA).toAlgorithm? = Except.ok alg) ∧
    -- (2) the full backward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_delta_bwd_surface q k v beta dht dh0 do_ dq dk dv
      dbeta h0 s_qk_h s_vo_h NK B H T K V BK BV scale USE_INITIAL_STATE
      IS_HEADWISE_BETA USE_DH0 USE_DHT).toAlgorithm? = Except.ok alg) ∧
    -- (3) the `v_new` writeback realizes the genuine delta `vNewClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_vnew_step_slice HPrev k v
        m s_qk_h s_vo_h K V BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (v, vRowOffset s m s_vo_h V BV jv)))
      (expected := fun jv : Fin BV =>
        vNewClosed s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV m jv)) ∧
    -- (4) the state-update body realizes the genuine `deltaState(m+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_state_step_slice_headwise HPrev k v beta HOut
            m s_qk_h s_vo_h T K V BK BV
        else
          fused_recurrent_delta_state_step_slice_scalarbeta HPrev k v beta HOut
            m s_qk_h s_vo_h T K V BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BV, BK] =>
        some (HOut, stateOffset s K V BK BV idx))
      (expected := fun idx =>
        deltaState s k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV (m + 1) idx)) ∧
    -- (5) the output body realizes the genuine `outputClosed(m)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_output_step_slice HOut q o
        m s_qk_h s_vo_h B H K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (o, outOffset s m s_vo_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        outputClosed s q k v beta h0 IS_HEADWISE_BETA USE_INITIAL_STATE
          s_qk_h s_vo_h T K V BK BV scale m jv)) ∧
    -- (6) the backward loop-1 `dk` body realizes the genuine `dkStepSpec`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dk_step_slice_headwise DHPrev q do_ v beta dk
            m s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dk_step_slice_scalarbeta DHPrev q do_ v beta dk
            m s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s K BK jk)
        (fun jk => (dk, dkRowOffset s m s_qk_h B H K BK jk)))
      (expected := fun jk : Fin BK =>
        dkStepSpec s DHPrev q do_ v beta IS_HEADWISE_BETA
          m s_qk_h s_vo_h T K V BK BV scale jk)) ∧
    -- (7) the backward loop-1 `dv` body realizes the genuine `dvStepSpec`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dv_step_slice_headwise DHPrev q do_ k beta dv
            m s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dv_step_slice_scalarbeta DHPrev q do_ k beta dv
            m s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (dv, outOffset s m s_vo_h B H V BV jv)))
      (expected := fun jv : Fin BV =>
        dvStepSpec s DHPrev q do_ k beta IS_HEADWISE_BETA
          m s_qk_h s_vo_h T K V BK BV scale jv)) ∧
    -- (8) the backward loop-1 headwise `dbeta` body realizes `dbetaStepSpec`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_bwd_dbeta_step_slice_headwise DHPrev q do_ k v
        dbeta m s_qk_h s_vo_h NK B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jv : Fin BV => activeV s V BV jv)
        (fun jv => (dbeta, dbetaRowOffset s m s_vo_h B H NK V BV jv)))
      (expected := fun jv : Fin BV =>
        dbetaStepSpec s DHPrev q do_ k v m s_qk_h s_vo_h K V BK BV scale jv)) ∧
    -- (9) the backward loop-1 scalar `dbeta` body realizes `dbetaScalarStepSpec`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta DHPrev q do_ k v
        dbeta m s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := fun _ : PUnit => some (dbeta, dbetaScalarOffset s m T B H))
      (expected := fun _ =>
        dbetaScalarStepSpec s DHPrev q do_ k v m s_qk_h s_vo_h K V BK BV scale)) ∧
    -- (10) the backward loop-2 `dk` correction body, **sequenced onto the loop-1
    --      stores**: given that the `dk`/`dv` rows still hold what clauses 6 and 7
    --      put there, the final `dk` row is `dkCorrClosed`
    ((∀ jk : Fin BK,
        s.readMem dk (dkRowOffset s m s_qk_h B H K BK jk)
          = dkStepSpec s DHPrev q do_ v beta IS_HEADWISE_BETA
              m s_qk_h s_vo_h T K V BK BV scale jk) →
      (∀ jv : Fin BV,
        s.readMem dv (outOffset s m s_vo_h B H V BV jv)
          = dvStepSpec s DHPrev q do_ k beta IS_HEADWISE_BETA
              m s_qk_h s_vo_h T K V BK BV scale jv) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_recurrent_delta_bwd_dk_correction_step_slice HRec dv dk
          m s_qk_h s_vo_h B H K V BK BV)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (fun jk : Fin BK => activeK s K BK jk)
          (fun jk => (dk, dkRowOffset s m s_qk_h B H K BK jk)))
        (expected := fun jk : Fin BK =>
          dkCorrClosed s DHPrev HRec q do_ k v beta IS_HEADWISE_BETA
            m s_qk_h s_vo_h T K V BK BV scale jk)) ∧
    -- (11) the backward loop-2 `dq` body realizes the genuine `dqStepSpec`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := if IS_HEADWISE_BETA then
          fused_recurrent_delta_bwd_dq_step_slice_headwise HRec k v beta do_ dq
            m s_qk_h s_vo_h B H T K V BK BV scale
        else
          fused_recurrent_delta_bwd_dq_step_slice_scalarbeta HRec k v beta do_ dq
            m s_qk_h s_vo_h B H T K V BK BV scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun jk : Fin BK => activeK s K BK jk)
        (fun jk => (dq, dkRowOffset s m s_qk_h B H K BK jk)))
      (expected := fun jk : Fin BK =>
        dqStepSpec s HRec k v beta do_ IS_HEADWISE_BETA
          m s_qk_h s_vo_h T K V BK BV scale jk)) := by
  have hStateInj := fused_recurrent_delta_state_offset_injective_general
    s K V BK BV hBV hBK
  have hVRowInj := fused_recurrent_delta_v_row_offset_injective_general
    s m s_vo_h V BV
  have hOutInj := fused_recurrent_delta_out_offset_injective_general
    s m s_vo_h B H V BV
  have hDkInj := fused_recurrent_delta_dk_row_offset_injective_general
    s m s_qk_h B H K BK
  have hDbetaInj := fused_recurrent_delta_dbeta_row_offset_injective_general
    s m s_vo_h B H NK V BV
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact fused_recurrent_delta_fwd_surface_toAlgorithm_supported q k v beta o
      h0 ht s_qk_h s_vo_h B H T K V BK BV scale USE_INITIAL_STATE
      STORE_FINAL_STATE IS_HEADWISE_BETA
  · exact fused_recurrent_delta_bwd_surface_toAlgorithm_supported q k v beta
      dht dh0 do_ dq dk dv dbeta h0 s_qk_h s_vo_h NK B H T K V BK BV scale
      USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT
  · exact fused_recurrent_delta_vnew_step_closed_form HPrev k v beta h0
      IS_HEADWISE_BETA USE_INITIAL_STATE m s_qk_h s_vo_h T K V BK BV s
      hVRowInj hPrev
  · exact fused_recurrent_delta_state_step_closed_form HPrev k v beta h0 HOut
      IS_HEADWISE_BETA USE_INITIAL_STATE m s_qk_h s_vo_h T K V BK BV s
      hStateInj hPrev
  · exact fused_recurrent_delta_output_step_closed_form HOut q k v beta h0 o
      IS_HEADWISE_BETA USE_INITIAL_STATE m s_qk_h s_vo_h B H T K V BK BV scale
      s hOutInj hNext
  · exact fused_recurrent_delta_bwd_dk_step_slice_compute_correct DHPrev q do_
      v beta dk IS_HEADWISE_BETA m s_qk_h s_vo_h B H T K V BK BV scale s hDkInj
  · exact fused_recurrent_delta_bwd_dv_step_slice_compute_correct DHPrev q do_
      k beta dv IS_HEADWISE_BETA m s_qk_h s_vo_h B H T K V BK BV scale s hOutInj
  · exact fused_recurrent_delta_bwd_dbeta_step_slice_headwise_compute_correct
      DHPrev q do_ k v dbeta m s_qk_h s_vo_h NK B H T K V BK BV scale s hDbetaInj
  · exact fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta_compute_correct
      DHPrev q do_ k v dbeta m s_qk_h s_vo_h B H T K V BK BV scale s
  · intro hDkRow hDvRow
    exact fused_recurrent_delta_bwd_dk_correction_step_sequenced DHPrev HRec q do_
      k v beta dk dv IS_HEADWISE_BETA m s_qk_h s_vo_h B H T K V BK BV scale s
      hDkInj hDkRow hDvRow
  · exact fused_recurrent_delta_bwd_dq_step_slice_compute_correct HRec k v
      beta do_ dq IS_HEADWISE_BETA m s_qk_h s_vo_h B H T K V BK BV scale s hDkInj

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta

