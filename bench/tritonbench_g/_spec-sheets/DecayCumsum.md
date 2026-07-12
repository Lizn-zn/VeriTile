# Spec sheet — `bench/tritonbench_g/decay_cumsum/DecayCumsum.lean`

**Python source:** `bench/tritonbench_g/decay_cumsum/decay_cumsum.py`

## Public theorem: `of`

**Statement:**
```lean
specification of the `q *= exp2(g) * scale` map (here `scale = 1`). -/
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `decay_cumsum_backward_closed_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General `output_summary`.** The executed backward surface realizes all three
genuine closed forms (`bwdDQInterClosed` / `bwdDKInterClosed` / `bwdDGClosed`). -/
```
</details>

**Statement:**
```lean
specification decay_cumsum_backward_closed_output_summary_general :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DQInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DKInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i))
```

**Closed-form spec defs (transitive):** `bwd_decay_global_cumsum_surface`, `active`, `offset`, `bwdDQInterClosed`, `bwdDKInterClosed`, `bwdDGClosed`, `elemIndex`, `baseOffset`, `bwdDGSummand`

<details><summary><code>bwd_decay_global_cumsum_surface</code></summary>

```
/-- Surface transcription of `decay_cumsum.py`'s `bwd_decay_global_cumsum`.

The Python kernel traverses the chunk in reverse and decrements pointers; the
DSL surface preserves that reverse range and pointer movement directly. -/
```
```lean
def bwd_decay_global_cumsum_surface
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  p_q = Q + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_k = K + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_g = G + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dg = DG + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inner = DQInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inner = DKInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inter = DQInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inter = DKInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  cum_grad_dg = tl.zeros([$(BK)], dtype=tl.float32)
  mask = (i_k * $(BK) + offs) < $(DK)
  last_g = tl.zeros([$(BK)], dtype=tl.float32)
  for t in range($(BT) - $(1), -$(1), -$(1)) {
    g_val = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    if t == $(BT) - $(1) {
      last_g = g_val
    }
    dq1 = tl.load(p_dq_inner, mask=mask, other=0)
    dq2 = tl.load(p_dq_inter, mask=mask, other=0)
    dq2 *= tl.math.exp2(g_val)
    dq = dq1 + dq2
    tl.store(p_dq_inter, dq, mask=mask)
    dk1 = tl.load(p_dk_inner, mask=mask, other=0)
    dk2 = tl.load(p_dk_inter, mask=mask, other=0)
    dk2 *= tl.math.exp2(last_g - g_val)
    dk = dk1 + dk2
    tl.store(p_dk_inter, dk, mask=mask)
    q_val = tl.load(p_q, mask=mask, other=0)
    k_val = tl.load(p_k, mask=mask, other=0)
    dg_val = dq * q_val - dk * k_val
    cum_grad_dg += dg_val
    tl.store(p_dg, (cum_grad_dg).to(p_dg.dtype.element_ty), mask=mask)
    p_g -= $(DK)
    p_k -= $(DK)
    p_q -= $(DK)
    p_dq_inner -= $(DK)
    p_dk_inner -= $(DK)
    p_dq_inter -= $(DK)
    p_dk_inter -= $(DK)
    p_dg -= $(DK)
  }
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (DK BK : Nat) (i : Fin BK) : Prop :=
  elemIndex s BK i < DK
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset
    (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : Nat :=
  baseOffset s s_qk_h DK t_rel BT BK + i.val
```
</details>

<details><summary><code>bwdDQInterClosed</code></summary>

```
/-- **Genuine `dq_inter` closed form.** At chunk row `t_rel` and lane `i`, the
backward kernel writes `dq_inner[idx] + dq_inter_in[idx] * exp2(g[idx])` into
`dq_inter`, with `exp2(x) = Real.exp (x * Real.log 2)`. -/
```
```lean
noncomputable def bwdDQInterClosed
    (s : BlockState) (DQInner DQInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem DQInner (offset s s_qk_h DK t_rel.val BT BK i) +
    s.readMem DQInter (offset s s_qk_h DK t_rel.val BT BK i) *
      Real.exp (s.readMem G (offset s s_qk_h DK t_rel.val BT BK i) * Real.log 2)
```
</details>

<details><summary><code>bwdDKInterClosed</code></summary>

```
/-- **Genuine `dk_inter` closed form.** At chunk row `t_rel` and lane `i`, the
backward kernel writes
`dk_inner[idx] + dk_inter_in[idx] * exp2(g[row BT-1] - g[idx])` into `dk_inter`,
where `g[row BT-1]` is the captured `last_g`. -/
```
```lean
noncomputable def bwdDKInterClosed
    (s : BlockState) (DKInner DKInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem DKInner (offset s s_qk_h DK t_rel.val BT BK i) +
    s.readMem DKInter (offset s s_qk_h DK t_rel.val BT BK i) *
      Real.exp ((s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i) -
        s.readMem G (offset s s_qk_h DK t_rel.val BT BK i)) * Real.log 2)
```
</details>

<details><summary><code>bwdDGClosed</code></summary>

```
/-- **Genuine `dg` closed form.** At chunk row `t_rel` and lane `i`, the backward
kernel writes the reverse cumulative sum
`Σ_{j = t_rel}^{BT-1} (dq_inter[j]*q[j] - dk_inter[j]*k[j])` into `dg`. This is
the honest reverse-prefix-scan specification of the carried `cum_grad_dg`
accumulator (the `range(BT-1,-1,-1)` loop threads `cum_grad_dg += dq*q - dk*k`).
This is *not* the executed kernel readback. -/
```
```lean
noncomputable def bwdDGClosed
    (s : BlockState) (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t_rel.val),
    bwdDGSummand s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
      ⟨t_rel.val + d.val, by omega⟩ i

/-! ### Proof recipe (backward closed forms)

The three genuine closed forms above (`bwdDQInterClosed`, `bwdDKInterClosed`,
`bwdDGClosed`) are the honest, non self-referential specifications that replace
the (now-deleted) `decayBackwardSurfaceValue`. They are connected to the executed
`bwd_decay_global_cumsum_surface` in
`decay_cumsum_backward_closed_output_summary_general` (and its three
faces `bwd_decay_cumsum_d{q,k}_inter_closed_compute_correct_general` /
`bwd_decay_cumsum_dg_closed_compute_correct_general`) at the end of this file, following
the same closed-form recipe as the forward/prepare general stacks,
but the backward
loop body is ~25 statements with a conditional `last_g` capture and three masked
stores per iteration, traversed over the reverse `range(BT-1,-1,-1)` rows (lowered to a
forward `forRangeDyn "__rev_t" 0 BT 1` with `t := BT-1 - __rev_t`). A single
`simp [exec, …, evalOp.eq_def, stepForRangeAux.*]` blast does *not* scale to this
body (it does not terminate within ~9 min even at 8M heartbeats), so the
mandated per-statement architecture is required:

1. `exec → stepStmts toAlgKernel.body`, with the surface body decomposed by
   `bwd_body_decomp_general` into the 15-stmt prologue + the `forRangeDyn` reverse loop.
2. Drive the `forRangeDyn` loop with `forRangeAux_inv` /
   `VeriTile.Triton.forRangeDyn_inv` (carry invariant on `cum_grad_dg` =
   partial reverse prefix sum), *not* a `simp` over the whole loop.
3. Per body statement: `stepStmts.cons_some` + `simp only` over the named
   `evalOp_*` lemmas (`evalOp_add/mul/sub/ref/…`, `evalOp_ref_setReg*`) — never
   `evalOp.eq_def` whnf over the nested `setReg` literal state.
4. Read back each output with the masked-scatter lemmas
   (`scatter_readback_prop_masked_nd`,
   `scatter_prop_masked_preserves_other_{offset,region}`), peeling the later
   stores in reverse, exactly as the forward row-1 proof does.
5. Bridge to `ComputeCorrect.Realizes_without_Rounding` via `realizes_writeIf_iff` +
   `computeCorrect_of_toAlgKernel` (done; `decayBackwardSurfaceValue` deleted).

This plan is now fully realized dimension-generally: `bwd_prologue_eval_general`
runs the 15-stmt prologue, `bwd_decay_cumsum_step_general` advances the reverse-loop
invariant `bwdInvG` (one iteration, head + conditional `last_g` capture + three
masked stores), `bwd_loop_drive_general` assembles prologue + the full `range(BT)`
reverse loop, and the three `_general` readback theorems certify the closed forms
(the `dg` face uses the genuine reverse cumsum via `bwdCumPartialG`).

The `dq_inter`/`dk_inter` faces are pointwise (no carry); only `dg` needs the
reverse-scan invariant. Region-distinctness side hypotheses (`DQInter ≠ DKInter`
etc.) are needed so a later store does not clobber an earlier readback, mirroring
the forward `G ≠ GO` and `prepare` `Q ≠ QG …` hypotheses. -/

/-! ## Per-statement op-eval recipes (backward kernel, recipe layer)

These are the standalone, register-readback-abstracted `stepStmt`/`evalOp`
reduction lemmas for *each statement kind* appearing in the
`bwd_decay_global_cumsum_surface` body (15-stmt prologue + 25-stmt reverse loop
body). They are the mandated per-statement architecture building blocks: every
```
</details>

<details><summary><code>elemIndex</code></summary>

```lean
def elemIndex (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 0 * BK + i.val
```
</details>

<details><summary><code>baseOffset</code></summary>

```lean
def baseOffset (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) : Nat :=
  s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK + s.pids 0 * BK
```
</details>

<details><summary><code>bwdDGSummand</code></summary>

```
/-- The per-row `dg` summand `dq_inter[j] * q[j] - dk_inter[j] * k[j]`, written
in terms of the genuine `dq_inter`/`dk_inter` closed forms above. -/
```
```lean
noncomputable def bwdDGSummand
    (s : BlockState) (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (j : Fin BT) (i : Fin BK) : ℝ :=
  bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK j i *
      s.readMem Q (offset s s_qk_h DK j.val BT BK i) -
    bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK j i *
      s.readMem K (offset s s_qk_h DK j.val BT BK i)
```
</details>

## Also present (pinned special-case summaries)
- `fwd_decay_cumsum_surface_closed_compute_correct_general`
- `prepare_qg_kg_surface_qg_closed_compute_correct_general`
- `prepare_qg_kg_surface_kg_closed_compute_correct_general`
- `bwd_decay_cumsum_dq_inter_closed_compute_correct_general`
- `bwd_decay_cumsum_dk_inter_closed_compute_correct_general`
- `bwd_decay_cumsum_dg_closed_compute_correct_general`
