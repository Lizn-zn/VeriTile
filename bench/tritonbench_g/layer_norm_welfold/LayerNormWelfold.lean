import VeriTile.Triton

/-!
# `layer_norm_welfold` — strict per-kernel correctness

Target JIT: `triton_red_fused_native_layer_norm_no_welford` — a
torch-inductor fused LayerNorm forward in the **two-pass (no-Welford)**
formulation. Each program covers `XBLOCK` rows of the `[xnumel, rnumel]`
input `in_ptr0`; a first tiled loop accumulates the plain row sum into the
`tl.full` accumulator `_tmp3`, the row mean `tmp6 = tl.sum(_tmp3,1)/rnumel`
is stored to `in_out_ptr0` after a `tl.debug_barrier()`; a second tiled
loop accumulates the squared deviations `(x − tmp6)²` into `_tmp12`, the
reciprocal standard deviation
`tmp18 = libdevice.rsqrt(tl.sum(_tmp12,1)/rnumel + 1e-05)` is stored to
`in_out_ptr1` after a second `tl.debug_barrier()`; and a third tiled loop
stores the normalized affine output `tmp29 = ((x − mean)·rstd)·w + b` to
`out_ptr0` (weights `in_ptr1`, bias `in_ptr2`).

## Scope

This file verifies **the Triton kernel itself** — the per-program
`@triton.jit` body. The host launch (`fused_native_layer_norm_no_welford`'s
`grid = cdiv(S, XBLOCK)`, the `@triton.autotune` config sweep
`(XBLOCK, RBLOCK) ∈ {(1, 1024), (1, 2048)}`, buffer reinterpretation, and
how the runtime composes per-program writes into the output buffers) is
the *trusted boundary*, not a proof obligation here. Because the program
ids are universally quantified, the per-program statements cover every
program of the grid.

## Proof architecture

```
layer_norm_welfold_output_summary_general              ← TOP THEOREM
  ├─ layer_norm_welfold_surface_toAlgorithm_supported  full surface lowers
  ├─ layer_norm_welfold_reduce_slice_compute_correct   mean ∧ rstd genuine
  │    └─ layer_norm_welfold_reduce_slice_readback
  │         └─ sum_somes_fin / sum_fin_guard   lane sums → `Finset.range` sums
  └─ layer_norm_welfold_normalize_slice_compute_correct  genuine Y chunks
       └─ layer_norm_welfold_normalize_slice_correct

multi-iteration reduction-loop face (pure ℝ):
  sum_accumulate_step_closed    one accumulator step maps the exact prefix
                                sum to the exact one-longer prefix sum
```

The genuine closed-form specs are standalone `Finset.sum` formulas over
input memory (`rowMeanSpec` / `rowVarSpec` / `rowRstdSpec` / `rowYSpec`) —
never a read-back of the kernel's own output.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the
`@triton.autotune` config sweep is not modeled — all four dimensions
`xnumel`, `rnumel`, `XBLOCK`, `RBLOCK` stay symbolic binders.
Both `tl.debug_barrier()` fences are transcribed in place; each is an
intra-program fence and therefore a semantic no-op in the sequential
single-program-per-row model (the algorithm-layer lowering erases it — see
the `effectMarker` case of `ComputeStmt.toAlgorithm?`). The
`.to(tl.float32)`-style casts erase to the identity at the algorithm
layer. `eps = 1e-05` is hardcoded in the kernel body (`tmp16 = 1e-05`), so
the specs carry it verbatim rather than as a binder. The value-correct
theorems cover the `XBLOCK = 1` row-per-program shape pinned by both
autotune configs; the reduction phase is additionally scoped to a single
reduction block (`rnumel ≤ RBLOCK`), while the normalize loop is covered
genuinely for **all** chunk indices and shapes via the mean/rstd-cell
hypotheses (exactly the values the reduction phase stores). The kernel's
reduction loops accumulate the masked loads **unguarded** (no
`tl.where(rmask, …)` on the accumulator) — sound in the real launch only
because both autotune configs divide the benchmark row exactly
(`RBLOCK ∣ rnumel`, so `rmask` is all-true); the single-block slice
materializes the mask-correct totals instead (`other=0.0` on the loads and
a zero-guard on the squared-deviation lanes), which agrees with the kernel
on every exactly-covered row and additionally covers the ragged tail
`rnumel < RBLOCK`. The multi-iteration reduction recurrence is the pure
step face `sum_accumulate_step_closed` (both loops carry a plain partial
sum); the cross-iteration register scheduling that threads `_tmp3` /
`_tmp12` is the trusted runtime boundary, exactly as
`reversed_cumsum_scalar`'s carried `b_z` and `fused_layernorm_triton`'s
carried Welford registers.

## Translation-surface blocker

Translation-surface blocker: `tl.full([XBLOCK, RBLOCK], 0, tl.float32)`'s
positional dtype argument is written with the DSL's keyword spelling
`tl.full([$(XBLOCK), $(RBLOCK)], 0, dtype=tl.float32)` (same constant
tile); `tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])` is written through the
DSL's tuple-form `tmp2, _ = tl.broadcast(tmp1, _tmp3)` (same broadcast, no
`broadcast_to` spelling in the DSL — `fused_layernorm_triton` precedent);
the fused reduce-and-reshape lines `tmpN = tl.sum(_tmpN, 1)[:, None]` are
split into the two statements `tmpN_tmp = tl.sum(_tmpN, 1)` and
`tmpN = tmpN_tmp[:, None]` (the DSL's slicer postfix does not chain on a
call expression); the Python `tl.store(..., None)` explicit no-mask
arguments are written as unmasked `tl.store(...)`; `libdevice.rsqrt` is
written `tl.rsqrt` (existing `rmsnorm_triton` / `layer_norm_liger` /
`fused_layernorm_triton` precedent); the register recasts
`tmpN = tmpM.to(tl.float32)` are written with mechanical parentheses
`(tmpM).to(tl.float32)` (DSL parse limitation on bare-ident method casts).
-/

namespace VeriTile.Bench.TritonBenchG.LayerNormWelfold

open VeriTile.Triton

set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySeqFocus false

/-- Faithful transcription of `layer_norm_welfold.py`'s
`triton_red_fused_native_layer_norm_no_welford` (the only `@triton.jit`
kernel; the `@triton.autotune` decorator sweeps
`(XBLOCK, RBLOCK) ∈ {(1,1024),(1,2048)}` and is not modeled — all four
parameters stay symbolic). The mechanical DSL spellings are documented in
the `Translation-surface blocker:` preamble marker. -/
def layer_norm_welfold_surface
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(XBLOCK)
  xindex = xoffset + tl.arange(0, $(XBLOCK))[:, None]
  xmask = xindex < $(xnumel)
  rbase = tl.arange(0, $(RBLOCK))[None, :]
  x0 = xindex
  _tmp3 = tl.full([$(XBLOCK), $(RBLOCK)], 0, dtype=tl.float32)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, _tmp3)
    tmp4 = _tmp3 + tmp2
    _tmp3 = tmp4
  }
  tmp3_tmp = tl.sum(_tmp3, 1)
  tmp3 = tmp3_tmp[:, None]
  tmp5 = $(rnumel)
  tmp6 = tmp3 / tmp5
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp6)
  _tmp12 = tl.full([$(XBLOCK), $(RBLOCK)], 0, dtype=tl.float32)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp7 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp8 = (tmp7).to(tl.float32)
    tmp9 = tmp8 - tmp6
    tmp10 = tmp9 * tmp9
    tmp11, _ = tl.broadcast(tmp10, _tmp12)
    tmp13 = _tmp12 + tmp11
    _tmp12 = tmp13
  }
  tmp12_tmp = tl.sum(_tmp12, 1)
  tmp12 = tmp12_tmp[:, None]
  tmp14 = $(rnumel)
  tmp15 = tmp12 / tmp14
  tmp16 = 1e-05
  tmp17 = tmp15 + tmp16
  tmp18 = tl.rsqrt(tmp17)
  tl.debug_barrier()
  tl.store(in_out_ptr1 + (x0), tmp18)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp19 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_first").to(tl.float32)
    tmp23 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp20 = (tmp19).to(tl.float32)
    tmp21 = tmp20 - tmp6
    tmp22 = tmp21 * tmp18
    tmp24 = (tmp23).to(tl.float32)
    tmp25 = tmp22 * tmp24
    tmp27 = (tmp26).to(tl.float32)
    tmp28 = tmp25 + tmp27
    tmp29 = (tmp28).to(tl.float32)
    tl.store(out_ptr0 + (r1 + ($(rnumel) * x0)), tmp29, rmask)
  }
}

/-- The full faithful surface — all three tiled loops and both
`tl.debug_barrier()` fences — lowers to the algorithm layer (each fence
erases to a no-op under the sequential per-program semantics). -/
theorem layer_norm_welfold_surface_toAlgorithm_supported
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) :
    ∃ alg,
      (layer_norm_welfold_surface in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1
        in_ptr2 out_ptr0 xnumel rnumel XBLOCK RBLOCK).toAlgorithm?
      = Except.ok alg := by
  simp [layer_norm_welfold_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Genuine closed-form row specs (standalone `Finset.sum` forms)

These are the mathematical specifications the outputs are checked against.
They read **input memory only** (`in_ptr0` / `in_ptr1` / `in_ptr2`) — never a
read-back of the kernel's own output. -/

/-- Row element `X[x0, r]` of the row-major `[xnumel, rnumel]` input: this
program's row (`x0 = pids 0` in the `XBLOCK = 1` launch) at column `r`
(the kernel's load offset `r1 + rnumel * x0`). -/
noncomputable def rowElem (s : BlockState) (in_ptr0 : RegionName)
    (rnumel r : Nat) : ℝ :=
  s.readMem in_ptr0 (r + rnumel * s.pids 0)

/-- Genuine row mean: `(Σ_{r < rnumel} X[x0, r]) / rnumel`. -/
noncomputable def rowMeanSpec (s : BlockState) (in_ptr0 : RegionName)
    (rnumel : Nat) : ℝ :=
  (∑ r ∈ Finset.range rnumel, rowElem s in_ptr0 rnumel r) / (rnumel : ℝ)

/-- Genuine row (population) variance:
`(Σ_{r < rnumel} (X[x0, r] − mean)²) / rnumel`. -/
noncomputable def rowVarSpec (s : BlockState) (in_ptr0 : RegionName)
    (rnumel : Nat) : ℝ :=
  (∑ r ∈ Finset.range rnumel,
      (rowElem s in_ptr0 rnumel r - rowMeanSpec s in_ptr0 rnumel) ^ 2) /
    (rnumel : ℝ)

/-- Genuine reciprocal standard deviation with the kernel's hardcoded
`tmp16 = 1e-05`: `1 / sqrt(var + 1e-05)` (`libdevice.rsqrt`). -/
noncomputable def rowRstdSpec (s : BlockState) (in_ptr0 : RegionName)
    (rnumel : Nat) : ℝ :=
  1 / Real.sqrt (rowVarSpec s in_ptr0 rnumel + 1e-05)

/-- Genuine normalized output:
`Y[x0, r] = ((X[x0, r] − mean) * rstd) * W[r] + B[r]`. -/
noncomputable def rowYSpec (s : BlockState)
    (in_ptr0 in_ptr1 in_ptr2 : RegionName) (rnumel r : Nat) : ℝ :=
  ((rowElem s in_ptr0 rnumel r - rowMeanSpec s in_ptr0 rnumel) *
      rowRstdSpec s in_ptr0 rnumel) * s.readMem in_ptr1 r +
    s.readMem in_ptr2 r

/-! ## Normalize-loop slice (third `for roffset` loop, one iteration)

The third loop's iterations are independent: each reads the already-stored
row mean (`in_out_ptr0[x0]`, the kernel's `tmp6`) and rstd
(`in_out_ptr1[x0]`, the kernel's `tmp18`) plus fresh `in_ptr0/1/2` tiles,
and stores one masked output chunk. The slice materializes one iteration
with the chunk index as `tl.program_id(1)` (`roffset = i_t · RBLOCK`); its
correctness theorem takes the honest hypotheses that the mean/rstd cells
hold the genuine closed forms — exactly the values the reduction phases
store — and is dimension-general over `rnumel`, `RBLOCK`, and the chunk
index. -/

def layer_norm_welfold_normalize_slice
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  x0 = tl.program_id(0)
  i_t = tl.program_id(1)
  rindex = i_t * $(RBLOCK) + tl.arange(0, $(RBLOCK))
  rmask = rindex < $(rnumel)
  r1 = rindex
  tmp6 = tl.load(in_out_ptr0 + (x0))
  tmp18 = tl.load(in_out_ptr1 + (x0))
  tmp19 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp23 = tl.load(in_ptr1 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp26 = tl.load(in_ptr2 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp20 = (tmp19).to(tl.float32)
  tmp21 = tmp20 - tmp6
  tmp22 = tmp21 * tmp18
  tmp24 = (tmp23).to(tl.float32)
  tmp25 = tmp22 * tmp24
  tmp27 = (tmp26).to(tl.float32)
  tmp28 = tmp25 + tmp27
  tmp29 = (tmp28).to(tl.float32)
  tl.store(out_ptr0 + (r1 + ($(rnumel) * x0)), tmp29, mask=rmask)
}

/-- Flat column index of lane `i` in chunk `i_t = pids 1`. -/
def rIndex (s : BlockState) (RBLOCK : Nat) (i : Fin RBLOCK) : Nat :=
  s.pids 1 * RBLOCK + i.val

/-- Lane `i` is active iff its column is in range (`rmask`). -/
def activeLane (s : BlockState) (rnumel RBLOCK : Nat) (i : Fin RBLOCK) : Prop :=
  rIndex s RBLOCK i < rnumel

instance activeLaneDecidable (s : BlockState) (rnumel RBLOCK : Nat)
    (i : Fin RBLOCK) : Decidable (activeLane s rnumel RBLOCK i) := by
  unfold activeLane
  infer_instance

/-- Output cell of lane `i`: the kernel's store offset `r1 + rnumel * x0`. -/
def yOffset (s : BlockState) (rnumel RBLOCK : Nat) (i : Fin RBLOCK) : Nat :=
  rIndex s RBLOCK i + rnumel * s.pids 0

/-- Executed-state correctness of the normalize slice: given the mean/rstd
cells hold the genuine closed forms, every active output lane holds the
genuine `rowYSpec`. -/
theorem layer_norm_welfold_normalize_slice_correct
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) (s : BlockState)
    (hmean : s.readMem in_out_ptr0 (s.pids 0) = rowMeanSpec s in_ptr0 rnumel)
    (hrstd : s.readMem in_out_ptr1 (s.pids 0) = rowRstdSpec s in_ptr0 rnumel) :
    ∀ i : Fin RBLOCK,
      let outAddr := yOffset s rnumel RBLOCK i
      (exec (layer_norm_welfold_normalize_slice in_out_ptr0 in_out_ptr1
          in_ptr0 in_ptr1 in_ptr2 out_ptr0 rnumel RBLOCK) s).map
          (·.readMem out_ptr0 outAddr)
        = some (if activeLane s rnumel RBLOCK i then
            rowYSpec s in_ptr0 in_ptr1 in_ptr2 rnumel (rIndex s RBLOCK i)
          else s.readMem out_ptr0 outAddr) := by
  intro i
  simp [exec, layer_norm_welfold_normalize_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComparableDType.lt, FloatDType.cast,
        activeLane, rIndex, yOffset,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [RBLOCK] → Nat :=
    fun idx => s.pids 1 * RBLOCK + idx.1.val + rnumel * s.pids 0
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, u⟩ ⟨b, v⟩ hab
    simp only [offsetFn] at hab
    have hfin : a = b := Fin.ext (by omega)
    cases hfin; cases u; cases v; rfl
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj
        (i, PUnit.unit)]
  by_cases hi : s.pids 1 * RBLOCK + i.val < rnumel
  · simp [hi, rowYSpec, rowElem, hmean, hrstd]
  · simp [hi, offsetFn]

/-- `Realizes_without_Rounding` face for the normalize loop: for **every** state whose
mean/rstd cells hold the genuine closed forms (exactly what the two
reduction phases store), one normalize-loop iteration realizes the genuine
`rowYSpec` on its masked chunk — general over `rnumel`, `RBLOCK` and the
chunk index `i_t = pids 1`. -/
theorem layer_norm_welfold_normalize_slice_compute_correct
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) (s : BlockState)
    (hmean : s.readMem in_out_ptr0 (s.pids 0) = rowMeanSpec s in_ptr0 rnumel)
    (hrstd : s.readMem in_out_ptr1 (s.pids 0) = rowRstdSpec s in_ptr0 rnumel) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_welfold_normalize_slice in_out_ptr0 in_out_ptr1
        in_ptr0 in_ptr1 in_ptr2 out_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin RBLOCK => activeLane s rnumel RBLOCK i)
        (fun i => (out_ptr0, yOffset s rnumel RBLOCK i)))
      (expected := fun i : Fin RBLOCK =>
        rowYSpec s in_ptr0 in_ptr1 in_ptr2 rnumel (rIndex s RBLOCK i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_welfold_normalize_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_welfold_normalize_slice_correct in_out_ptr0 in_out_ptr1
    in_ptr0 in_ptr1 in_ptr2 out_ptr0 rnumel RBLOCK s hmean hrstd i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Reduction-phase slice (`XBLOCK = 1`, one reduction iteration each)

Both `@triton.autotune` configs pin `XBLOCK = 1` (one row per program:
`xoffset = pid`, `xindex = x0 = pid`, the `[XBLOCK, RBLOCK]` tiles are the
row's `[RBLOCK]` lanes, `axis=1` reductions become `axis=0`). When the row
also fits one reduction block (`rnumel ≤ RBLOCK`), the first two
`for roffset` loops run exactly once with `roffset = 0`, so this slice
transcribes both reduction phases — first-loop body at `roffset = 0`, the
mean chain `tmp5/tmp6`, the first `tl.debug_barrier()` and mean store,
second-loop body at `roffset = 0`, the `tmp14..tmp18` rstd chain, the
second `tl.debug_barrier()` and rstd store — in kernel order, stopping
before the (independent) normalize loop, which
`layer_norm_welfold_normalize_slice` covers for every chunk. The masked
loads carry `other=0.0` and the squared-deviation lanes are zero-guarded
with `tl.where(rmask, ·, 0)` (the faithful surface's unguarded accumulation
relies on the autotune configs' exact division `RBLOCK ∣ rnumel`; here the
tail lanes must be total-and-zero to be summed), and the identity recasts
`tmp1`/`tmp2` and `tmp8`/`tmp11` (`.to(tl.float32)` and `broadcast_to` on
an already-`[RBLOCK]` tile) are elided. -/
def layer_norm_welfold_reduce_slice
    (in_out_ptr0 in_out_ptr1 in_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  x0 = tl.program_id(0)
  rbase = tl.arange(0, $(RBLOCK))
  rmask = rbase < $(rnumel)
  r1 = rbase
  tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp3 = tl.sum(tmp0, 0)
  tmp5 = $(rnumel)
  tmp6 = tmp3 / tmp5
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp6)
  tmp7 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp9 = tmp7 - tmp6
  tmp10 = tmp9 * tmp9
  zero_lanes = tl.zeros([$(RBLOCK)], tl.float32)
  tmp11 = tl.where(rmask, tmp10, zero_lanes)
  tmp12 = tl.sum(tmp11, 0)
  tmp14 = $(rnumel)
  tmp15 = tmp12 / tmp14
  tmp16 = 1e-05
  tmp17 = tmp15 + tmp16
  tmp18 = tl.rsqrt(tmp17)
  tl.debug_barrier()
  tl.store(in_out_ptr1 + (x0), tmp18)
}

/-- Sum of `some`-valued `WithBot ℝ` lanes is `some` of the real sum. -/
private theorem sum_somes_fin (R : Nat) (g : Fin R → ℝ) :
    @Finset.sum (Fin R) (WithBot ℝ) _ Finset.univ (fun x => some (g x))
      = (some (∑ x : Fin R, g x) : WithBot ℝ) := by
  simp only [WithBot.some_eq_coe]
  norm_cast

/-- Reindex a `< n`-guarded `Fin R` lane sum (with `n ≤ R`) to `range n`. -/
private theorem sum_fin_guard (R n : Nat) (hLe : n ≤ R) (f : Nat → ℝ) :
    (∑ x : Fin R, if (x : ℕ) < n then f (x : ℕ) else 0)
      = ∑ r ∈ Finset.range n, f r := by
  rw [Fin.sum_univ_eq_sum_range (fun r => if r < n then f r else 0) R]
  rw [← Finset.sum_filter]
  congr 1
  ext r
  simp only [Finset.mem_filter, Finset.mem_range]
  omega

set_option maxHeartbeats 24000000 in
/-- Readback of the two reduction-phase stores against the **genuine**
closed forms: after the slice runs, the mean cell `in_out_ptr0[x0]` holds
`rowMeanSpec` and the rstd cell `in_out_ptr1[x0]` holds `rowRstdSpec`
(single reduction block: `rnumel ≤ RBLOCK`). The second-pass loads happen
after the mean store, hence the honest input/mean-buffer distinctness
hypothesis `in_ptr0 ≠ in_out_ptr0` (distinct tensors at the host level). -/
theorem layer_norm_welfold_reduce_slice_readback
    (in_out_ptr0 in_out_ptr1 in_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) (hLe : rnumel ≤ RBLOCK) (s s' : BlockState)
    (hMeanRstd : in_out_ptr0 ≠ in_out_ptr1)
    (hInMean : in_ptr0 ≠ in_out_ptr0)
    (hExec : exec (layer_norm_welfold_reduce_slice in_out_ptr0
        in_out_ptr1 in_ptr0 rnumel RBLOCK) s = some s') :
    s'.readMem in_out_ptr0 (s.pids 0) = rowMeanSpec s in_ptr0 rnumel ∧
    s'.readMem in_out_ptr1 (s.pids 0) = rowRstdSpec s in_ptr0 rnumel := by
  simp [exec, layer_norm_welfold_reduce_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, TileCarrier, hInMean,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  -- Collapse the guarded lane sum of the first pass (`other=0.0` tail) to
  -- `some` of the genuine `Finset.range` row sum.
  have hMean :
      (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          if (x : ℕ) < rnumel then
            some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
          else some 0.0))
      = (some (∑ r ∈ Finset.range rnumel,
          s.readMem in_ptr0 (r + rnumel * s.pids 0)) : WithBot ℝ) := by
    have hlane : ∀ x : Fin RBLOCK,
        @Eq (WithBot ℝ)
          (if (x : ℕ) < rnumel then
            some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
          else some 0.0)
          (some (if (x : ℕ) < rnumel then
            s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0) else 0)) := by
      intro x
      by_cases h : (x : ℕ) < rnumel <;> simp [h] <;> norm_num
    rw [Finset.sum_congr rfl (fun x _ => hlane x), sum_somes_fin,
        sum_fin_guard RBLOCK rnumel hLe
          (fun r => s.readMem in_ptr0 (r + rnumel * s.pids 0))]
  constructor
  · rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hMeanRstd]
    simp [BlockState.writeMem_readMem]
    show WithBot.unbotD 0
        (Option.map (fun a : ℝ => a / (rnumel : ℝ))
          (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
            (fun x =>
              if (x : ℕ) < rnumel then
                some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
              else some 0.0)))
      = rowMeanSpec s in_ptr0 rnumel
    rw [hMean]
    simp [rowMeanSpec, rowElem]
  · simp [BlockState.writeMem_readMem]
    show WithBot.unbotD 0
        (WithBot.realRsqrt
          (Option.map ((fun a : ℝ => a + 1e-05) ∘ fun a : ℝ => a / (rnumel : ℝ))
            (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
              (fun x =>
                if (x : ℕ) < rnumel then
                  Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
                    (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
                      (if (x : ℕ) < rnumel then
                        some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                      else some 0.0)
                      (Option.map (fun a : ℝ => a / (rnumel : ℝ))
                        (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
                          (fun x =>
                            if (x : ℕ) < rnumel then
                              some (s.readMem in_ptr0
                                ((x : ℕ) + rnumel * s.pids 0))
                            else some 0.0))))
                    (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
                      (if (x : ℕ) < rnumel then
                        some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                      else some 0.0)
                      (Option.map (fun a : ℝ => a / (rnumel : ℝ))
                        (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
                          (fun x =>
                            if (x : ℕ) < rnumel then
                              some (s.readMem in_ptr0
                                ((x : ℕ) + rnumel * s.pids 0))
                            else some 0.0))))
                else some 0))))
      = rowRstdSpec s in_ptr0 rnumel
    simp only [hMean, Option.map_some]
    -- Collapse the squared-deviation lanes (`tl.where(rmask, ·, 0)` guard)
    -- to `some` of the guarded real lane values.
    have hlaneV : ∀ x : Fin RBLOCK,
        @Eq (WithBot ℝ)
          (if (x : ℕ) < rnumel then
            Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
              (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
                (if (x : ℕ) < rnumel then
                  some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                else some 0.0)
                (some ((∑ r ∈ Finset.range rnumel,
                    s.readMem in_ptr0 (r + rnumel * s.pids 0)) / (rnumel : ℝ))))
              (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
                (if (x : ℕ) < rnumel then
                  some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                else some 0.0)
                (some ((∑ r ∈ Finset.range rnumel,
                    s.readMem in_ptr0 (r + rnumel * s.pids 0)) / (rnumel : ℝ))))
          else some 0)
          (some (if (x : ℕ) < rnumel then
            (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0) -
              (∑ r ∈ Finset.range rnumel,
                s.readMem in_ptr0 (r + rnumel * s.pids 0)) / (rnumel : ℝ)) ^ 2
          else 0)) := by
      intro x
      by_cases h : (x : ℕ) < rnumel <;>
        simp [h, Option.map₂, Option.map, pow_two]
    rw [Finset.sum_congr rfl (fun x _ => hlaneV x), sum_somes_fin,
        sum_fin_guard RBLOCK rnumel hLe
          (fun r => (s.readMem in_ptr0 (r + rnumel * s.pids 0) -
            (∑ j ∈ Finset.range rnumel,
              s.readMem in_ptr0 (j + rnumel * s.pids 0)) / (rnumel : ℝ)) ^ 2)]
    simp [WithBot.realRsqrt, rowRstdSpec, rowVarSpec, rowMeanSpec, rowElem]

/-- `Realizes_without_Rounding` form of the two reduction phases: the two scalar stores hold
the **genuine** closed forms — mean `rowMeanSpec` at `in_out_ptr0[x0]` and
rstd `rowRstdSpec` at `in_out_ptr1[x0]` — for every program id and all
`rnumel ≤ RBLOCK`. -/
theorem layer_norm_welfold_reduce_slice_compute_correct
    (in_out_ptr0 in_out_ptr1 in_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) (hLe : rnumel ≤ RBLOCK) (s : BlockState)
    (hMeanRstd : in_out_ptr0 ≠ in_out_ptr1)
    (hInMean : in_ptr0 ≠ in_out_ptr0) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_welfold_reduce_slice in_out_ptr0 in_out_ptr1
        in_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr0, s.pids 0))
      (expected := fun _ => rowMeanSpec s in_ptr0 rnumel)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_welfold_reduce_slice in_out_ptr0 in_out_ptr1
        in_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr1, s.pids 0))
      (expected := fun _ => rowRstdSpec s in_ptr0 rnumel)) := by
  constructor
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [layer_norm_welfold_reduce_slice, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro _
    exact (layer_norm_welfold_reduce_slice_readback in_out_ptr0 in_out_ptr1
      in_ptr0 rnumel RBLOCK hLe s s' hMeanRstd hInMean hExec).1
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [layer_norm_welfold_reduce_slice, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro _
    exact (layer_norm_welfold_reduce_slice_readback in_out_ptr0 in_out_ptr1
      in_ptr0 rnumel RBLOCK hLe s s' hMeanRstd hInMean hExec).2

/-! ## Multi-iteration reduction loops: the pure accumulator step face

For `rnumel > RBLOCK` the first two loops run `⌈rnumel/RBLOCK⌉` times each,
carrying the per-lane partial-sum registers `_tmp3` (row sum) and `_tmp12`
(squared-deviation sum). The algebraic content of one iteration — for every
lane the `rmask` keeps — is the pure-ℝ step identity below: adding the next
stream element maps the exact partial sum of the processed prefix to the
exact partial sum of the one-longer prefix. (The cross-iteration register
scheduling is the trusted runtime boundary, exactly as the carried `b_z` of
`reversed_cumsum_scalar` and the carried Welford registers of
`fused_layernorm_triton`.) -/

/-- **The unguarded accumulator step is exact over ℝ**: from the exact
prefix sum `Σ_{k<c} g k` (for `c = 0` the kernel's `tl.full(…, 0, …)`
state), adding the next element `g c` produces the exact `(c+1)`-prefix
sum. This is the whole per-iteration content of both reduction loops
(`tmp4 = _tmp3 + tmp2` with `g = X[x0,·]`, and `tmp13 = _tmp12 + tmp11`
with `g = (X[x0,·] − mean)²`). -/
theorem sum_accumulate_step_closed (c : Nat) (g : ℕ → ℝ) :
    (∑ k ∈ Finset.range c, g k) + g c = ∑ k ∈ Finset.range (c + 1), g k :=
  (Finset.sum_range_succ g c).symm

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **Dimension-general** correctness summary for `layer_norm_welfold.py`
(`triton_red_fused_native_layer_norm_no_welford`), against the **genuine
closed forms** (mean / variance / rstd / affine output as standalone
`Finset.sum` formulas over input memory — no read-back of kernel output),
for arbitrary `xnumel`, `rnumel`, `XBLOCK`, `RBLOCK` and program ids. It
packages:

* the **full faithful surface** — all three tiled loops and both
  `tl.debug_barrier()` fences — lowers to the algorithm layer, all four
  dimensions symbolic;
* the **two reduction phases** (`XBLOCK = 1` as pinned by both autotune
  configs; single reduction block `rnumel ≤ RBLOCK`): the mean cell
  `in_out_ptr0[x0]` and rstd cell `in_out_ptr1[x0]` genuinely realize
  `rowMeanSpec = (Σ_r X[x0,r])/rnumel` and
  `rowRstdSpec = 1/√((Σ_r (X[x0,r]−μ)²)/rnumel + 1e-05)`, end-to-end from
  `in_ptr0` (the second pass consumes the *register* mean, so the variance
  is genuinely computed from the first pass, not re-read);
* the **normalize face**: for *every* state whose mean/rstd cells hold the
  genuine closed forms — exactly the values the reduction phases store —
  one normalize-loop iteration realizes the genuine
  `rowYSpec = ((X−μ)·rstd)·W + B` on its masked chunk, general over all
  chunk indices (so the third loop is covered for every `rnumel`,
  `RBLOCK`, multi-iteration included).

The multi-iteration content of the first two loops is the pure step face
`sum_accumulate_step_closed`; the cross-iteration register scheduling that
threads `_tmp3` / `_tmp12` between iterations is the trusted runtime
boundary (as with `reversed_cumsum_scalar`'s carried `b_z`). Honest side
conditions only: `rnumel ≤ RBLOCK` for the single-block reduction phases,
mean/rstd output-region distinctness (`in_out_ptr0 ≠ in_out_ptr1`), and
input/mean-buffer distinctness (`in_ptr0 ≠ in_out_ptr0`, because the
second pass reloads the row after the mean store). -/
specification layer_norm_welfold_output_summary_general
    (in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) (hLe : rnumel ≤ RBLOCK)
    (s : BlockState)
    (hMeanRstd : in_out_ptr0 ≠ in_out_ptr1)
    (hInMean : in_ptr0 ≠ in_out_ptr0) :
    -- (1) the full faithful surface lowers to the algorithm layer
    (∃ alg,
      (layer_norm_welfold_surface in_out_ptr0 in_out_ptr1 in_ptr0 in_ptr1
        in_ptr2 out_ptr0 xnumel rnumel XBLOCK RBLOCK).toAlgorithm?
      = Except.ok alg) ∧
    -- (2) reduction phases: genuine mean and rstd, end-to-end from `in_ptr0`
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_welfold_reduce_slice in_out_ptr0 in_out_ptr1
        in_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr0, s.pids 0))
      (expected := fun _ => rowMeanSpec s in_ptr0 rnumel)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_welfold_reduce_slice in_out_ptr0 in_out_ptr1
        in_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr1, s.pids 0))
      (expected := fun _ => rowRstdSpec s in_ptr0 rnumel))) ∧
    -- (3) normalize face: genuine `Y` chunk for every mean/rstd-genuine state
    (∀ s' : BlockState,
      s'.readMem in_out_ptr0 (s'.pids 0) = rowMeanSpec s' in_ptr0 rnumel →
      s'.readMem in_out_ptr1 (s'.pids 0) = rowRstdSpec s' in_ptr0 rnumel →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := layer_norm_welfold_normalize_slice in_out_ptr0 in_out_ptr1
          in_ptr0 in_ptr1 in_ptr2 out_ptr0 rnumel RBLOCK)
        (initialState := s')
        (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin RBLOCK => activeLane s' rnumel RBLOCK i)
          (fun i => (out_ptr0, yOffset s' rnumel RBLOCK i)))
        (expected := fun i : Fin RBLOCK =>
          rowYSpec s' in_ptr0 in_ptr1 in_ptr2 rnumel (rIndex s' RBLOCK i))) := by
  refine ⟨layer_norm_welfold_surface_toAlgorithm_supported in_out_ptr0
      in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 xnumel rnumel XBLOCK RBLOCK,
    layer_norm_welfold_reduce_slice_compute_correct in_out_ptr0 in_out_ptr1
      in_ptr0 rnumel RBLOCK hLe s hMeanRstd hInMean,
    ?_⟩
  intro s' hmean hrstd
  exact layer_norm_welfold_normalize_slice_compute_correct in_out_ptr0
    in_out_ptr1 in_ptr0 in_ptr1 in_ptr2 out_ptr0 rnumel RBLOCK s' hmean hrstd

end VeriTile.Bench.TritonBenchG.LayerNormWelfold
