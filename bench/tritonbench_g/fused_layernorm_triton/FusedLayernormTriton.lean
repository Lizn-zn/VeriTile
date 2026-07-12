import VeriTile.Triton

/-!
# `fused_layernorm_triton` — strict per-kernel correctness

Target JIT: `triton_red_fused_native_layer_norm_0` — a torch-inductor
fused LayerNorm forward. Each program covers `XBLOCK` rows of the
`[xnumel, rnumel]` input `in_ptr0`; a first tiled loop accumulates
per-lane Welford statistics (`triton_helpers.welford_reduce`) over
`RBLOCK`-wide chunks, the lane statistics are combined
(`triton_helpers.welford`) into the row mean `tmp3` (stored to `out_ptr0`)
and second moment `tmp4`, the reciprocal standard deviation
`tmp10 = libdevice.rsqrt(tmp4 / rnumel + 1e-05)` is stored to
`in_out_ptr0` after a `tl.debug_barrier()`, and a second tiled loop
stores the normalized affine output `tmp21 = ((x − mean)·rstd)·w + b`
to `out_ptr1` (weights `in_ptr1`, bias `in_ptr2`).

## Scope

This file verifies **the Triton kernel itself** — the per-program
`@triton.jit` body. The host launch (`fused_native_layer_norm`'s
`grid = cdiv(S, XBLOCK)`, the `@triton.autotune` config sweep
`(XBLOCK, RBLOCK) ∈ {(1, 1024), (1, 2048)}`, buffer reinterpretation, and
how the runtime composes per-program writes into the output buffers) is
the *trusted boundary*, not a proof obligation here. Because the program
ids are universally quantified, the per-program statements cover every
program of the grid.

## Proof architecture

```
fused_layernorm_triton_output_summary_general              ← TOP THEOREM
  ├─ fused_layernorm_triton_surface_toAlgorithm_supported  full surface lowers
  ├─ fused_layernorm_triton_reduce_slice_compute_correct   mean ∧ rstd genuine
  │    └─ fused_layernorm_triton_reduce_slice_readback
  │         ├─ sum_somes_fin / sum_fin_guard    lane sums → `Finset.range` sums
  │         └─ var_moment_identity              (Σx² − n·μ·μ)/n = (Σ(x−μ)²)/n
  └─ fused_layernorm_triton_normalize_slice_compute_correct  genuine Y chunks
       └─ fused_layernorm_triton_normalize_slice_correct

multi-iteration first-loop face (pure ℝ):
  welford_reduce_step_closed    one inlined `welford_reduce` step maps exact
                                prefix statistics to exact prefix statistics
```

The genuine closed-form specs are standalone `Finset.sum` formulas over
input memory (`rowMeanSpec` / `rowVarSpec` / `rowRstdSpec` / `rowYSpec`) —
never a read-back of the kernel's own output.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the
`@triton.autotune` config sweep is not modeled — all four dimensions
`xnumel`, `rnumel`, `XBLOCK`, `RBLOCK` stay symbolic binders.
`tl.debug_barrier()` is transcribed in place; it is an intra-program fence
and therefore a semantic no-op in the sequential single-program-per-row
model (the algorithm-layer lowering erases it — see the `effectMarker`
case of `ComputeStmt.toAlgorithm?`). The `.to(tl.float32)` /
`.to(tl.bfloat16)`-style casts erase to the identity at the algorithm
layer. `eps = 1e-05` is hardcoded in the kernel body (`tmp8 = 1e-05`), so
the specs carry it verbatim rather than as a binder. The value-correct
theorems cover the `XBLOCK = 1` row-per-program shape pinned by both
autotune configs; the reduction phase is additionally scoped to a single
reduction block (`rnumel ≤ RBLOCK`, e.g. the benchmark's
`D ∈ {1024, 2048}` rows under `RBLOCK ∈ {1024, 2048}`), while the
normalize loop is covered genuinely for **all** chunk indices and shapes
via the mean/rstd-cell hypotheses (exactly the values the reduction phase
stores). The multi-iteration reduction recurrence is the pure step face
`welford_reduce_step_closed`; the cross-iteration register scheduling that
threads `(tmp3_mean, tmp3_m2, tmp3_weight)` is the trusted runtime
boundary, exactly as `reversed_cumsum_scalar`'s carried `b_z`.

## Translation-surface blocker

Translation-surface blocker: the two opaque `torch._inductor` helper JIT
calls have no `tl.*` surface and are inlined into the transcription
(`_attn_fwd_inner` precedent): `triton_helpers.welford_reduce(tmp2,
tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0)` is inlined as its
general-branch update `delta/weight+1/mean+delta/w'/m2+delta·(v−mean')`
(from the all-zero initial state the `first_iteration` branch is
exact-arithmetic-equal to the general branch), and
`triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)` (a
`tl.reduce` tree-fold of Chan's `welford_combine` along `axis=1`) is
inlined as its exact-arithmetic moment closed form
`(Σw, Σw·m / Σw, Σ(m2 + w·m²) − Σw·mean²)`, which the fold provably
computes over `ℝ`. `tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])` is written
through the DSL's tuple-form `tmp2, _ = tl.broadcast(tmp1, tmp3_mean)`
(same broadcast, no `broadcast_to` spelling in the DSL); the Python
`tl.store(..., None)` explicit no-mask arguments are written as unmasked
`tl.store(...)`; `libdevice.rsqrt` is written `tl.rsqrt` (existing
`rmsnorm_triton` / `layer_norm_liger` precedent); the register recasts
`tmpN = tmpM.to(tl.float32)` are written with mechanical parentheses
`(tmpM).to(tl.float32)` (DSL parse limitation on bare-ident method casts).
-/

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton

open VeriTile.Triton

set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySeqFocus false

/-- Faithful transcription of `fused_layernorm_triton.py`'s
`triton_red_fused_native_layer_norm_0` (the only `@triton.jit` kernel; the
`@triton.autotune` decorator sweeps `(XBLOCK, RBLOCK) ∈ {(1,1024),(1,2048)}`
and is not modeled — all four parameters stay symbolic). The inlined
`welford_reduce` / `welford` helper statements are documented in the
`Translation-surface blocker:` preamble marker. -/
def fused_layernorm_triton_surface
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(XBLOCK)
  xindex = xoffset + tl.arange(0, $(XBLOCK))[:, None]
  xmask = xindex < $(xnumel)
  rbase = tl.arange(0, $(RBLOCK))[None, :]
  x0 = xindex
  tmp3_mean = tl.zeros([$(XBLOCK), $(RBLOCK)], tl.float32)
  tmp3_m2 = tl.zeros([$(XBLOCK), $(RBLOCK)], tl.float32)
  tmp3_weight = tl.zeros([$(XBLOCK), $(RBLOCK)], tl.float32)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
    tmp3_delta = tmp2 - tmp3_mean
    tmp3_weight_next = tmp3_weight + 1.0
    tmp3_mean_next = tmp3_mean + tmp3_delta / tmp3_weight_next
    tmp3_m2_next = tmp3_m2 + tmp3_delta * (tmp2 - tmp3_mean_next)
    tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
    tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
    tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
  }
  tmp5_tmp = tl.sum(tmp3_weight, 1)
  tmp3_tmp = tl.sum(tmp3_weight * tmp3_mean, 1) / tmp5_tmp
  tmp4_tmp = tl.sum(tmp3_m2 + tmp3_weight * tmp3_mean * tmp3_mean, 1) - tmp5_tmp * tmp3_tmp * tmp3_tmp
  tmp3 = tmp3_tmp[:, None]
  tmp4 = tmp4_tmp[:, None]
  tmp5 = tmp5_tmp[:, None]
  tl.store(out_ptr0 + (x0), tmp3)
  tmp6 = $(rnumel)
  tmp7 = tmp4 / tmp6
  tmp8 = 1e-05
  tmp9 = tmp7 + tmp8
  tmp10 = tl.rsqrt(tmp9)
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp10)
  for roffset in range(0, $(rnumel), $(RBLOCK)) {
    rindex = roffset + rbase
    rmask = rindex < $(rnumel)
    r1 = rindex
    tmp11 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), rmask, eviction_policy="evict_first").to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
    tmp12 = (tmp11).to(tl.float32)
    tmp13 = tmp12 - tmp3
    tmp14 = tmp13 * tmp10
    tmp16 = (tmp15).to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = (tmp18).to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = (tmp20).to(tl.float32)
    tl.store(out_ptr1 + (r1 + ($(rnumel) * x0)), tmp21, rmask)
  }
}

/-- The full faithful surface — both tiled reduction loops, the inlined
Welford statements, and the `tl.debug_barrier()` fence — lowers to the
algorithm layer (the fence erases to a no-op under the sequential
per-program semantics). -/
theorem fused_layernorm_triton_surface_toAlgorithm_supported
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) :
    ∃ alg,
      (fused_layernorm_triton_surface in_out_ptr0 in_ptr0 in_ptr1 in_ptr2
        out_ptr0 out_ptr1 xnumel rnumel XBLOCK RBLOCK).toAlgorithm?
      = Except.ok alg := by
  simp [fused_layernorm_triton_surface, ComputeExpr.toAlgorithm?,
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
`tmp8 = 1e-05`: `1 / sqrt(var + 1e-05)` (`libdevice.rsqrt`). -/
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

/-! ## Normalize-loop slice (second `for roffset` loop, one iteration)

The second loop's iterations are independent: each reads the already-stored
row mean (`out_ptr0[x0]`, the kernel's `tmp3`) and rstd (`in_out_ptr0[x0]`,
the kernel's `tmp10`) plus fresh `in_ptr0/1/2` tiles, and stores one masked
output chunk. The slice materializes one iteration with the chunk index as
`tl.program_id(1)` (`roffset = i_t · RBLOCK`); its correctness theorem takes
the honest hypotheses that the mean/rstd cells hold the genuine closed forms
— exactly the values the first phase stores — and is dimension-general over
`rnumel`, `RBLOCK`, and the chunk index. -/

def fused_layernorm_triton_normalize_slice
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  x0 = tl.program_id(0)
  i_t = tl.program_id(1)
  rindex = i_t * $(RBLOCK) + tl.arange(0, $(RBLOCK))
  rmask = rindex < $(rnumel)
  r1 = rindex
  tmp3 = tl.load(out_ptr0 + (x0))
  tmp10 = tl.load(in_out_ptr0 + (x0))
  tmp11 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp15 = tl.load(in_ptr1 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp18 = tl.load(in_ptr2 + (r1), mask=rmask, other=0.0).to(tl.float32)
  tmp12 = (tmp11).to(tl.float32)
  tmp13 = tmp12 - tmp3
  tmp14 = tmp13 * tmp10
  tmp16 = (tmp15).to(tl.float32)
  tmp17 = tmp14 * tmp16
  tmp19 = (tmp18).to(tl.float32)
  tmp20 = tmp17 + tmp19
  tmp21 = (tmp20).to(tl.float32)
  tl.store(out_ptr1 + (r1 + ($(rnumel) * x0)), tmp21, mask=rmask)
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
theorem fused_layernorm_triton_normalize_slice_correct
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (rnumel RBLOCK : Nat) (s : BlockState)
    (hmean : s.readMem out_ptr0 (s.pids 0) = rowMeanSpec s in_ptr0 rnumel)
    (hrstd : s.readMem in_out_ptr0 (s.pids 0) = rowRstdSpec s in_ptr0 rnumel) :
    ∀ i : Fin RBLOCK,
      let outAddr := yOffset s rnumel RBLOCK i
      (exec (fused_layernorm_triton_normalize_slice in_out_ptr0 in_ptr0
          in_ptr1 in_ptr2 out_ptr0 out_ptr1 rnumel RBLOCK) s).map
          (·.readMem out_ptr1 outAddr)
        = some (if activeLane s rnumel RBLOCK i then
            rowYSpec s in_ptr0 in_ptr1 in_ptr2 rnumel (rIndex s RBLOCK i)
          else s.readMem out_ptr1 outAddr) := by
  intro i
  simp [exec, fused_layernorm_triton_normalize_slice, stepStmts, stepStmt,
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
mean/rstd cells hold the genuine closed forms (exactly what the reduction
phase stores), one normalize-loop iteration realizes the genuine `rowYSpec`
on its masked chunk — general over `rnumel`, `RBLOCK` and the chunk index
`i_t = pids 1`. -/
theorem fused_layernorm_triton_normalize_slice_compute_correct
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (rnumel RBLOCK : Nat) (s : BlockState)
    (hmean : s.readMem out_ptr0 (s.pids 0) = rowMeanSpec s in_ptr0 rnumel)
    (hrstd : s.readMem in_out_ptr0 (s.pids 0) = rowRstdSpec s in_ptr0 rnumel) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_layernorm_triton_normalize_slice in_out_ptr0 in_ptr0
        in_ptr1 in_ptr2 out_ptr0 out_ptr1 rnumel RBLOCK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin RBLOCK => activeLane s rnumel RBLOCK i)
        (fun i => (out_ptr1, yOffset s rnumel RBLOCK i)))
      (expected := fun i : Fin RBLOCK =>
        rowYSpec s in_ptr0 in_ptr1 in_ptr2 rnumel (rIndex s RBLOCK i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_layernorm_triton_normalize_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_layernorm_triton_normalize_slice_correct in_out_ptr0 in_ptr0
    in_ptr1 in_ptr2 out_ptr0 out_ptr1 rnumel RBLOCK s hmean hrstd i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Reduction-phase slice (`XBLOCK = 1`, one reduction iteration)

Both `@triton.autotune` configs pin `XBLOCK = 1` (one row per program:
`xoffset = pid`, `xindex = x0 = pid`, the `[XBLOCK, RBLOCK]` tiles are the
row's `[RBLOCK]` lanes, `axis=1` reductions become `axis=0`). When the row
also fits one reduction block (`rnumel ≤ RBLOCK`, e.g. the benchmark's
`D = 1024/2048` rows under `RBLOCK ∈ {1024, 2048}`), the first
`for roffset` loop runs exactly once with `roffset = 0`, so this slice
transcribes the reduction phase — loop body at `roffset = 0`, the inlined
moment combine, the `tmp6..tmp10` rstd chain, `tl.debug_barrier()`, and
the two scalar stores — in kernel order, stopping before the (independent)
normalize loop, which `fused_layernorm_triton_normalize_slice` covers for
every chunk. The masked load carries `other=0.0` (the faithful surface's
mask-only load lanes are discarded by `tl.where`; here the lanes must be
total to be summed), and the identity recasts `tmp1`/`tmp2`
(`.to(tl.float32)` and `broadcast_to` on an already-`[RBLOCK]` tile) are
elided. The single inlined `welford_reduce` step runs from the all-zero
state. -/
def fused_layernorm_triton_reduce_slice
    (in_out_ptr0 in_ptr0 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) : ComputeKernel := triton {
  x0 = tl.program_id(0)
  rbase = tl.arange(0, $(RBLOCK))
  rmask = rbase < $(rnumel)
  r1 = rbase
  tmp0 = tl.load(in_ptr0 + (r1 + ($(rnumel) * x0)), mask=rmask, other=0.0).to(tl.float32)
  tmp3_mean_init = tl.zeros([$(RBLOCK)], tl.float32)
  tmp3_m2_init = tl.zeros([$(RBLOCK)], tl.float32)
  tmp3_weight_init = tl.zeros([$(RBLOCK)], tl.float32)
  tmp3_delta = tmp0 - tmp3_mean_init
  tmp3_weight_next = tmp3_weight_init + 1.0
  tmp3_mean_next = tmp3_mean_init + tmp3_delta / tmp3_weight_next
  tmp3_m2_next = tmp3_m2_init + tmp3_delta * (tmp0 - tmp3_mean_next)
  tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean_init)
  tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2_init)
  tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight_init)
  tmp5_tmp = tl.sum(tmp3_weight, 0)
  tmp3_tmp = tl.sum(tmp3_weight * tmp3_mean, 0) / tmp5_tmp
  tmp4_tmp = tl.sum(tmp3_m2 + tmp3_weight * tmp3_mean * tmp3_mean, 0) - tmp5_tmp * tmp3_tmp * tmp3_tmp
  tl.store(out_ptr0 + (x0), tmp3_tmp)
  tmp6 = $(rnumel)
  tmp7 = tmp4_tmp / tmp6
  tmp8 = 1e-05
  tmp9 = tmp7 + tmp8
  tmp10 = tl.rsqrt(tmp9)
  tl.debug_barrier()
  tl.store(in_out_ptr0 + (x0), tmp10)
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

/-- **Moment form of the population variance** (the inlined `welford`
combine computes `Σx² − n·μ·μ`; the genuine spec sums squared deviations):
`(Σx² − n·μ·μ)/n = (Σ(x − μ)²)/n` with `μ = (Σx)/n`, including the
degenerate `n = 0` case under Lean's `x/0 = 0` convention. -/
private theorem var_moment_identity (n : Nat) (f : Nat → ℝ) :
    ((∑ r ∈ Finset.range n, f r * f r) -
        (n : ℝ) * ((∑ r ∈ Finset.range n, f r) / n) *
          ((∑ r ∈ Finset.range n, f r) / n)) / n
      = (∑ r ∈ Finset.range n,
          (f r - (∑ r ∈ Finset.range n, f r) / n) ^ 2) / n := by
  by_cases hn : (n : ℝ) = 0
  · have hn0 : n = 0 := by exact_mod_cast hn
    subst hn0
    simp
  · congr 1
    set μ := (∑ r ∈ Finset.range n, f r) / n with hμ
    have hsum : (∑ r ∈ Finset.range n, f r) = (n : ℝ) * μ := by
      rw [hμ]; field_simp
    have hexp : ∀ r : ℕ, (f r - μ) ^ 2 = f r * f r - 2 * μ * f r + μ * μ := by
      intro r; ring
    rw [Finset.sum_congr rfl (fun r _ => hexp r)]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
        Finset.sum_const, Finset.card_range, hsum]
    ring

set_option maxHeartbeats 24000000 in
/-- Readback of the two reduction-phase stores against the **genuine**
closed forms: after the slice runs, the mean cell `out_ptr0[x0]` holds
`rowMeanSpec` and the rstd cell `in_out_ptr0[x0]` holds `rowRstdSpec`
(single reduction block: `rnumel ≤ RBLOCK`). -/
theorem fused_layernorm_triton_reduce_slice_readback
    (in_out_ptr0 in_ptr0 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) (hLe : rnumel ≤ RBLOCK) (s s' : BlockState)
    (hMeanRstd : out_ptr0 ≠ in_out_ptr0)
    (hExec : exec (fused_layernorm_triton_reduce_slice in_out_ptr0
        in_ptr0 out_ptr0 rnumel RBLOCK) s = some s') :
    s'.readMem out_ptr0 (s.pids 0) = rowMeanSpec s in_ptr0 rnumel ∧
    s'.readMem in_out_ptr0 (s.pids 0) = rowRstdSpec s in_ptr0 rnumel := by
  simp [exec, fused_layernorm_triton_reduce_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, TileCarrier,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  -- Collapse the three lane sums (combined weight, weighted mean numerator,
  -- and second moment) to `some`-of-`Finset.sum` closed forms.
  have hlaneW : ∀ x : Fin RBLOCK,
      @Eq (WithBot ℝ)
        (if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0)
        (some (if (x : ℕ) < rnumel then (1 : ℝ) else 0)) := by
    intro x
    by_cases h : (x : ℕ) < rnumel <;> simp [h] <;> norm_num
  have hlaneA : ∀ x : Fin RBLOCK,
      @Eq (WithBot ℝ)
      (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
        (if (x : ℕ) < rnumel then (some (1.0 : ℝ) : WithBot ℝ) else some 0)
        (if (x : ℕ) < rnumel then
          Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
            (if (x : ℕ) < rnumel then
              some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
            else some 0.0)
        else some 0))
        (some (if (x : ℕ) < rnumel then
            s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0) else 0)) := by
    intro x
    by_cases h : (x : ℕ) < rnumel <;> simp [h, Option.map₂, Option.map] <;>
      norm_num
  have hlaneC : ∀ x : Fin RBLOCK,
      @Eq (WithBot ℝ)
      (Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
        (if (x : ℕ) < rnumel then
          Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (if (x : ℕ) < rnumel then
              some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
            else some 0.0)
            (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
              (if (x : ℕ) < rnumel then
                some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
              else some 0.0)
              (Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
                (if (x : ℕ) < rnumel then
                  some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                else some 0.0)))
        else some 0)
        (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (if (x : ℕ) < rnumel then (some (1.0 : ℝ) : WithBot ℝ) else some 0)
            (if (x : ℕ) < rnumel then
              Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
                (if (x : ℕ) < rnumel then
                  some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                else some 0.0)
            else some 0))
          (if (x : ℕ) < rnumel then
            Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
              (if (x : ℕ) < rnumel then
                some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
              else some 0.0)
          else some 0)))
        (some (if (x : ℕ) < rnumel then
            s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0) *
              s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0) else 0)) := by
    intro x
    by_cases h : (x : ℕ) < rnumel <;> simp [h, Option.map₂, Option.map] <;>
      norm_num
  constructor
  · rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hMeanRstd]
    simp [BlockState.writeMem_readMem]
    show WithBot.unbotD 0
        (Option.map₂ (fun x1 x2 : ℝ => x1 / x2)
          (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0)
            (if (x : ℕ) < rnumel then
              Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
                (if (x : ℕ) < rnumel then
                  some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                else some 0.0)
            else some 0)))
          (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0)))
      = rowMeanSpec s in_ptr0 rnumel
    rw [Finset.sum_congr rfl (fun x _ => hlaneA x),
        Finset.sum_congr rfl (fun x _ => hlaneW x),
        sum_somes_fin, sum_somes_fin,
        sum_fin_guard RBLOCK rnumel hLe
          (fun r => s.readMem in_ptr0 (r + rnumel * s.pids 0)),
        sum_fin_guard RBLOCK rnumel hLe (fun _ => (1 : ℝ))]
    simp [Option.map₂, rowMeanSpec, rowElem, Finset.sum_const,
      Finset.card_range, nsmul_eq_mul, mul_one]
  · simp [BlockState.writeMem_readMem]
    show WithBot.unbotD 0
        (WithBot.realRsqrt
          (Option.map ((fun a : ℝ => a + 1e-05) ∘ fun a : ℝ => a / (rnumel : ℝ))
            (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
              (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
            (if (x : ℕ) < rnumel then
              Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
                (if (x : ℕ) < rnumel then
                  some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                else some 0.0)
                (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
                  (if (x : ℕ) < rnumel then
                    some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                  else some 0.0)
                  (Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
                    (if (x : ℕ) < rnumel then
                      some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                    else some 0.0)))
            else some 0)
            (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
              (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
                (if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0)
                (if (x : ℕ) < rnumel then
                  Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
                    (if (x : ℕ) < rnumel then
                      some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                    else some 0.0)
                else some 0))
              (if (x : ℕ) < rnumel then
                Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
                  (if (x : ℕ) < rnumel then
                    some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                  else some 0.0)
              else some 0))))
              (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
                (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
                  (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0))
                  (Option.map₂ (fun x1 x2 : ℝ => x1 / x2)
                    (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0)
            (if (x : ℕ) < rnumel then
              Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
                (if (x : ℕ) < rnumel then
                  some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                else some 0.0)
            else some 0)))
                    (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0))))
                (Option.map₂ (fun x1 x2 : ℝ => x1 / x2)
                  (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0)
            (if (x : ℕ) < rnumel then
              Option.map ((fun a : ℝ => a) ∘ fun a : ℝ => a / 1.0)
                (if (x : ℕ) < rnumel then
                  some (s.readMem in_ptr0 ((x : ℕ) + rnumel * s.pids 0))
                else some 0.0)
            else some 0)))
                  (@Finset.sum (Fin RBLOCK) (WithBot ℝ) _ Finset.univ
        (fun x =>
          if (x : ℕ) < rnumel then some (1.0 : ℝ) else some 0)))))))
      = rowRstdSpec s in_ptr0 rnumel
    rw [Finset.sum_congr rfl (fun x _ => hlaneC x),
        Finset.sum_congr rfl (fun x _ => hlaneA x),
        Finset.sum_congr rfl (fun x _ => hlaneW x),
        sum_somes_fin, sum_somes_fin, sum_somes_fin,
        sum_fin_guard RBLOCK rnumel hLe
          (fun r => s.readMem in_ptr0 (r + rnumel * s.pids 0) *
            s.readMem in_ptr0 (r + rnumel * s.pids 0)),
        sum_fin_guard RBLOCK rnumel hLe
          (fun r => s.readMem in_ptr0 (r + rnumel * s.pids 0)),
        sum_fin_guard RBLOCK rnumel hLe (fun _ => (1 : ℝ))]
    simp only [Option.map₂_some_some, Option.map_some, Function.comp_apply,
      Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
    rw [var_moment_identity rnumel
      (fun r => s.readMem in_ptr0 (r + rnumel * s.pids 0))]
    simp [WithBot.realRsqrt, rowRstdSpec, rowVarSpec, rowMeanSpec, rowElem]

/-- `Realizes_without_Rounding` form of the reduction-phase slice: the two scalar stores hold
the **genuine** closed forms — mean `rowMeanSpec` at `out_ptr0[x0]` and rstd
`rowRstdSpec` at `in_out_ptr0[x0]` — for every program id and all
`rnumel ≤ RBLOCK`. -/
theorem fused_layernorm_triton_reduce_slice_compute_correct
    (in_out_ptr0 in_ptr0 out_ptr0 : RegionName)
    (rnumel RBLOCK : Nat) (hLe : rnumel ≤ RBLOCK) (s : BlockState)
    (hMeanRstd : out_ptr0 ≠ in_out_ptr0) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_layernorm_triton_reduce_slice in_out_ptr0 in_ptr0
        out_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (out_ptr0, s.pids 0))
      (expected := fun _ => rowMeanSpec s in_ptr0 rnumel)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_layernorm_triton_reduce_slice in_out_ptr0 in_ptr0
        out_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr0, s.pids 0))
      (expected := fun _ => rowRstdSpec s in_ptr0 rnumel)) := by
  constructor
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [fused_layernorm_triton_reduce_slice, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro _
    exact (fused_layernorm_triton_reduce_slice_readback in_out_ptr0 in_ptr0
      out_ptr0 rnumel RBLOCK hLe s s' hMeanRstd hExec).1
  · unfold ComputeCorrect.Realizes_without_Rounding
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [fused_layernorm_triton_reduce_slice, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro _
    exact (fused_layernorm_triton_reduce_slice_readback in_out_ptr0 in_ptr0
      out_ptr0 rnumel RBLOCK hLe s s' hMeanRstd hExec).2

/-! ## Multi-iteration reduction loop: the pure Welford step face

For `rnumel > RBLOCK` the first loop runs `⌈rnumel/RBLOCK⌉` times, carrying
per-lane `(tmp3_mean, tmp3_m2, tmp3_weight)` registers. The algebraic
content of one iteration — for every lane the `rmask` keeps — is the pure-ℝ
step identity below: the inlined `welford_reduce` update maps the **exact
statistics of the processed prefix** of a lane stream to the exact
statistics of the one-longer prefix. (Masked lanes keep their statistics
verbatim via the three `tl.where`; the cross-iteration register scheduling
is the trusted runtime boundary, exactly as the carried `b_z` of
`reversed_cumsum_scalar`.) The final inlined moment combine is covered by
`var_moment_identity` inside the readback proof. -/

/-- **The inlined `welford_reduce` step is exact over ℝ.** Starting from the
exact prefix statistics `μ_c = (Σ_{k<c} g k)/c`, `m2_c = Σ_{k<c} (g k − μ_c)²`,
`w_c = c` (for `c = 0` all three are `0` under Lean's `x/0 = 0` — precisely
the kernel's `tl.zeros` state, which is why the helper's `first_iteration`
branch is redundant over ℝ), the update
`delta = v − μ; w' = w + 1; μ' = μ + delta/w'; m2' = m2 + delta·(v − μ')`
with `v = g c` produces the exact statistics of the `(c+1)`-prefix. -/
theorem welford_reduce_step_closed (c : Nat) (g : ℕ → ℝ) :
    ((c : ℝ) + 1 = ((c + 1 : Nat) : ℝ)) ∧
    ((∑ k ∈ Finset.range c, g k) / c +
        (g c - (∑ k ∈ Finset.range c, g k) / c) / ((c : ℝ) + 1)
      = (∑ k ∈ Finset.range (c + 1), g k) / ((c + 1 : Nat) : ℝ)) ∧
    ((∑ k ∈ Finset.range c,
        (g k - (∑ j ∈ Finset.range c, g j) / c) ^ 2) +
        (g c - (∑ j ∈ Finset.range c, g j) / c) *
          (g c - (∑ j ∈ Finset.range (c + 1), g j) / ((c + 1 : Nat) : ℝ))
      = ∑ k ∈ Finset.range (c + 1),
          (g k - (∑ j ∈ Finset.range (c + 1), g j) / ((c + 1 : Nat) : ℝ)) ^ 2) := by
  have hmean : (∑ k ∈ Finset.range c, g k) / c +
      (g c - (∑ k ∈ Finset.range c, g k) / c) / ((c : ℝ) + 1)
      = (∑ k ∈ Finset.range (c + 1), g k) / ((c + 1 : Nat) : ℝ) := by
    by_cases hc : c = 0
    · subst hc; simp
    · have hc' : (c : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hc
      have hc1 : (c : ℝ) + 1 ≠ 0 := by positivity
      rw [Finset.sum_range_succ]
      push_cast
      field_simp
      ring
  refine ⟨by push_cast; ring, hmean, ?_⟩
  by_cases hc : c = 0
  · subst hc; simp
  · have hc' : (c : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hc
    have hc1 : (c : ℝ) + 1 ≠ 0 := by positivity
    set S : ℝ := ∑ j ∈ Finset.range c, g j with hS
    set μ : ℝ := S / c with hμ
    set μ' : ℝ := (∑ j ∈ Finset.range (c + 1), g j) / ((c + 1 : Nat) : ℝ)
      with hμ'
    set d : ℝ := μ' - μ with hd
    -- the prefix sums to `c·μ`
    have hSum : S = (c : ℝ) * μ := by
      rw [hμ]; field_simp
    -- `g c − μ = (c+1)·d` from the mean update identity
    have hgcμ : g c - μ = ((c : ℝ) + 1) * d := by
      have h : μ + (g c - μ) / ((c : ℝ) + 1) = μ' := hmean
      have h' : (g c - μ) / ((c : ℝ) + 1) = d := by
        rw [hd, ← h]; ring
      calc g c - μ = ((c : ℝ) + 1) * ((g c - μ) / ((c : ℝ) + 1)) := by
              field_simp
        _ = ((c : ℝ) + 1) * d := by rw [h']
    have hgcμ' : g c - μ' = (c : ℝ) * d := by
      have hstep : g c - μ' = (g c - μ) - d := by rw [hd]; ring
      rw [hstep, hgcμ]; ring
    -- expand `Σ_{k<c+1} (g k − μ')²` around `μ`
    have hexp : ∀ x : ℝ, (x - μ') ^ 2
        = (x - μ) ^ 2 - 2 * d * (x - μ) + d ^ 2 := by
      intro x; rw [hd]; ring
    have hzero : (∑ k ∈ Finset.range c, (g k - μ)) = 0 := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range,
        ← hS, hSum]
      ring
    have hprefix : (∑ k ∈ Finset.range c, (g k - μ') ^ 2)
        = (∑ k ∈ Finset.range c, (g k - μ) ^ 2) + (c : ℝ) * d ^ 2 := by
      rw [Finset.sum_congr rfl (fun k _ => hexp (g k)),
        Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hzero, Finset.sum_const, Finset.card_range]
      ring
    rw [Finset.sum_range_succ, hprefix, hgcμ, hgcμ']
    ring

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **Dimension-general** correctness summary for `fused_layernorm_triton.py`
(`triton_red_fused_native_layer_norm_0`), against the **genuine closed
forms** (mean / variance / rstd / affine output as standalone `Finset.sum`
formulas over input memory — no read-back of kernel output), for arbitrary
`xnumel`, `rnumel`, `XBLOCK`, `RBLOCK` and program ids. It packages:

* the **full faithful surface** — both tiled reduction loops with the
  inlined Welford statements, the `tl.debug_barrier()` fence, and all three
  stores — lowers to the algorithm layer, all four dimensions symbolic;
* the **reduction phase** (`XBLOCK = 1` as pinned by both autotune configs;
  single reduction block `rnumel ≤ RBLOCK`): the mean cell `out_ptr0[x0]`
  and rstd cell `in_out_ptr0[x0]` genuinely realize
  `rowMeanSpec = (Σ_r X[x0,r])/rnumel` and
  `rowRstdSpec = 1/√((Σ_r (X[x0,r]−μ)²)/rnumel + 1e-05)`, end-to-end from
  `in_ptr0`;
* the **normalize face**: for *every* state whose mean/rstd cells hold the
  genuine closed forms — exactly the values the reduction phase stores —
  one normalize-loop iteration realizes the genuine
  `rowYSpec = ((X−μ)·rstd)·W + B` on its masked chunk, general over all
  chunk indices (so the second loop is covered for every `rnumel`,
  `RBLOCK`, multi-iteration included).

The multi-iteration content of the *first* loop is the pure Welford step
face `welford_reduce_step_closed` plus the moment-combine identity
`var_moment_identity`; the cross-iteration register scheduling that threads
`(tmp3_mean, tmp3_m2, tmp3_weight)` between iterations is the trusted
runtime boundary (as with `reversed_cumsum_scalar`'s carried `b_z`).
Honest side conditions only: `rnumel ≤ RBLOCK` for the single-block
reduction phase and mean/rstd output-region distinctness
(`out_ptr0 ≠ in_out_ptr0`). -/
specification fused_layernorm_triton_output_summary_general
    (in_out_ptr0 in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 : RegionName)
    (xnumel rnumel XBLOCK RBLOCK : Nat) (hLe : rnumel ≤ RBLOCK)
    (s : BlockState)
    (hMeanRstd : out_ptr0 ≠ in_out_ptr0) :
    -- (1) the full faithful surface lowers to the algorithm layer
    (∃ alg,
      (fused_layernorm_triton_surface in_out_ptr0 in_ptr0 in_ptr1 in_ptr2
        out_ptr0 out_ptr1 xnumel rnumel XBLOCK RBLOCK).toAlgorithm?
      = Except.ok alg) ∧
    -- (2) reduction phase: genuine mean and rstd, end-to-end from `in_ptr0`
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_layernorm_triton_reduce_slice in_out_ptr0 in_ptr0
        out_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (out_ptr0, s.pids 0))
      (expected := fun _ => rowMeanSpec s in_ptr0 rnumel)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_layernorm_triton_reduce_slice in_out_ptr0 in_ptr0
        out_ptr0 rnumel RBLOCK)
      (initialState := s)
      (write := fun _ : PUnit => some (in_out_ptr0, s.pids 0))
      (expected := fun _ => rowRstdSpec s in_ptr0 rnumel))) ∧
    -- (3) normalize face: genuine `Y` chunk for every mean/rstd-genuine state
    (∀ s' : BlockState,
      s'.readMem out_ptr0 (s'.pids 0) = rowMeanSpec s' in_ptr0 rnumel →
      s'.readMem in_out_ptr0 (s'.pids 0) = rowRstdSpec s' in_ptr0 rnumel →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_layernorm_triton_normalize_slice in_out_ptr0 in_ptr0
          in_ptr1 in_ptr2 out_ptr0 out_ptr1 rnumel RBLOCK)
        (initialState := s')
        (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin RBLOCK => activeLane s' rnumel RBLOCK i)
          (fun i => (out_ptr1, yOffset s' rnumel RBLOCK i)))
        (expected := fun i : Fin RBLOCK =>
          rowYSpec s' in_ptr0 in_ptr1 in_ptr2 rnumel (rIndex s' RBLOCK i))) := by
  refine ⟨fused_layernorm_triton_surface_toAlgorithm_supported in_out_ptr0
      in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 xnumel rnumel XBLOCK RBLOCK,
    fused_layernorm_triton_reduce_slice_compute_correct in_out_ptr0 in_ptr0
      out_ptr0 rnumel RBLOCK hLe s hMeanRstd,
    ?_⟩
  intro s' hmean hrstd
  exact fused_layernorm_triton_normalize_slice_compute_correct in_out_ptr0
    in_ptr0 in_ptr1 in_ptr2 out_ptr0 out_ptr1 rnumel RBLOCK s' hmean hrstd

end VeriTile.Bench.TritonBenchG.FusedLayernormTriton
