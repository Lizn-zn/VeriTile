import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Fused SiLU ≡ three-kernel pipeline — rounding-invariant (boundary) refinement

The **boundary-rounding** sibling of `bench/examples/FusedSiLU.lean` (the exact-ℝ
template). Only the **output** store is rounded to bf16 (`(y).to(tl.bfloat16)`);
the pipeline's scratch materializations (`zReg`/`siluReg`) stay in ℝ and are
excluded from the refinement. Both the fused kernel and the unfused pipeline
compute the **same** per-lane ℝ output `residual + silu(x·gate)`, so any
rounding model `R` quantizes the two equal values identically at the shared bf16
output store — the writes agree outside the scratch temporaries.

Contrast: `bench/examples/FusedSwiglu.lean` rounds the materialized
**intermediate** too (bf16 scratch), needing idempotence; here the scratch is
ℝ, so only the boundary store matters and no idempotence is used.

## The public result (bottom of file)

The single public headline is **`silu_kernels_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines R` (the rounding-model
surface): for the rounding model `R`, from the same state the fused kernel and
the unfused pipeline perform the same writes, agreeing at every cell outside the
declared scratch regions `[zReg, siluReg]`. Its statement mentions only the two
kernels, the loaded-input contract, the rounding-model refinement surface, and
the state/region types — **no spec** (the `#stmtSurfaceSubset` gate enforces
this).
-/

namespace VeriTile.Bench.Examples.FusedSiLURounded

open VeriTile.Triton VeriTile.Examples

/-! ## Kernels -/
section FusedSiLURounded.kernels

/-- Fused SiLU with a bf16-rounded output store. -/
def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store($(outReg) + offsets, (y).to(tl.bfloat16))
}

/-- Step A: materialize `z = x * gate` into the ℝ scratch tensor `z`. -/
def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load($(xReg) + offsets)
  gate    := tl.load($(gateReg) + offsets)
  z       := x * gate
  tl.store($(zReg) + offsets, z)
}

/-- Step B: materialize `silu = z * sigmoid(z)` into the ℝ scratch tensor `silu`. -/
def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load($(zReg) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store($(siluReg) + offsets, silu)
}

/-- Step C: `out = residual + silu`, with a bf16-rounded output store. -/
def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, (y).to(tl.bfloat16))
}

/-- The unfused pipeline as one kernel: `ComputeKernel.seq` of the three step
kernels — the concatenation of their bodies (registers flow across the seams).
The staged execution is recovered by `execR_unfusedSiLU_split` below. -/
def unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGate xReg gateReg zReg blockSize,
     siluStepSilu zReg siluReg blockSize,
     siluStepResidual siluReg residualReg outReg blockSize]

end FusedSiLURounded.kernels

/- Shared parameters of every lemma and the headline — the pipeline's regions in
data-flow order (`keepReg` is the generic frame region), the block size, the
state, the three per-lane input vectors, and the rounding model `R`. -/
variable (xReg gateReg zReg siluReg residualReg outReg keepReg : RegionName)
variable (blockSize : Nat) (s : BlockState)
variable (xs gates residuals : Fin blockSize → ℝ)
variable (R : RoundingModel)

/-! ## Supporting lemmas (private plumbing) -/
section FusedSiLURounded.lemmas

/-- Two `writeMemAsR` scatters over the same offsets agree cell-by-cell when
their per-lane values agree — the rounding-store analogue of
`BlockState.foldl_writeMem_mem_congr`. -/
private theorem foldl_writeMemAsR_mem_congr {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName} (l : List α) (offsetFn : α → Nat)
    (vL vR : α → TileCarrier dtype.toTileDType)
    (hv : ∀ k ∈ l, vL k = vR k) (r : RegionName) (o : Nat) :
    ∀ sL sR : BlockState, sL.mem r o = sR.mem r o →
      (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vL k)) sL).mem r o
        = (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vR k)) sR).mem r o := by
  induction l with
  | nil => intro sL sR h; exact h
  | cons hd tl ih =>
      intro sL sR h
      refine ih (fun k hk => hv k (List.mem_cons_of_mem _ hk)) _ _ ?_
      rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem,
          hv hd List.mem_cons_self]
      by_cases hc : r = region ∧ o = offsetFn hd
      · rw [if_pos hc, if_pos hc]
      · rw [if_neg hc, if_neg hc]; exact h

/-- A `writeMemAsR` scatter `foldl` into `region` leaves every cell of any other
region untouched — the rounding-store analogue of
`BlockState.foldl_writeMem_mem_preserve_other_region`. -/
private theorem foldl_writeMemAsR_preserve_other {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName} (l : List α) (offsetFn : α → Nat)
    (vf : α → TileCarrier dtype.toTileDType) (r : RegionName) (hr : r ≠ region)
    (o : Nat) : ∀ s0 : BlockState,
      (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vf k)) s0).mem r o
        = s0.mem r o := by
  induction l with
  | nil => intro s0; rfl
  | cons hd tl ih =>
      intro s0
      simp only [List.foldl_cons]
      rw [ih, BlockState.writeMemAsR_mem, if_neg (fun hc => hr hc.1)]

/-- The two ℝ scratch-storing step kernels contain no `castFloat`, so they step
identically under `execR R` and `exec` (state universally quantified so the
degeneration rewrites under the `Option.bind` binders of the pipeline split). -/
private theorem execR_siluStepGate (t : BlockState) :
    execR R (siluStepGate xReg gateReg zReg blockSize) t
      = exec (siluStepGate xReg gateReg zReg blockSize) t := by
  simp [execR, exec, siluStepGate, stepStmtsR, stepStmts, stepStmtR, stepStmt,
        evalOpR.eq_def, Tile.bop, NumericDType.add, NumericDType.mul,
        BlockState.writeMemTypedR, ComputeExpr.toAlgorithm?]

private theorem execR_siluStepSilu (t : BlockState) :
    execR R (siluStepSilu zReg siluReg blockSize) t
      = exec (siluStepSilu zReg siluReg blockSize) t := by
  simp [execR, exec, siluStepSilu, stepStmtsR, stepStmts, stepStmtR, stepStmt,
        evalOpR.eq_def, evalOp, Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        BlockState.writeMemTypedR, ComputeExpr.toAlgorithm?]

private theorem stepStmtsR_append (l₁ l₂ : List Stmt) (t : BlockState) :
    stepStmtsR R (l₁ ++ l₂) t = (stepStmtsR R l₁ t).bind (fun t' => stepStmtsR R l₂ t') := by
  induction l₁ generalizing t with
  | nil => simp [stepStmtsR]
  | cons st rest ih =>
      simp only [List.cons_append, stepStmtsR]
      cases stepStmtR R st t <;> simp [ih]

/-- The unfused pipeline's `execR` splits into its three stages run in sequence;
the two ℝ scratch stages degenerate to `exec` (they carry no `castFloat`), while
the final residual stage keeps its bf16-rounded store under `execR R`. -/
private theorem execR_unfusedSiLU_split :
    execR R (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize) s =
      (exec (siluStepGate xReg gateReg zReg blockSize) s).bind (fun s1 =>
        (exec (siluStepSilu zReg siluReg blockSize) s1).bind (fun s2 =>
          execR R (siluStepResidual siluReg residualReg outReg blockSize) s2)) := by
  have hsplit :
      execR R (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize) s =
        (execR R (siluStepGate xReg gateReg zReg blockSize) s).bind (fun s1 =>
          (execR R (siluStepSilu zReg siluReg blockSize) s1).bind (fun s2 =>
            execR R (siluStepResidual siluReg residualReg outReg blockSize) s2)) := by
    simp [unfusedSiLUKernel, execR, stepStmtsR_append]
  rw [hsplit]
  simp only [execR_siluStepGate, execR_siluStepSilu]

/-! ### ℝ scratch-stage facts (reused verbatim from the exact template) -/

private theorem silu_step_gate_correct
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_g : InputLoadedAt s gateReg blockSize gates) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepGate xReg gateReg zReg blockSize) s)
                zReg blockSize s.pid i
        = some (xs i * gates i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStepGate, stepStmts, stepStmt,
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_x _h_g
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_g]

private theorem silu_step_gate_preserves
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_g : InputLoadedAt s gateReg blockSize gates)
    (h_keep : keepReg ≠ zReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepGate xReg gateReg zReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStepGate, stepStmts, stepStmt,
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_x _h_g
  simp_rw [_h_x, _h_g]
  rw [BlockState.scatter_preserves_other_region zReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => xs k.1 * gates k.1)
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

private theorem silu_step_silu_correct (zs : Fin blockSize → ℝ)
    (_h_z : InputLoadedAt s zReg blockSize zs) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepSilu zReg siluReg blockSize) s)
                siluReg blockSize s.pid i
        = some (Triton.TiledActivation.silu (zs i)) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStepSilu, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        Triton.TiledActivation.silu]
  unfold InputLoadedAt at _h_z
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_z]

private theorem silu_step_silu_preserves (zs : Fin blockSize → ℝ)
    (_h_z : InputLoadedAt s zReg blockSize zs)
    (h_keep : keepReg ≠ siluReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepSilu zReg siluReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStepSilu, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_z
  simp_rw [_h_z]
  rw [BlockState.scatter_preserves_other_region siluReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => zs k.1 * Real.sigmoid (zs k.1))
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

private theorem silu_step_gate_mem_frame {s' : BlockState}
    (h : exec (siluStepGate xReg gateReg zReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ zReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, siluStepGate, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

private theorem silu_step_silu_mem_frame {s' : BlockState}
    (h : exec (siluStepSilu zReg siluReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ siluReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, siluStepSilu, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

/-! ### bf16-rounded output-store frames (the two stores that carry a `castFloat`) -/

private theorem fusedR_silu_mem_frame {s' : BlockState}
    (h : execR R (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ outReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [execR, fusedSiLUKernel, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at h
  subst h
  exact foldl_writeMemAsR_preserve_other R .bf16 _ _ _ r hr o _

private theorem siluR_step_residual_mem_frame {s' : BlockState}
    (h : execR R (siluStepResidual siluReg residualReg outReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ outReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [execR, siluStepResidual, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop,
        NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at h
  subst h
  exact foldl_writeMemAsR_preserve_other R .bf16 _ _ _ r hr o _

end FusedSiLURounded.lemmas

/-! ## The headline theorem -/
section FusedSiLURounded.theorems

set_option maxHeartbeats 1600000 in
/-- **fused refines pipeline** (`ComputeRefine.Refines R`): for the rounding
model `R`, from the same initial state the fused kernel and the unfused pipeline
perform the same writes — the two final memories agree at every cell outside the
declared scratch regions `zReg`/`siluReg`. Both compute the same per-lane ℝ
output `residual + silu(x·gate)` and round it at the shared bf16 output store, so
`R` quantizes equal values equally. -/
specification silu_kernels_refinement_view
    (hin : InputsLoaded s blockSize
      [(xReg, xs), (gateReg, gates), (residualReg, residuals)])
    (hscratch : residualReg ∉ [zReg, siluReg]) :
    ComputeRefine.Refines R
      (fusedSiLUKernel xReg gateReg residualReg outReg blockSize)
      (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize)
      s [zReg, siluReg] := by
  obtain ⟨h_x, h_g, h_res, -⟩ := hin
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hscratch
  have h_zRes : zReg ≠ residualReg := fun h => hscratch.1 h.symm
  have h_siluRes : siluReg ≠ residualReg := fun h => hscratch.2 h.symm
  have hx := h_x
  have hg := h_g
  have hres := h_res
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hr
  obtain ⟨hrz, hrsilu⟩ := hr
  have hRsplit := hR
  rw [execR_unfusedSiLU_split] at hRsplit
  cases h1 : exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none => simp [h1] at hRsplit
  | some s1 =>
    simp only [h1, Option.bind_some] at hRsplit
    cases h2 : exec (siluStepSilu zReg siluReg blockSize) s1 with
    | none => simp [h2] at hRsplit
    | some s2 =>
      simp only [h2, Option.bind_some] at hRsplit
      -- hRsplit : execR R (siluStepResidual …) s2 = some rhs'
      have hpid1 : s1.pid = s.pid := exec_pid h1
      have hpid2 : s2.pid = s1.pid := exec_pid h2
      have hpid2s : s2.pid = s.pid := hpid2.trans hpid1
      by_cases hrOut : r = outReg
      · -- both sides scatter equal bf16-rounded values to the same `outReg` offsets
        subst hrOut
        have h_z : InputLoadedAt s1 zReg blockSize (fun j => xs j * gates j) := by
          intro j
          have hzj :=
            silu_step_gate_correct xReg gateReg zReg blockSize s xs gates hx hg j
          rw [h1] at hzj
          simp [observeAt] at hzj
          rw [hpid1]
          exact hzj
        have h_silu_loaded :
            InputLoadedAt s2 siluReg blockSize ((fun j => Triton.TiledActivation.silu (xs j * gates j))) := by
          intro j
          have hsj :=
            silu_step_silu_correct zReg siluReg blockSize s1 (fun j => xs j * gates j) h_z j
          rw [h2] at hsj
          simp [observeAt] at hsj
          rw [hpid2]
          exact hsj
        have h_res1 : InputLoadedAt s1 residualReg blockSize residuals := by
          intro j
          have hrj :=
            silu_step_gate_preserves xReg gateReg zReg residualReg blockSize s
              xs gates hx hg (fun h => h_zRes h.symm) j
          rw [h1] at hrj
          simp [observeAt] at hrj
          rw [hpid1, hrj]
          exact hres j
        have h_res2 : InputLoadedAt s2 residualReg blockSize residuals := by
          intro j
          have hrj :=
            silu_step_silu_preserves zReg siluReg residualReg blockSize s1
              (fun j => xs j * gates j) h_z (fun h => h_siluRes h.symm) j
          rw [h2] at hrj
          simp [observeAt] at hrj
          rw [hpid2, hrj]
          exact h_res1 j
        -- symbolic execution of the two rounded stores
        simp [execR, fusedSiLUKernel, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop, Tile.uop,
              NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at hL
        simp [execR, siluStepResidual, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop,
              NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at hRsplit
        subst hL
        subst hRsplit
        have hpids : s2.pids 0 = s.pids 0 := hpid2s
        rw [hpids]
        refine foldl_writeMemAsR_mem_congr R .bf16 _ _ _ _ ?_ r o _ _ ?_
        · -- per-lane written values agree (equal ℝ output, rounded identically)
          intro k _
          refine congrArg (fun v : ℝ => R.cast FloatDType.real FloatDType.bf16 (some v)) ?_
          have hxk := hx k.1
          have hgk := hg k.1
          have hresk := hres k.1
          have hsiluk := h_silu_loaded k.1
          have hres2k := h_res2 k.1
          rw [hpid2s] at hsiluk hres2k
          simp [hxk, hgk, hresk, hsiluk, hres2k,
                Triton.TiledActivation.silu]
        · -- the two scatter bases agree at the observed cell
          simp only [BlockState.setReg_mem]
          rw [silu_step_silu_mem_frame zReg siluReg blockSize s1 h2 hrsilu o,
              silu_step_gate_mem_frame xReg gateReg zReg blockSize s h1 hrz o]
      · -- `r` is none of `zReg`/`siluReg`/`outReg`: both sides preserve the cell
        rw [fusedR_silu_mem_frame xReg gateReg residualReg outReg blockSize s R hL hrOut o,
            siluR_step_residual_mem_frame siluReg residualReg outReg blockSize s2 R hRsplit hrOut o,
            silu_step_silu_mem_frame zReg siluReg blockSize s1 h2 hrsilu o,
            silu_step_gate_mem_frame xReg gateReg zReg blockSize s h1 hrz o]

/-! ## Trust audit (compile-time gate)

If either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See `VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean silu_kernels_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the loaded-input contract, the rounding-model surface,
-- and the state/region types — NO spec.
#stmtSurfaceSubset silu_kernels_refinement_view ⊆
  [fusedSiLUKernel, unfusedSiLUKernel, InputsLoaded, InputLoadedAt,
   ComputeRefine.Refines, RoundingModel, BlockState, RegionName]

end FusedSiLURounded.theorems

end VeriTile.Bench.Examples.FusedSiLURounded
