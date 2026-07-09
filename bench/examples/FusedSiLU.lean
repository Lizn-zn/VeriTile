import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Fused SiLU ≡ three-kernel pipeline — writes-equality refinement

Self-contained showcase, read top to bottom: **kernels** first (the fused
kernel and the three step kernels), the **supporting lemmas** in the middle
(`private` plumbing — per-kernel realizations, aliasing frames, the pipeline
split), the **theorem** last (one public headline
`silu_kernels_refinement_view`), then a compile-time **trust audit**. The three
real sections below are `FusedSiLU.kernels`, `FusedSiLU.lemmas`,
`FusedSiLU.theorems`.

The compositional-refinement template that `bench/examples/FusedSwiglu.lean`
mirrors: a fused single kernel

  y = residual + (x * gate) * sigmoid(x * gate)

is proven equal to a three-kernel pipeline that materializes the intermediate
`z = x * gate`, then `silu = z * sigmoid(z)`, then adds the residual. The proof
is memory-aware: the unfused pipeline writes temporary regions `zReg`/`siluReg`,
and the refinement requires these not to alias `residualReg` so the residual
input survives to the final stage.

## The public result (bottom of file)

The single public headline is **`silu_kernels_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines`: from the same state the
fused kernel and the unfused pipeline perform the same writes, agreeing at every
cell outside the declared scratch temporaries `[zReg, siluReg]`. Its statement
mentions only the two kernels, the loaded-input contract, the writes-equality
surface, and the state/region types — **no spec** (the `#stmtSurfaceSubset`
gate below enforces this; the per-lane specs and correctness lemmas are all
`private`). For the rounding-model (∀R) analogue of this compositional pattern
see `bench/examples/FusedSwiglu.lean`.
-/

namespace VeriTile.Bench.Examples.FusedSiLU

open VeriTile.Triton VeriTile.Examples

/-! ## Kernels -/
section FusedSiLU.kernels

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
  tl.store($(outReg) + offsets, y)
}

def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load($(xReg) + offsets)
  gate    := tl.load($(gateReg) + offsets)
  z       := x * gate
  tl.store($(zReg) + offsets, z)
}

def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load($(zReg) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store($(siluReg) + offsets, silu)
}

def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

/-- The unfused pipeline as one kernel: `ComputeKernel.seq` of the three step
kernels — the concatenation of their bodies (registers flow across the seams).
The staged execution is recovered by `exec_unfusedSiLU_split` below. -/
def unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGate xReg gateReg zReg blockSize,
     siluStepSilu zReg siluReg blockSize,
     siluStepResidual siluReg residualReg outReg blockSize]

end FusedSiLU.kernels

/- Shared parameters of every lemma and the headline — the pipeline's regions in
data-flow order (`keepReg` is the generic frame region), the block size, the
state, and the three per-lane input vectors. Declared once at namespace scope so
both `FusedSiLU.lemmas` and `FusedSiLU.theorems` inherit them (each declaration
picks up only the variables it mentions). Block nonemptiness is never needed, so
there is no `0 < blockSize` variable. -/
variable (xReg gateReg zReg siluReg residualReg outReg keepReg : RegionName)
variable (blockSize : Nat) (s : BlockState)
variable (xs gates residuals : Fin blockSize → ℝ)

/-! ## Supporting lemmas (private plumbing) -/
section FusedSiLU.lemmas

private theorem stepStmts_append (l₁ l₂ : List Stmt) (t : BlockState) :
    stepStmts (l₁ ++ l₂) t = (stepStmts l₁ t).bind (fun t' => stepStmts l₂ t') := by
  induction l₁ generalizing t with
  | nil => simp
  | cons st rest ih =>
      simp [stepStmts]
      cases stepStmt st t <;> simp [ih]

/-- The unfused pipeline's execution splits into its three stages run in
sequence (no register reset across the seams) — the `ComputeKernel.seq` fold
made concrete. -/
private theorem exec_unfusedSiLU_split :
    exec (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize) s =
      (exec (siluStepGate xReg gateReg zReg blockSize) s).bind (fun s1 =>
        (exec (siluStepSilu zReg siluReg blockSize) s1).bind (fun s2 =>
          exec (siluStepResidual siluReg residualReg outReg blockSize) s2)) := by
  simp [unfusedSiLUKernel, exec, stepStmts_append]

private theorem fused_silu_correct
    (_h_x   : InputLoadedAt s xReg blockSize xs)
    (_h_g   : InputLoadedAt s gateReg blockSize gates)
    (_h_res : InputLoadedAt s residualReg blockSize residuals) :
    ∀ i : Fin blockSize,
      observeAt (exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (Triton.TiledActivation.fusedSiLU (xs i) (gates i) (residuals i)) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        Triton.TiledActivation.fusedSiLU,
        Triton.TiledActivation.silu]
  unfold InputLoadedAt at _h_x _h_g _h_res
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_g, _h_res]

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
  simp [observeAt, exec, siluStepGate, stepStmts, stepStmt, evalOp,
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
  simp [observeAt, exec, siluStepGate, stepStmts, stepStmt, evalOp,
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

private theorem silu_step_residual_correct (silus resids : Fin blockSize → ℝ)
    (_h_silu : InputLoadedAt s siluReg blockSize silus)
    (_h_res : InputLoadedAt s residualReg blockSize resids) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepResidual siluReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (resids i + silus i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStepResidual, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_silu _h_res
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_silu, _h_res]

private theorem silu_step_residual_preserves (silus resids : Fin blockSize → ℝ)
    (_h_silu : InputLoadedAt s siluReg blockSize silus)
    (_h_res : InputLoadedAt s residualReg blockSize resids)
    (h_keep : keepReg ≠ outReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepResidual siluReg residualReg outReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStepResidual, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_silu _h_res
  simp_rw [_h_silu, _h_res]
  rw [BlockState.scatter_preserves_other_region outReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => resids k.1 + silus k.1)
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

set_option maxHeartbeats 800000 in
private theorem unfused_silu_correct
    (_h_x   : InputLoadedAt s xReg blockSize xs)
    (_h_g   : InputLoadedAt s gateReg blockSize gates)
    (_h_res : InputLoadedAt s residualReg blockSize residuals)
    (_h_zRes    : zReg ≠ residualReg)
    (_h_siluRes : siluReg ≠ residualReg) :
    ∀ i : Fin blockSize,
      observeAt
        (exec (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize) s)
        outReg blockSize s.pid i
        = some (Triton.TiledActivation.fusedSiLU (xs i) (gates i) (residuals i)) := by
  intro i
  rw [exec_unfusedSiLU_split]
  cases h_gate : exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none =>
      have hz :=
        silu_step_gate_correct xReg gateReg zReg blockSize s xs gates
          _h_x _h_g i
      rw [h_gate] at hz
      simp [observeAt] at hz
  | some s1 =>
      simp only [Option.bind_some]
      have hpid1 : s1.pid = s.pid := exec_pid h_gate
      have h_z : InputLoadedAt s1 zReg blockSize (fun j => xs j * gates j) := by
        intro j
        have hzj :=
          silu_step_gate_correct xReg gateReg zReg blockSize s xs gates
            _h_x _h_g j
        rw [h_gate] at hzj
        simp [observeAt] at hzj
        rw [hpid1]
        exact hzj
      have h_res1 : InputLoadedAt s1 residualReg blockSize residuals := by
        intro j
        have hrj :=
          silu_step_gate_preserves xReg gateReg zReg residualReg blockSize s
            xs gates _h_x _h_g (fun h => _h_zRes h.symm) j
        rw [h_gate] at hrj
        simp [observeAt] at hrj
        rw [hpid1]
        rw [hrj]
        exact _h_res j
      cases h_silu : exec (siluStepSilu zReg siluReg blockSize) s1 with
      | none =>
          have hsj :=
            silu_step_silu_correct zReg siluReg blockSize s1
              (fun j => xs j * gates j) h_z i
          rw [h_silu] at hsj
          simp [observeAt] at hsj
      | some s2 =>
          simp only [Option.bind_some]
          have hpid2 : s2.pid = s1.pid := exec_pid h_silu
          have hpid2s : s2.pid = s.pid := hpid2.trans hpid1
          have h_silu_loaded :
              InputLoadedAt s2 siluReg blockSize ((fun j => Triton.TiledActivation.silu (xs j * gates j))) := by
            intro j
            have hsj :=
              silu_step_silu_correct zReg siluReg blockSize s1
                (fun j => xs j * gates j) h_z j
            rw [h_silu] at hsj
            simp [observeAt] at hsj
            rw [hpid2]
            exact hsj
          have h_res2 : InputLoadedAt s2 residualReg blockSize residuals := by
            intro j
            have hrj :=
              silu_step_silu_preserves zReg siluReg residualReg blockSize s1
                (fun j => xs j * gates j) h_z (fun h => _h_siluRes h.symm) j
            rw [h_silu] at hrj
            simp [observeAt] at hrj
            rw [hpid2]
            rw [hrj]
            exact h_res1 j
          have hout :=
            silu_step_residual_correct siluReg residualReg outReg blockSize s2
              ((fun j => Triton.TiledActivation.silu (xs j * gates j))) residuals h_silu_loaded h_res2 i
          rw [hpid2s] at hout
          rw [hout]
          simp [Triton.TiledActivation.fusedSiLU, Triton.TiledActivation.silu]

private theorem silu_kernels_refinement
    (h_x   : InputLoadedAt s xReg blockSize xs)
    (h_g   : InputLoadedAt s gateReg blockSize gates)
    (h_res : InputLoadedAt s residualReg blockSize residuals)
    (h_zRes    : zReg ≠ residualReg)
    (h_siluRes : siluReg ≠ residualReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i =
      observeAt
        (exec (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize) s)
        outReg blockSize s.pid i := by
  intro i
  rw [fused_silu_correct
        xReg gateReg residualReg outReg blockSize s xs gates residuals h_x h_g h_res i,
      unfused_silu_correct
        xReg gateReg zReg siluReg residualReg outReg blockSize s xs gates residuals
        h_x h_g h_res h_zRes h_siluRes i]

/-! ### Whole-memory frame lemmas for the writes-equality refinement surface

`ComputeRefine.Refines` compares the two final memories cell-by-cell, so the
refinement proof needs `.mem`-level frames: each kernel touches only its own
store target. Proved by symbolic execution down to the store's `writeMem`
scatter, then the generic scatter frame lemma. -/

private theorem fused_silu_mem_frame {s' : BlockState}
    (h : exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ outReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

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

private theorem silu_step_residual_mem_frame {s' : BlockState}
    (h : exec (siluStepResidual siluReg residualReg outReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ outReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, siluStepResidual, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

end FusedSiLU.lemmas

/-! ## The headline theorem -/
section FusedSiLU.theorems

set_option maxHeartbeats 1600000 in
/-- Compute-facing surface for `silu_kernels_refinement`: writes-equality
refinement. From the same initial state, the fused kernel and the unfused
pipeline perform the same writes — the two final memories agree at every cell
outside the declared scratch regions `zReg`/`siluReg` (the pipeline's
temporaries). Inputs enter via the compact `InputLoadedAt` contract; the two
`≠ residualReg` hypotheses keep the pipeline's scratch temporaries from
clobbering the residual input before the final stage reads it. -/
theorem silu_kernels_refinement_view
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_g : InputLoadedAt s gateReg blockSize gates)
    (h_res : InputLoadedAt s residualReg blockSize residuals)
    (h_zRes : zReg ≠ residualReg)
    (h_siluRes : siluReg ≠ residualReg) :
    ComputeRefine.Refines
      (fusedSiLUKernel xReg gateReg residualReg outReg blockSize)
      (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize)
      s [zReg, siluReg] := by
  have hx := h_x
  have hg := h_g
  have hres := h_res
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hr
  obtain ⟨hrz, hrsilu⟩ := hr
  have hRsplit := hR
  rw [exec_unfusedSiLU_split] at hRsplit
  cases h1 : exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none => simp [h1] at hRsplit
  | some s1 =>
    simp only [h1, Option.bind_some] at hRsplit
    cases h2 : exec (siluStepSilu zReg siluReg blockSize) s1 with
    | none => simp [h2] at hRsplit
    | some s2 =>
      simp only [h2, Option.bind_some] at hRsplit
      -- hRsplit : exec (siluStepResidual …) s2 = some rhs'
      have hpid1 : s1.pid = s.pid := exec_pid h1
      have hpid2 : s2.pid = s1.pid := exec_pid h2
      have hpid2s : s2.pid = s.pid := hpid2.trans hpid1
      by_cases hrOut : r = outReg
      · -- both sides scatter the same values to the same `outReg` offsets
        subst hrOut
        -- pipeline stage facts at `s2` (mirrors `unfused_silu_correct`)
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
        -- symbolic execution of the two stores
        simp [exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
              NumericDType.add, NumericDType.mul] at hL
        simp [exec, siluStepResidual, stepStmts, stepStmt, Tile.bop,
              NumericDType.add, NumericDType.mul] at hRsplit
        subst hL
        subst hRsplit
        have hpids : s2.pids 0 = s.pids 0 := hpid2s
        rw [hpids]
        refine BlockState.foldl_writeMem_mem_congr _ _ _ _ ?_ r o _ _ ?_
        · -- per-lane written values agree
          intro k _
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
        rw [fused_silu_mem_frame xReg gateReg residualReg outReg blockSize s hL hrOut o,
            silu_step_residual_mem_frame siluReg residualReg outReg blockSize s2 hRsplit hrOut o,
            silu_step_silu_mem_frame zReg siluReg blockSize s1 h2 hrsilu o,
            silu_step_gate_mem_frame xReg gateReg zReg blockSize s h1 hrz o]

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean silu_kernels_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the loaded-input contract, the writes-equality surface,
-- and the state/region types — NO spec.
#stmtSurfaceSubset silu_kernels_refinement_view ⊆
  [fusedSiLUKernel, unfusedSiLUKernel, InputLoadedAt,
   ComputeRefine.Refines, BlockState, RegionName]

end FusedSiLU.theorems

end VeriTile.Bench.Examples.FusedSiLU
