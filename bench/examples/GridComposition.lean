/-
bench/examples/GridComposition

Representative whole-grid disjoint-frame composition smoke.
-/

import VeriTile.Triton.Memory.Footprint
import VeriTile.Triton.Launch.Composition
import VeriTile.Triton.Concurrency.Atomic
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Examples.Common

namespace VeriTile.Bench.Examples.GridComposition

open VeriTile.Triton
open VeriTile.Examples

/-- One-dimensional constant store indexed by `tl.program_id(0)`. -/
def gridConstStoreKernel (outReg : RegionName) : Kernel :=
  { inputs := []
    outputs := [outReg]
    body :=
      [ Stmt.store .real [] (MemAccess.region outReg (Op.programId 0))
          (Op.const 1) MaskOpt.none
      ] }

private abbrev Grid1 (n : Nat) : Grid := { dims := [n] }

private def grid1Index {n : Nat} (i : Fin n) : GridIndex (Grid1 n) :=
  fun axis => by
    rcases axis with ⟨axis, haxis⟩
    cases axis with
    | zero =>
        simpa [Grid1, Grid.dim] using i
    | succ axis =>
        simp [Grid.rank] at haxis

private theorem grid1Index_zero {n : Nat} (i : Fin n) :
    (grid1Index i ⟨0, by simp [Grid.rank]⟩).val = i.val := by
  change (Fin.cast (by simp [Grid.dim]) i).val = i.val
  simp

private theorem grid1_ext {n : Nat} {idx₁ idx₂ : GridIndex (Grid1 n)}
    (h0 : (idx₁ ⟨0, by simp [Grid.rank]⟩).val =
      (idx₂ ⟨0, by simp [Grid.rank]⟩).val) :
    idx₁ = idx₂ := by
  funext axis
  rcases axis with ⟨axis, haxis⟩
  cases axis with
  | zero =>
      exact Fin.ext (by simpa using h0)
  | succ axis =>
      simp [Grid.rank] at haxis

def gridConstStoreFrame (outReg : RegionName) (n : Nat) (s : BlockState)
    (idx : GridIndex (Grid1 n)) :
    Kernel.ExecFrame (gridConstStoreKernel outReg) (s.withGridIndex idx) :=
  let axis0 : Fin (Grid1 n).rank := ⟨0, by simp [Grid.rank]⟩
  let off : Nat := (idx axis0).val
  { final := (s.withGridIndex idx).writeMemTyped .real outReg off (some 1)
    writes := WriteFootprint.tileImage outReg (fun _ : TileIndex [] => off)
    h_exec := by
      simp [gridConstStoreKernel, exec, stepStmts, stepStmt, evalOp,
        Grid1, off, axis0]
    h_writeWithin := by
      exact BlockState.writeMemTyped_writeWithin .real outReg off (some 1)
        (by simp [WriteFootprint.tileImage]) }

def gridConstStoreFrames (outReg : RegionName) (n : Nat) (s : BlockState) :
    Kernel.GridFrames (gridConstStoreKernel outReg) (Grid1 n) s :=
  fun idx => gridConstStoreFrame outReg n s idx

theorem gridConstStoreFrames_disjoint
    (outReg : RegionName) (n : Nat) (s : BlockState) :
    Kernel.GridWritesDisjoint (gridConstStoreFrames outReg n s) := by
  apply Kernel.GridWritesDisjoint.of_tileImage
    (frames := gridConstStoreFrames outReg n s)
    (shape := []) (region := outReg)
    (offset := fun idx _ => (idx ⟨0, by simp [Grid.rank]⟩).val)
  · intro idx
    rfl
  · intro idx₁ idx₂ hne _ _ hEq
    exact hne (grid1_ext hEq)

/-- Canonical ordinary launch witness for the one-dimensional disjoint store smoke. -/
def gridConstStoreLaunched (outReg : RegionName) (n : Nat) (s : BlockState) :
    Kernel.GridLaunchedOrdinary (gridConstStoreKernel outReg) (Grid1 n) s
      (Kernel.mergeFrames (Grid1 n) s (gridConstStoreFrames outReg n s)) :=
  { frames := gridConstStoreFrames outReg n s
    h_disjoint := gridConstStoreFrames_disjoint outReg n s
    h_final := rfl }

specification gridConstStore_launched_observe_written
    (outReg : RegionName) (n : Nat) (s : BlockState) (i : Fin n) :
    (Kernel.mergeFrames (Grid1 n) s (gridConstStoreFrames outReg n s)).mem outReg i.val =
      ((gridConstStoreFrames outReg n s) (grid1Index i)).final.mem outReg i.val := by
  apply (gridConstStoreLaunched outReg n s).observeOrdinaryCell
  change WriteFootprint.tileImage outReg
      (fun _ : TileIndex [] =>
        (grid1Index i ⟨0, by simp [Grid.rank]⟩).val) (outReg, i.val)
  simp [WriteFootprint.tileImage]
  exact (grid1Index_zero i).symm

theorem gridConstStore_launched_unrelated
    (outReg otherReg : RegionName) (n : Nat) (s : BlockState) (offset : Nat)
    (hOther : otherReg ≠ outReg) :
    (Kernel.mergeFrames (Grid1 n) s (gridConstStoreFrames outReg n s)).mem otherReg offset =
      s.mem otherReg offset := by
  apply (gridConstStoreLaunched outReg n s).observeUnwrittenCell
  apply Kernel.GridWriteFootprint.not_region_of_forall_frames
  intro idx offset
  exact WriteFootprint.not_tileImage_of_region_ne hOther

theorem gridConstStore_merge_unrelated
    (outReg otherReg : RegionName) (n : Nat) (s : BlockState) (offset : Nat)
    (hOther : otherReg ≠ outReg) :
    (Kernel.mergeFrames (Grid1 n) s (gridConstStoreFrames outReg n s)).mem otherReg offset =
      s.mem otherReg offset := by
  exact Kernel.mergeFrames_mem_eq_of_not_written (frames := gridConstStoreFrames outReg n s)
    (region := otherReg) (offset := offset) (by
      apply Kernel.GridWriteFootprint.not_region_of_forall_frames
      intro idx offset
      exact WriteFootprint.not_tileImage_of_region_ne hOther)

/-- Minimal split-K-style atomic accumulation smoke.

Every program contributes one scalar value to the same output cell.  The
ordinary frame merge deliberately excludes that cell; the atomic trace supplies
the per-program contributions consumed by `mergeFramesWithAtomic`. -/
def splitKAtomicAddKernel (outReg : RegionName) : Kernel :=
  { inputs := []
    outputs := [outReg]
    body :=
      [ Stmt.assign .nat [] "pid" (Op.programId 0)
      , Stmt.atomicAdd NumericDType.real [] (MemAccess.region outReg (Op.constNat 0))
          (Op.natToReal (Op.ref .nat [] "pid")) MaskOpt.none
      ] }

/-- Toy trace for the split-K smoke: one Real atomic-add event per program. -/
def splitKAtomicTrace {n : Nat} (outReg : RegionName)
    (contrib : GridIndex (Grid1 n) → ℝ) (idx : GridIndex (Grid1 n)) : Trace :=
  [Stmt.atomicTraceEvent (Kernel.threadIdOf idx) outReg 0 .real (contrib idx)]

@[simp] theorem splitKAtomicTrace_sum {n : Nat} (outReg : RegionName)
    (contrib : GridIndex (Grid1 n) → ℝ) (idx : GridIndex (Grid1 n)) :
    (splitKAtomicTrace outReg contrib idx).atomicAddRealSum (outReg, 0) =
      contrib idx := by
  simp [splitKAtomicTrace]

/-- Example consumer for `ProgramRun.of_uniform`: package a uniform trace
theorem over all split-K program ids into the launcher `runs` field. -/
def splitKAtomicRuns_of_uniform
    (outReg : RegionName) (n : Nat) (s : BlockState)
    (contrib : GridIndex (Grid1 n) → ℝ)
    (final : GridIndex (Grid1 n) → BlockState)
    (hTrace :
      ∀ idx,
        Kernel.AtomicTraceStateful (splitKAtomicAddKernel outReg)
          (Kernel.threadIdOf idx) (s.withGridIndex idx)
          (splitKAtomicTrace outReg contrib idx) (final idx)) :
    (idx : GridIndex (Grid1 n)) →
      Kernel.ProgramRun (splitKAtomicAddKernel outReg) (Grid1 n) s idx :=
  Kernel.ProgramRun.of_uniform
    (k := splitKAtomicAddKernel outReg) (g := Grid1 n) (s := s)
    (trace := splitKAtomicTrace outReg contrib) (final := final) hTrace

/-- Minimal atomic-grid composition theorem.

This is the second consumer of the generic `mergeFramesWithAtomic` theorem
after FA-1 backward: when every program contributes one atomic-add payload to a
single output cell, the merged cell equals the initial value plus the Finset
sum of all per-program contributions. -/
specification splitKAtomicAdd_launched_sum
    (outReg : RegionName) (n : Nat) (s sFinal : BlockState)
    (hLaunch :
      Kernel.GridLaunchedAtomic (splitKAtomicAddKernel outReg) (Grid1 n) s sFinal)
    (contrib : GridIndex (Grid1 n) → ℝ)
    (hNoOrdinaryWrite : ¬ Kernel.GridWriteFootprint hLaunch.frames (outReg, 0))
    (hContrib :
      ∀ idx ∈ hLaunch.contributors,
        (hLaunch.runs idx).trace.atomicAddRealSum (outReg, 0) = contrib idx) :
    sFinal.readMem outReg 0 =
      s.readMem outReg 0 +
        hLaunch.contributors.sum contrib := by
  classical
  rw [hLaunch.observeAtomicCell outReg 0 hNoOrdinaryWrite]
  congr 1
  exact Finset.sum_congr rfl (by
    intro idx hidx
    exact hContrib idx hidx)

/-- Order-sensitive atomic launcher smoke.

If a `GridLaunchedRMW` witness linearizes two `atomic_xchg` events at one cell,
the final cell is the second exchanged value.  This is intentionally stated over
the generic launcher witness: kernel-specific proofs only need to construct the
per-program runs, ordinary frames, and linearization list. -/
specification gridLaunchedRMW_xchg_two_final
    {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedRMW k g s sFinal)
    (cell : MemCellAddr) (old first second : MemCell)
    (hCell : h.cell = cell)
    (hInitial : s.mem cell.1 cell.2 = old)
    (hLinearization :
      h.linearization =
        [ { cell := cell, op := RMWOp.xchg, input := first }
        , { cell := cell, op := RMWOp.xchg, input := second } ]) :
    sFinal.mem cell.1 cell.2 = second := by
  have hApply := h.applyLinearized
  rw [hCell, hInitial, hLinearization] at hApply
  rw [RMWTrace.applyLinearized_xchg_two old first second cell] at hApply
  injection hApply with hFinalCell
  have hFinal : h.finalCell = second := by
    simpa using congrArg Prod.fst hFinalCell.symm
  have hObs := h.observeRMWCell
  rw [hCell] at hObs
  exact hObs.trans hFinal

/-! ## Trust gates -/

#axiomsClean gridConstStore_launched_observe_written
#axiomsClean splitKAtomicAdd_launched_sum
#axiomsClean gridLaunchedRMW_xchg_two_final

end VeriTile.Bench.Examples.GridComposition
