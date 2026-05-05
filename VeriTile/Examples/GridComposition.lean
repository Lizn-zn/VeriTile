/-
VeriTile.Examples.GridComposition

Representative whole-grid disjoint-frame composition smoke.
-/

import VeriTile.Triton.Memory.Footprint
import VeriTile.Triton.Launch.Composition
import VeriTile.Triton.Concurrency.Atomic

namespace VeriTile.Examples.GridComposition

open VeriTile.Triton

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
  intro idx₁ idx₂ hne
  apply WriteFootprint.disjoint_tileImage_of_image_disjoint
  intro _ _ hEq
  exact hne (grid1_ext hEq)

theorem gridConstStore_merge_observe_written
    (outReg : RegionName) (n : Nat) (s : BlockState) (i : Fin n) :
    (Kernel.mergeFrames (Grid1 n) s (gridConstStoreFrames outReg n s)).mem outReg i.val =
      ((gridConstStoreFrames outReg n s) (grid1Index i)).final.mem outReg i.val := by
  apply Kernel.mergeFrames_mem_written
  · exact gridConstStoreFrames_disjoint outReg n s
  · change WriteFootprint.tileImage outReg
      (fun _ : TileIndex [] =>
        (grid1Index i ⟨0, by simp [Grid.rank]⟩).val) (outReg, i.val)
    simp [WriteFootprint.tileImage]
    exact (grid1Index_zero i).symm

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

private def grid1ThreadId {n : Nat} (idx : GridIndex (Grid1 n)) : ThreadId :=
  { program := [(idx ⟨0, by simp [Grid.rank]⟩).val]
    lane := 0 }

/-- Toy trace for the split-K smoke: one Real atomic-add event per program. -/
def splitKAtomicTrace {n : Nat} (outReg : RegionName)
    (contrib : GridIndex (Grid1 n) → ℝ) (idx : GridIndex (Grid1 n)) : Trace :=
  [Stmt.atomicTraceEvent (grid1ThreadId idx) outReg 0 .real (contrib idx)]

@[simp] theorem splitKAtomicTrace_sum {n : Nat} (outReg : RegionName)
    (contrib : GridIndex (Grid1 n) → ℝ) (idx : GridIndex (Grid1 n)) :
    (splitKAtomicTrace outReg contrib idx).atomicAddRealSum (outReg, 0) =
      contrib idx := by
  simp [splitKAtomicTrace]

/-- Minimal atomic-grid composition theorem.

This is the second consumer of the generic `mergeFramesWithAtomic` theorem
after FA-1 backward: when every program contributes one atomic-add payload to a
single output cell, the merged cell equals the initial value plus the Finset
sum of all per-program contributions. -/
theorem splitKAtomicAdd_merge_sum
    (outReg : RegionName) (n : Nat) (s : BlockState)
    (frames : Kernel.GridFrames (splitKAtomicAddKernel outReg) (Grid1 n) s)
    (contrib : GridIndex (Grid1 n) → ℝ)
    (hNoOrdinaryWrite : ¬ Kernel.GridWriteFootprint frames (outReg, 0)) :
    (Kernel.mergeFramesWithAtomic (Grid1 n) s frames
        (Finset.univ : Finset (GridIndex (Grid1 n)))
        (splitKAtomicTrace outReg contrib)).readMem outReg 0 =
      s.readMem outReg 0 +
        (Finset.univ : Finset (GridIndex (Grid1 n))).sum contrib := by
  classical
  rw [Kernel.mergeFramesWithAtomic_atomicAdd_eq_finsetSum
    (frames := frames)
    (contributors := (Finset.univ : Finset (GridIndex (Grid1 n))))
    (atomicTrace := splitKAtomicTrace outReg contrib)
    (region := outReg) (offset := 0)
    (hNoOrdinaryWrite := hNoOrdinaryWrite)]
  congr 1
  apply Finset.sum_congr rfl
  intro idx _
  simp

end VeriTile.Examples.GridComposition
