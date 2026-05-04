/-
VeriTile.Examples.GridComposition

Representative whole-grid disjoint-frame composition smoke.
-/

import VeriTile.Triton.Memory.Footprint
import VeriTile.Triton.Launch.Composition

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

end VeriTile.Examples.GridComposition
