/-
VeriTile.Triton.Launch.Composition

Layer-2b disjoint whole-grid composition over explicit per-program frames.
-/

import VeriTile.Triton.Launch.Grid
import VeriTile.Triton.MemoryFrame

namespace VeriTile.Triton

namespace Kernel

/-- One successful framed execution for every program index in a grid. -/
abbrev GridFrames (k : Kernel) (g : Grid) (s : BlockState) : Type :=
  (idx : GridIndex g) → Kernel.ExecFrame k (s.withGridIndex idx)

/-- Union of all per-program write footprints. -/
def GridWriteFootprint {k : Kernel} {g : Grid} {s : BlockState}
    (frames : Kernel.GridFrames k g s) : WriteFootprint :=
  fun addr => ∃ idx : GridIndex g, (frames idx).writes addr

/-- Pairwise disjoint per-program write footprints. -/
def GridWritesDisjoint {k : Kernel} {g : Grid} {s : BlockState}
    (frames : Kernel.GridFrames k g s) : Prop :=
  ∀ idx₁ idx₂ : GridIndex g, idx₁ ≠ idx₂ →
    WriteFootprint.disjoint (frames idx₁).writes (frames idx₂).writes

theorem GridWritesDisjoint.eq_of_both {k : Kernel} {g : Grid} {s : BlockState}
    {frames : Kernel.GridFrames k g s}
    (hDisjoint : Kernel.GridWritesDisjoint frames)
    {idx₁ idx₂ : GridIndex g} {addr : MemCellAddr}
    (h₁ : (frames idx₁).writes addr) (h₂ : (frames idx₂).writes addr) :
    idx₁ = idx₂ := by
  by_contra hne
  exact hDisjoint idx₁ idx₂ hne addr h₁ h₂

/-- Extensional merge of independent per-program frame results.

This is noncomputable because `WriteFootprint` is an arbitrary `Prop`
predicate. Classical choice only selects a writer from the existential proof
that some program wrote the queried cell. -/
noncomputable def mergeFrames {k : Kernel} (g : Grid) (s : BlockState)
    (frames : Kernel.GridFrames k g s) : BlockState := by
  classical
  exact { s with mem := fun region offset =>
    (by
      by_cases h : ∃ idx : GridIndex g, (frames idx).writes (region, offset)
      · exact (frames (Classical.choose h)).final.mem region offset
      · exact s.mem region offset) }

theorem mergeFrames_mem_written {k : Kernel} {g : Grid} {s : BlockState}
    {frames : Kernel.GridFrames k g s}
    (hDisjoint : Kernel.GridWritesDisjoint frames)
    (idx : GridIndex g) (region : RegionName) (offset : Nat)
    (hWrite : (frames idx).writes (region, offset)) :
    (Kernel.mergeFrames g s frames).mem region offset =
      (frames idx).final.mem region offset := by
  classical
  unfold Kernel.mergeFrames
  dsimp
  have hExists : ∃ idx : GridIndex g, (frames idx).writes (region, offset) :=
    ⟨idx, hWrite⟩
  rw [dif_pos hExists]
  let chosen : GridIndex g := Classical.choose hExists
  have hChosen : (frames chosen).writes (region, offset) :=
    Classical.choose_spec hExists
  have hEq : chosen = idx :=
    hDisjoint.eq_of_both hChosen hWrite
  subst hEq
  rfl

theorem mergeFrames_mem_unwritten {k : Kernel} {g : Grid} {s : BlockState}
    {frames : Kernel.GridFrames k g s}
    (region : RegionName) (offset : Nat)
    (hNotWritten : ¬ Kernel.GridWriteFootprint frames (region, offset)) :
    (Kernel.mergeFrames g s frames).mem region offset = s.mem region offset := by
  classical
  unfold Kernel.mergeFrames Kernel.GridWriteFootprint at *
  dsimp
  rw [dif_neg hNotWritten]

theorem mergeFrames_writeWithin {k : Kernel} {g : Grid} {s : BlockState}
    (frames : Kernel.GridFrames k g s) :
    BlockState.WriteWithin (Kernel.GridWriteFootprint frames) s
      (Kernel.mergeFrames g s frames) := by
  intro region offset hNotWritten
  exact (Kernel.mergeFrames_mem_unwritten (frames := frames) region offset hNotWritten).symm

theorem execFrame_mem_eq_initial_of_not_writes {k : Kernel} {g : Grid}
    {s : BlockState} (idx : GridIndex g)
    (frame : Kernel.ExecFrame k (s.withGridIndex idx))
    (region : RegionName) (offset : Nat)
    (hNotWritten : ¬ frame.writes (region, offset)) :
    frame.final.mem region offset = s.mem region offset := by
  have h := frame.h_writeWithin region offset hNotWritten
  simpa [BlockState.withGridIndex, BlockState.withPids] using h.symm

end Kernel

end VeriTile.Triton
