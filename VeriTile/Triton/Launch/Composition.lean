/-
VeriTile.Triton.Launch.Composition

Layer-2b disjoint whole-grid composition over explicit per-program frames.
-/

import VeriTile.Triton.Launch.Grid
import VeriTile.Triton.Memory.Footprint

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

/-- Derive grid-write disjointness for the common structured footprint shape:
each program writes the image of one tile of offsets into a fixed region, and
offset images for distinct program ids are pairwise disjoint. -/
theorem GridWritesDisjoint.of_tileImage
    {k : Kernel} {g : Grid} {s : BlockState}
    {frames : Kernel.GridFrames k g s}
    {shape : TileShape} {region : RegionName}
    (offset : GridIndex g → TileIndex shape → Nat)
    (hWrites :
      ∀ idx, (frames idx).writes = WriteFootprint.tileImage region (offset idx))
    (hOffsets :
      ∀ idx₁ idx₂, idx₁ ≠ idx₂ →
        ∀ i j, offset idx₁ i ≠ offset idx₂ j) :
    Kernel.GridWritesDisjoint frames := by
  intro idx₁ idx₂ hne
  rw [hWrites idx₁, hWrites idx₂]
  exact WriteFootprint.disjoint_tileImage_of_image_disjoint
    (hOffsets idx₁ idx₂ hne)

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

theorem mergeFrames_mem_eq_of_not_written {k : Kernel} {g : Grid} {s : BlockState}
    {frames : Kernel.GridFrames k g s}
    {region : RegionName} {offset : Nat}
    (hNotWritten : ¬ Kernel.GridWriteFootprint frames (region, offset)) :
    (Kernel.mergeFrames g s frames).mem region offset = s.mem region offset :=
  Kernel.mergeFrames_mem_unwritten region offset hNotWritten

theorem mergeFrames_mem_eq_of_region_not_written {k : Kernel} {g : Grid} {s : BlockState}
    {frames : Kernel.GridFrames k g s}
    {region : RegionName}
    (hNotWritten : ∀ offset, ¬ Kernel.GridWriteFootprint frames (region, offset)) :
    (Kernel.mergeFrames g s frames).mem region = s.mem region := by
  funext offset
  exact Kernel.mergeFrames_mem_eq_of_not_written (frames := frames)
    (region := region) (offset := offset) (hNotWritten offset)

theorem GridWriteFootprint.not_region_of_forall_frames
    {k : Kernel} {g : Grid} {s : BlockState}
    {frames : Kernel.GridFrames k g s} {region : RegionName}
    (h : ∀ idx offset, ¬ (frames idx).writes (region, offset)) :
    ∀ offset, ¬ Kernel.GridWriteFootprint frames (region, offset) := by
  intro offset hWritten
  rcases hWritten with ⟨idx, hidx⟩
  exact h idx offset hidx

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
  have h := frame.mem_eq_of_not_written hNotWritten
  simpa [BlockState.withGridIndex, BlockState.withPids] using h

/-- Relational whole-grid launch surface for kernels whose per-program writes
are ordinary disjoint frame writes.

This is the launcher-facing wrapper around `mergeFrames`: callers provide one
successful framed execution for every program index plus pairwise disjointness,
and the relation exposes the merged final state without committing to an
arbitrary program execution order. -/
structure GridLaunchedOrdinary (k : Kernel) (g : Grid)
    (s sFinal : BlockState) where
  frames : Kernel.GridFrames k g s
  h_disjoint : Kernel.GridWritesDisjoint frames
  h_final : sFinal = Kernel.mergeFrames g s frames

namespace GridLaunchedOrdinary

theorem observeOrdinaryCell {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedOrdinary k g s sFinal)
    (idx : GridIndex g) (region : RegionName) (offset : Nat)
    (hWrite : (h.frames idx).writes (region, offset)) :
    sFinal.mem region offset = (h.frames idx).final.mem region offset := by
  rcases h with ⟨frames, hDisjoint, hFinal⟩
  subst sFinal
  exact Kernel.mergeFrames_mem_written hDisjoint idx region offset hWrite

theorem observeUnwrittenCell {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedOrdinary k g s sFinal)
    (region : RegionName) (offset : Nat)
    (hNotWritten : ¬ Kernel.GridWriteFootprint h.frames (region, offset)) :
    sFinal.mem region offset = s.mem region offset := by
  rcases h with ⟨frames, _hDisjoint, hFinal⟩
  subst sFinal
  exact Kernel.mergeFrames_mem_eq_of_not_written (frames := frames)
    (region := region) (offset := offset) hNotWritten

theorem writeWithin {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedOrdinary k g s sFinal) :
    BlockState.WriteWithin (Kernel.GridWriteFootprint h.frames) s sFinal := by
  rcases h with ⟨frames, _hDisjoint, hFinal⟩
  subst sFinal
  exact Kernel.mergeFrames_writeWithin frames

end GridLaunchedOrdinary

/-- Whole-grid single-memory output correctness.

The launcher-facing analogue of `ComputeCorrect.Realizes`, lifted from a single
program to the merged whole-grid memory. `LaunchCorrect k g s write expected`
holds when there *exists* an ordinary disjoint launch of `k` over grid `g` from
initial state `s` whose single merged final memory realizes `expected` at every
active output cell selected by `write` (a `WriteMap`-style partial address map
indexed by `ι`).

Bundling the launch witness makes this an unconditional contract: a kernel
satisfies `LaunchCorrect` outright (by constructing one valid disjoint launch),
with no launch/footprint side hypotheses for callers to discharge. -/
def LaunchCorrect {ι : Type} (k : Kernel) (g : Grid) (s : BlockState)
    (write : ι → Option MemCellAddr) (expected : ι → ℝ) : Prop :=
  ∃ sFinal, ∃ _ : Kernel.GridLaunchedOrdinary k g s sFinal,
    ∀ (i : ι) (addr : MemCellAddr), write i = some addr →
      sFinal.readMem addr.1 addr.2 = expected i

end Kernel

end VeriTile.Triton
