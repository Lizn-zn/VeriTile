import bench.tritonbench_g.attention_score.Proofs.RealContract

/-! Allocation regressions for the faithful source: checked Q/K tails and
masked output need only active cells, while the unmasked M read needs all
four visited rows for RN=3 and B=2. K is loaded even when the loop is empty. -/
namespace RealAttentionScoreIOTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionScore
noncomputable section

private def parameters : RealSource.Parameters where
  Q := "Q"
  K := "K"
  M := "M"
  Out := "Out"
  queryBatch := 0
  queryHead := 0
  queryRow := 1
  queryDim := 1
  keyBatch := 0
  keyHead := 0
  keyRow := 1
  keyDim := 1
  outputBatch := 0
  outputHead := 0
  outputInner := 99
  batches := 1
  heads := 1
  kvHeads := 1
  queryContext := 1
  roundedContext := 3
  keyContext := 1
  offset := 0
  size := 0
  scale := 0
  window := Bool.false
  complement := Bool.false
  evenM := Bool.false
  evenN := Bool.false

private def initial : BlockState where
  mem := fun _ _ => MemCell.real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0

private def allocated (normalizers : Nat) : RegionBounds :=
  fun r => if r = "M" then normalizers else 1

theorem partial_tail_bounds (candidate : Bool) :
    RealIO.Bounds (RealContract.io 2 1 parameters candidate) initial (allocated 4) := by
  constructor
  · rintro ⟨r,d,⟨⟩⟩ ha
    have hd : d.val = 0 := congrArg Fin.val (Fin.eq_zero d)
    simpa [RealContract.io, parameters, initial, allocated, hd] using ha
  · rintro ⟨d,k,⟨⟩⟩ ha
    have hd : d.val = 0 := congrArg Fin.val (Fin.eq_zero d)
    simpa [RealContract.io, parameters, initial, allocated, hd] using ha
  · rintro ⟨r,⟨⟩⟩
    simpa [RealContract.io, parameters, initial, allocated, RealSource.rows,
      RealSource.blocks, RealLoop.tripCount] using r.isLt
  · intro i ha
    simpa [RealContract.io, RealSource.outputBase, parameters, initial, allocated] using ha

theorem complete_source_safe (candidate : Bool) :
    (RealSource.kernel 2 1 parameters candidate).toAlgKernel.TraceSafe (allocated 4) initial :=
  RealSource.safe (allocated 4) initial 2 1 parameters candidate (by decide)
    (RealContract.bounds 2 1 parameters candidate initial (allocated 4) (partial_tail_bounds candidate))

theorem missing_padded_normalizer (candidate : Bool) :
    ¬ RealIO.Bounds (RealContract.io 2 1 parameters candidate) initial (allocated 3) := by
  intro h
  have hm := h.normalizer (⟨3, by decide⟩,PUnit.unit)
  norm_num [RealContract.io, parameters, initial, allocated] at hm

theorem empty_range_still_needs_key (candidate : Bool) :
    let p := { parameters with roundedContext := 0, evenN := Bool.true }
    ¬ RealIO.Bounds (RealContract.io 2 1 p candidate) initial (allocated 0) := by
  intro p h
  have hk := h.key ((0 : Fin 1),(1 : Fin 2),PUnit.unit) (Or.inl rfl)
  norm_num [p, RealContract.io, parameters, initial, allocated] at hk
  split_ifs at hk <;> omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end RealAttentionScoreIOTests
