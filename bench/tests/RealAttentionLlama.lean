import bench.tritonbench_g.attention_llama.Proofs.RealMemory

/-! The real bridge reaches an actual output for both full sources and both
causal flags. The second query lane is masked; its old memory cell survives. -/
namespace RealAttentionLlamaTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open RealKernel
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private def initial : BlockState :=
  { mem := fun r o =>
      if r = "V" then MemCell.real (if o = 0 then 7 else 11) else if r = "Out" then MemCell.real 99 else MemCell.real 0
    regs := fun _ _ _ => none
    pids := fun _ => 0 }

private def kernel (candidate causal : Bool) :=
  RealOutput.source "Q" "K" "V" "Out" 1
    2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 2 1 candidate causal

private theorem output_injective : Function.Injective (fun idx : TileIndex [2, 1] =>
    (initial.pids 0 * 2 + idx.1.val) * 1 + idx.2.1.val * 1) := by
  simpa only [initial, Nat.zero_mul, Nat.zero_add, Nat.mul_one] using
    (rowMajor2D_inj (I := 2) (J := 1) 0 1 (by decide))

private theorem query_is_zero : alQTileG initial "Q" 1 2 2 1 1 1 2 1 =
    fun _ => 0 := by
  funext i
  unfold alQTileG
  split <;> rfl

private theorem key_is_zero : alKTileG initial "K" 1 2 2 1 1 2 1 =
    fun _ => 0 := rfl

private theorem values_are_distinct : alVTileG initial "V" 1 2 2 1 1 2 1 =
    fun i => if i.1.val = 0 then 7 else 11 := by
  funext i
  have hd : i.2.1.val = 0 := by omega
  simp [alVTileG, alBase, initial, hd, BlockState.readMem, MemCell.real]

/-- Non-causal output averages both values to nine; causal output is seven.
The masked second query and the V input
are preserved as complete memory cells, not merely equal real readbacks. -/
theorem successful_output_and_mask_frame (candidate causal : Bool) :
    ∃ sF, exec (kernel candidate causal).eraseDType.toAlgKernel initial = some sF ∧
      sF.readMem "Out" 0 = (if causal then 7 else 9) ∧ sF.mem "Out" 1 = .real 99 ∧ sF.mem "V" 0 = .real 7 := by
  cases causal
  case' «false» =>
    obtain ⟨sF, hrun, hread, hframe⟩ := RealOutput.noncausal_correct
        "Q" "K" "V" "Out" 1
        2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 2 1 candidate initial 1
        (by decide) (by decide) (by decide) rfl output_injective (by intros; rfl)
  case' «true» =>
    obtain ⟨sF, hrun, hread, hframe⟩ := RealOutput.causal_correct
        "Q" "K" "V" "Out" 1
        2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 2 1 candidate initial 1
        (by decide) (by decide) rfl (by decide) (by decide)
        output_injective (by intros; rfl)
  all_goals
    refine ⟨sF, hrun, ?_, ?_, ?_⟩
    · have ho := hread (0, 0, PUnit.unit) (by decide)
      change sF.readMem "Out" 0 = _ at ho
      rw [query_is_zero, key_is_zero, values_are_distinct] at ho
      norm_num [attentionReal, attentionRealCausalBlock, scaledScore,
        initial, Fin.sum_univ_two, Finset.sum_ite_eq, Finset.sum_ite_eq'] at ho
      try simp +contextual [Finset.sum_ite_eq', Fin.sum_univ_two] at ho
      exact ho
    · have hf := hframe "Out" 1 (Or.inr (by
        intro i hi heq
        simp only [alActive, alRow, initial] at hi
        simp only [alOutOffset, alBase, alRow, initial] at heq
        have hd := i.2.1.isLt
        omega))
      simpa [initial] using hf
    · simpa [initial] using hframe "V" 0 (Or.inl (by decide))

/- The allocation has space for only the active Q row. A proof that accidentally
requires every masked lane to be in bounds would reject this valid launch. -/
private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["Q", "K", "V", "Out"]
  base := fun r => if r = "Q" then 0 else if r = "K" then 4 else if r = "V" then 8 else 12
  extent := fun r => if r = "Q" then 1 else if r ∈ (["K", "V", "Out"] : List RegionName) then 2 else 0

private theorem allocation_disjoint : allocation.Disjoint := by
  simp +decide [FlatAlloc.Disjoint, allocation, Region.mk.injEq]

private theorem allocation_covered :
    ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := by
  intro r hr
  simp_all [allocation]

private theorem query_bounds (i : TileIndex [2, 1]) (hi : alActive initial 1 2 i) :
    alBase initial 1 2 2 + alRow initial 2 i.1 * 1 + i.2.1.val * 1 <
      allocation.extent "Q" := by
  simp only [alActive, alRow, initial] at hi
  simp +decide [alBase, alRow, initial, allocation]
  have hd := i.2.1.isLt
  omega

private theorem key_bounds (r : Nat) (hr : r < 2 * 1) (d : Fin 1) :
    alBase initial 1 2 2 + r * 1 + d.val * 1 < allocation.extent "K" := by
  simp +decide [alBase, initial, allocation, Region.mk.injEq]
  have hd := d.isLt
  omega

private theorem value_bounds (r : Nat) (hr : r < 2 * 1) (d : Fin 1) :
    alBase initial 1 2 2 + r * 1 + d.val * 1 < allocation.extent "V" := by
  simp +decide [alBase, initial, allocation, Region.mk.injEq]
  have hd := d.isLt
  omega

private theorem output_bounds (i : TileIndex [2, 1]) (hi : alActive initial 1 2 i) :
    alOutOffset initial 1 2 2 1 1 2 i < allocation.extent "Out" := by
  simp only [alActive, alRow, initial] at hi
  simp +decide [alOutOffset, alBase, alRow, initial, allocation, Region.mk.injEq]
  have hd := i.2.1.isLt
  omega

/-- The same nonempty execution now runs in one allocation, with distinct input
and output intervals. Both branch results and complete-cell preservation remain. -/
theorem flat_output_and_mask_frame (candidate causal : Bool) :
    ∃ sF, exec (allocation.flattenKernel (kernel candidate causal).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some sF ∧
      sF.readMem "flat" 12 = (if causal then 7 else 9) ∧
      sF.mem "flat" 13 = .real 99 ∧ sF.mem "flat" 8 = .real 7 := by
  cases causal
  case' «false» =>
    obtain ⟨sF, hrun, hread, hframe⟩ := RealMemory.noncausal_correct
        "Q" "K" "V" "Out" 1
        2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 2 1
        allocation allocation_disjoint rfl allocation_covered candidate initial 1
        (by decide) (by decide) (by decide) rfl output_injective (by intros; rfl)
        query_bounds key_bounds value_bounds output_bounds
  case' «true» =>
    obtain ⟨sF, hrun, hread, hframe⟩ := RealMemory.causal_correct
        "Q" "K" "V" "Out" 1
        2 2 1 1 2 2 1 1 2 2 1 1 2 2 1 1 1 1 2 0 2 2 1
        allocation allocation_disjoint rfl allocation_covered candidate initial 1
        (by decide) (by decide) rfl (by decide) (by decide)
        output_injective (by intros; rfl) query_bounds key_bounds value_bounds output_bounds
  all_goals
    refine ⟨sF, hrun, ?_, ?_, ?_⟩
    · have ho := hread (0, 0, PUnit.unit) (by decide)
      change sF.readMem "flat" 12 = _ at ho
      rw [query_is_zero, key_is_zero, values_are_distinct] at ho
      norm_num [attentionReal, attentionRealCausalBlock, scaledScore,
        initial, Fin.sum_univ_two, Finset.sum_ite_eq, Finset.sum_ite_eq'] at ho
      try simp +contextual [Finset.sum_ite_eq', Fin.sum_univ_two] at ho
      exact ho
    · have hf := hframe "flat" 13 (Or.inr (by
        intro i hi heq
        simp only [alActive, alRow, initial] at hi
        simp +decide [alOutOffset, alBase, alRow, initial, FlatAlloc.addr, allocation, Region.mk.injEq] at heq
        have hd := i.2.1.isLt
        omega))
      exact hf.trans (by rfl)
    · have hf := hframe "flat" 8 (Or.inr (by
        intro i hi heq
        simp +decide [alOutOffset, alBase, alRow, initial, FlatAlloc.addr, allocation, Region.mk.injEq] at heq
        omega))
      exact hf.trans (by rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end RealAttentionLlamaTests
