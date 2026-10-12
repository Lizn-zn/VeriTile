import bench.tritonbench_g.attention_score.Proofs.RealExecution

/-! End-to-end real regressions for a partial final query block and the source's
key-even-controlled query mask. These execute both complete source variants;
they are region-memory checks, not GPU executions or allocation-safety tests. -/
namespace RealAttentionScoreTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionScore
noncomputable section

private def initial : BlockState where
  mem := fun _ _ => MemCell.real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0

private def source (evenN candidate : Bool) : ComputeKernel :=
  (if candidate then TypedSource.scalarLeft else TypedSource.original) "Q" "K" "M" "Out"
    0 0 1 1 0 0 1 1 0 0 99 1 1 1 1 3 (if evenN then 2 else 1) 2 1 2
    0 0 0 Bool.false Bool.false Bool.false evenN rfl

/-- N=1, range stop=3 and B=2 visit four padded query lanes. When evenN is
true the source does not zero those query lanes, so zero inputs contribute 4.
With evenN=false only query 0 survives, and output column 1 is not stored. -/
theorem partial_range_and_query_mask (evenN candidate : Bool) :
    ∃ t, stepStmts (source evenN candidate).eraseDType.toAlgKernel.body initial = some t ∧
      t.readMem "Out" 0 = (if evenN then 4 else 1) ∧
      t.readMem "Out" 1 = (if evenN then 4 else 0) ∧
      t.mem "Out" 2 = initial.mem "Out" 2 := by
  obtain ⟨t,he,ho,hframe⟩ := RealExecution.source_correct initial "Q" "K" "M" "Out"
    0 0 1 1 0 0 1 1 0 0 99 1 1 1 1 3 (if evenN then 2 else 1) 2 1 2
    0 0 0 Bool.false Bool.false Bool.false evenN candidate rfl (by decide)
  have h0 := ho (0 : Fin 2)
  have h1 := ho (1 : Fin 2)
  refine ⟨t, ?_, ?_, ?_, ?_⟩
  · cases candidate <;> exact he
  · cases evenN
    all_goals simp only [initial, Nat.zero_div, Nat.zero_mod, Nat.zero_mul, Nat.zero_add,
      Fin.val_zero, Bool.false_eq_true, ↓reduceIte, Nat.reduceLT] at h0
    all_goals simp +instances only [RealPrelude.config, RealLoop.Config.column, RealLoop.tripCount,
      RealFormula.column, Nat.reduceAdd, Nat.reduceSub, Nat.reduceDiv] at h0
    all_goals simp only [Fin.sum_univ_two] at h0
    all_goals norm_num [RealPrelude.config, RealLoop.Config.column, RealLoop.Config.inputs,
      RealLoop.tripCount, RealFormula.column, RealFormula.weight, RealFormula.windowAllows,
      RealFormula.score, RealLoads.inputs, initial, BlockState.readMem, Fin.sum_univ_succ] at h0 ⊢
    all_goals exact h0
  · cases evenN
    all_goals norm_num [RealPrelude.config, RealLoop.Config.column, RealLoop.Config.inputs,
      RealLoop.tripCount, RealFormula.column, RealFormula.weight, RealFormula.windowAllows,
      RealFormula.score, RealLoads.inputs, initial, BlockState.readMem, Fin.sum_univ_succ] at h1 ⊢
    all_goals exact h1
  · apply hframe "Out" 2
    right
    intro i hi
    simp only [initial, Nat.zero_div, Nat.zero_mod, Nat.zero_mul, Nat.zero_add]
    have := i.isLt
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end RealAttentionScoreTests
