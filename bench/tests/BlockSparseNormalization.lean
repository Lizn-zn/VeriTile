import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalizationMath

/-! Execute the checked normalization segment with a fully masked row and a
visible row. The second accumulator is absent in the single-channel case.
These real-semantics checks do not stand in for the complete kernel proof. -/
namespace BlockSparseNormalizationTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealNormalizationValues
noncomputable section

private def inputs : Inputs 2 2 2 where
  score := fun r j => if r.val = 0 ∨ j.val = 1 then ⊥ else (0 : WithBot ℝ)
  maximum := fun r => if r.val = 0 then ⊥ else (0 : WithBot ℝ)
  normalizer := fun r => if r.val = 0 then 0 else 1
  accumulator := fun r _ => if r.val = 0 then 0 else 5
  secondAccumulator := fun r _ => if r.val = 0 then 0 else -3

private def initial (two : Bool) : BlockState :=
  let s : BlockState :=
    { mem := fun _ _ => MemCell.real 13, regs := fun _ _ _ => none,
      pids := fun a => a + 1, undef := fun _ _ => 17, numPids := fun _ => 23 }
  let s := s.setReg "qk" .real [2, 2] (extendedMatrix inputs.score)
  let s := s.setReg "m_i" .real [2] (extendedRow inputs.maximum)
  let s := s.setReg "l_i" .real [2] (row inputs.normalizer)
  let s := s.setReg "acc" .real [2, 2] (matrix inputs.accumulator)
  if two then s.setReg "acc2" .real [2, 2] (matrix inputs.secondAccumulator) else s

theorem no_second_binding : (initial Bool.false).regs .real [2, 2] "acc2" = none := by
  simp [initial, BlockState.setReg]

theorem actual_normalization_executes (two : Bool) :
    stepStmts (RealNormalizationSource.code 2 2 2 (if two then 2 else 1)) (initial two) =
      some (RealNormalization.result (initial two) inputs (if two then 2 else 1)) := by
  apply RealNormalization.executes (by decide)
  all_goals cases two <;> simp [initial]

theorem full_state_frame (two : Bool) :
    { RealNormalization.result (initial two) inputs (if two then 2 else 1)
      with regs := (initial two).regs } = initial two :=
  RealNormalization.state_frame _ _ _

theorem single_channel_keeps_second_absent :
    (RealNormalization.result (initial Bool.false) inputs 1).regs .real [2, 2] "acc2" = none := by
  rw [RealNormalization.second_disabled _ _ _ (by decide)]
  exact no_second_binding

private theorem block_max_zero : inputs.blockMax = fun r => if r.val = 0 then ⊥ else (0 : WithBot ℝ) := by
  funext r
  fin_cases r <;> simp [Inputs.blockMax, inputs, Finset.univ_fin2]

private theorem shifted_zero : StreamingMath.shifted (0 : WithBot ℝ) 0 = 1 := by
  change Real.exp (0 - 0) = 1
  norm_num

private theorem probabilities : inputs.scaledProbability = fun r j =>
    if r.val = 0 ∨ j.val = 1 then 0 else (1 / 2 : ℝ) := by
  funext r j
  simp only [Inputs.scaledProbability, Inputs.probability, Inputs.probabilityScale,
    Inputs.beta, Inputs.alpha, Inputs.nextNormalizer, Inputs.blockNormalizer,
    Inputs.nextMax, block_max_zero]
  fin_cases r <;> fin_cases j
  all_goals
    norm_num [inputs, shifted_zero, Fin.sum_univ_two]

theorem masked_and_visible_probabilities (two : Bool) :
    (RealNormalization.result (initial two) inputs (if two then 2 else 1)).regs .real [2, 2] "p" =
      some (matrix (fun r j => if r.val = 0 ∨ j.val = 1 then 0 else (1 / 2 : ℝ))) := by
  simpa only [probabilities] using (RealNormalization.outputs (initial two) inputs (if two then 2 else 1)).probability

/-- Checking the actual first dot after normalization exposes an incorrect
normalization order: the visible result is (5 + 9)/2, the masked result is zero. -/
theorem first_value_dot (two : Bool) :
    let t := (RealNormalization.result (initial two) inputs (if two then 2 else 1)).setReg
      "v" .real [2, 2] (matrix (fun j _ => if j.val = 0 then 9 else -99))
    stepStmt (RealAccumulationSource.valueDot 2 2 2 "acc") t =
      some (t.setReg "acc" .real [2, 2] (matrix (fun r _ => if r.val = 0 then 0 else 7))) := by
  dsimp only
  have ho := RealNormalization.outputs (initial two) inputs (if two then 2 else 1)
  have hrun := RealNormalizationMath.value_dot_executes
    ((RealNormalization.result (initial two) inputs (if two then 2 else 1)).setReg
      "v" .real [2, 2] (matrix (fun j _ => if j.val = 0 then 9 else -99))) "acc"
    inputs.scaledProbability (fun j _ => if j.val = 0 then 9 else -99) inputs.scaledAccumulator
    (by simpa using ho.probability) (by simp) (by simpa using ho.accumulator)
  have hf : (fun r d => inputs.scaledAccumulator r d +
      ∑ j : Fin 2, inputs.scaledProbability r j * (if j.val = 0 then 9 else -99)) =
      fun r (_ : Fin 2) => if r.val = 0 then (0 : ℝ) else 7 := by
    funext r d
    simp only [probabilities, Inputs.scaledAccumulator, Inputs.accumulatorScale,
      Inputs.nextNormalizer, Inputs.alpha, Inputs.beta, Inputs.nextMax,
      Inputs.blockNormalizer, Inputs.probability, block_max_zero]
    fin_cases r
    all_goals
      norm_num [inputs, shifted_zero, Fin.sum_univ_two]
  simpa only [hf] using hrun

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseNormalizationTests
