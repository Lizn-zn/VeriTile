import bench.tritonbench_g.block_sparse_attn.Proofs.CSRMath

/-! Boundary regressions for real block-sparse row mathematics. These are
not complete-source execution certificates or GPU floating-point observations. -/
namespace BlockSparseMathTests

open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BlockSparseAttn
noncomputable section

/-- An all-masked first block does not prevent a later visible block from
contributing in the project's real semantics. -/
theorem masked_prefix_recovers :
    StreamingMath.accumulated (fun n (_ : Fin 1) => if n = 0 then ⊥ else (0 : WithBot ℝ))
      (fun _ _ => 7) 2 = 7 := by
  rw [StreamingMath.accumulated_eq_formula]
  norm_num [StreamingMath.moment, StreamingMath.mass, StreamingMath.weight,
    WithBot.realExp, Finset.sum_range_succ]

theorem masked_suffix_preserves :
    StreamingMath.accumulated (fun n (_ : Fin 1) => if n = 0 then (0 : WithBot ℝ) else ⊥)
      (fun _ _ => 7) 2 = 7 := by
  rw [StreamingMath.accumulated_eq_formula]
  norm_num [StreamingMath.moment, StreamingMath.mass, StreamingMath.weight,
    WithBot.realExp, Finset.sum_range_succ]

private def schedule : CSRMath.Schedule := ⟨1, 5, 13, 1⟩
private def inputs : CSRMath.Inputs where
  query := fun _ d => if d = 0 then 2 else 3
  key := fun _ d => if d = 0 then 5 else 7
  value := fun k _ => if k = 0 then 2 else 11
  rowIndex := fun a => if a = 6 then -1 else if a = 7 then 2 else -99
  columnIndex := fun a => if a = 12 ∨ a = 13 then 0 else if a = 14 then 1 else -99

/-- Signed visits -1,0,1 use the independent column stride, not row stride. -/
theorem signed_schedule :
    CSRMath.count schedule inputs = 3 ∧
    CSRMath.selectedKey schedule inputs 1 0 0 = 0 ∧
    CSRMath.selectedKey schedule inputs 1 1 0 = 0 ∧
    CSRMath.selectedKey schedule inputs 1 2 0 = 1 := by decide

/-- Duplicate CSR columns are counted twice: (2 + 2 + 11) / 3 = 5. -/
theorem duplicate_columns_count :
    StreamingMath.accumulated (CSRMath.score schedule inputs 1 1 1 1 0)
      (fun visit lane => inputs.value (CSRMath.selectedKey schedule inputs 1 visit lane) 0)
      (CSRMath.count schedule inputs) = 5 := by
  rw [CSRMath.recurrence_eq_attention]
  simp only [CSRMath.attention, signed_schedule.1, Finset.sum_range_succ,
    Finset.sum_range_zero, zero_add]
  norm_num [CSRMath.first, CSRMath.selectedKey, schedule, inputs,
    Finset.sum_range_succ, CSRMath.dotScore]

theorem one_or_two_channel_blocks :
    CSRMath.dotScore inputs 1 1 0 0 1 = 10 ∧
    CSRMath.dotScore inputs 1 2 0 0 1 = 31 ∧
    CSRMath.dotScore inputs 1 5 0 0 1 = 31 := by
  norm_num [CSRMath.dotScore, inputs]

/-- A reversed/empty signed range performs zero updates and has an empty sum. -/
theorem empty_schedule :
    CSRMath.attention schedule { inputs with rowIndex := fun a => if a = 6 then 2 else -1 }
      1 1 1 1 0 1 = 0 := by
  have hc : CSRMath.count schedule
      { inputs with rowIndex := fun a => if a = 6 then 2 else -1 } = 0 := by decide
  simp [CSRMath.attention, hc]

#guard_msgs (drop info) in
#auditModuleAxioms
end

end BlockSparseMathTests
