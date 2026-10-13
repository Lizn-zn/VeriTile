import bench.tritonbench_g.flash_attn.Proofs.NumericTypedExecution

/-! Numerical readback from the complete typed fp16 witness, under the
explicit exact-real backend. These values are not claimed to be IEEE results. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem noncausal_output_cells :
    (post Bool.false).val.mem "O" 20 = Cell.mk .fp16 (some 1) ∧
    (post Bool.false).val.mem "L" 3 = Cell.mk .real (some 2) := by
  have log_four : Real.log (4 : ℝ) / Real.log 2 = 2 := by
    have hl : Real.log (4 : ℝ) = 2 * Real.log 2 := by
      convert Real.log_pow (2 : ℝ) 2 using 1 <;> norm_num
    rw [hl]
    have hn : Real.log (2 : ℝ) ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
    field_simp
  simp [post,afterLoop,second,first,entry,initial,storageType,model,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,Tile.dot,
    TileShape.axisDim,TileShape.insertAxisIndex,Fin.sum_univ_succ,
    WithBot.realAdd,WithBot.realSub,WithBot.realMul,WithBot.realDiv,
    WithBot.realExp2,WithBot.realLog2,FP.Structural.store,State.write,
    TileShape.allIndices,List.finRange_succ,log_four,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,
    two_nsmul,WithBot.coe_add_coe_eq]
  all_goals norm_num [log_four,WithBot.some_eq_coe,WithBot.none_eq_bot,
    ←WithBot.coe_add,Option.map₂,Option.map,Option.bind,Fintype.card_fin,
    two_nsmul]
  erw [Fintype.card_fin]
  norm_num [log_four]
  exact ⟨rfl,rfl⟩


theorem original_numerical_execution :
    ∃ t, FP.Structural.exec (engine model) (original Bool.false).kernel (initial .fp16 .fp16) = some t ∧
      t.mem "O" 20 = Cell.mk .fp16 (some 1) ∧ t.mem "L" 3 = Cell.mk .real (some 2) := by
  exact ⟨(post Bool.false).val,original_execution Bool.false,noncausal_output_cells⟩

theorem candidate_numerical_execution :
    ∃ t, FP.Structural.exec (engine model)
        (FPProof.program Bool.true .fp16 .fp16 "Q" "K" "V" "L" "O" 1
          50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 Bool.false).kernel
        (initial .fp16 .fp16) = some t ∧
      t.mem "O" 20 = Cell.mk .fp16 (some 1) ∧ t.mem "L" 3 = Cell.mk .real (some 2) := by
  exact ⟨(post Bool.false).val,candidate_execution Bool.false,noncausal_output_cells⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
