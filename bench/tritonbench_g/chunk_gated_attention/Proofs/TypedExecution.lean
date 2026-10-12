import bench.tritonbench_g.chunk_gated_attention.Proofs.TypedExecutionSteps

/-! Observable typed history/final stores from complete original and candidate
executions. These values belong to the explicitly named abstract model, whose
dot/exp result is one; they are not GPU measurements or real-math specifications. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- History is stored before the corresponding update. The partial tile writes
only (k,v)=(2,2); neighboring cells preserve their original tagged values. -/
theorem values (gateK hasSeed storeFinal : Bool) :
    (FP.Structural.exec (engine model) (recurrent gateK hasSeed storeFinal).kernel initial).map
      (fun t => (t.mem "h" 26, t.mem "h" 35, t.mem "h" 27, t.mem "ht" 17, t.mem "ht" 16)) =
    some (.mk .bf16 (if hasSeed then 3 else 0), .mk .bf16 (if hasSeed then 4 else 1),
      .mk .bf16 99, .mk .fp32 (if storeFinal then (if hasSeed then 5 else 2) else 0), .mk .fp32 0) := by
  rw [execution]
  cases gateK <;> cases hasSeed <;> cases storeFinal
  all_goals
    simp only [Option.map_some, Bool.false_eq_true, ↓reduceIte, Option.some.injEq, Prod.mk.injEq]
    repeat' apply And.intro
    all_goals
      (conv_lhs => whnf) <;>
      norm_num [bop, Function.comp_def, numeric, ofFloat, toFloat,
        Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        State.write, State.setReg, TileShape.allIndices, List.finRange, List.ofFn_succ,
        String.reduceEq]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) (gateK hasSeed storeFinal : Bool) :
    FP.Structural.exec (engine model)
      (FPEquivProofs.recurrentSumLeft .fp16 .bf16 .bf16 .bf16 .fp32 "k" "v" "g" "h" "h0" "ht"
        15 3 1 15 3 1 18 3 1 5 3 3 4 2 2 2 gateK hasSeed storeFinal).kernel initial =
      some (finish gateK hasSeed storeFinal).val := by
  have he := FP.ProfiledRewrite.sound
    (FPEquivProofs.recurrent_equiv .fp16 .bf16 .bf16 .bf16 .fp32 "k" "v" "g" "h" "h0" "ht"
      15 3 1 15 3 1 18 3 1 5 3 3 4 2 2 2 gateK hasSeed storeFinal R)
  exact (he ℝ model domain (model_satisfies_rules R) initial (domain_holds gateK hasSeed storeFinal)).symm.trans
    (execution gateK hasSeed storeFinal)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedExecution
