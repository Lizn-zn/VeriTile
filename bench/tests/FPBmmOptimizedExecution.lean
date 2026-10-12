import bench.tritonbench_g.bmm_optimized.Proofs.TypedExecution
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedTactic false
set_option linter.unusedSimpArgs false

theorem reached_final_group :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.regs .nat [] "pid_m", t.regs .nat [] "pid_n",
        t.regs .nat [] "GROUP_SIZE", t.regs .nat [] "_i")) =
      some (some (fun _ => 2), some (fun _ => 0), some (fun _ => 1), some (fun _ => 1)) := by
  rw [original_execution]
  simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    conv_lhs => whnf
    try simp only [Option.some.injEq]
    try
      funext i
      cases i
      rfl

theorem output_and_mask :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "O" 45, t.mem "O" 48, t.mem "O" 50, t.mem "O" 44)) =
      some (.mk .fp16 2, .mk .fp16 2, .mk .fp16 9, .mk .fp16 9) := by
  rw [original_execution]
  simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    conv_lhs => whnf
    try norm_num [String.reduceEq, FP.Structural.bop, Function.comp_def, numeric, ofFloat, toFloat,
      Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]
    try simp only [String.reduceEq, if_false]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.TypedExecution
