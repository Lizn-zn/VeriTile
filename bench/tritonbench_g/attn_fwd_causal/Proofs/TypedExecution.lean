import bench.tritonbench_g.attn_fwd_causal.Proofs.ExecutionPrefix

/-! Both complete sources execute in the same consistent abstract model,
with one nonempty iteration in each helper and a nonzero final denominator.
Both defined outer stages are covered. This witnesses satisfiable FP guards;
it is not GPU emulation or the independent real IO correctness theorem. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 3200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private def firstLoop (d : Bool) : ComputeStmt :=
  .forRangeDyn "start_n" (.ref .nat [] "first_lo") (.ref .nat [] "first_hi")
    (.constNat 2) (firstBody d)

private theorem first_bounds (d : Bool) :
    (preloop d).val.regs .nat [] "first_lo" = some (fun _ => firstStart d) ∧
    (preloop d).val.regs .nat [] "first_hi" = some (fun _ => firstStart d + 2) := by
  cases d <;> exact ⟨rfl,rfl⟩

theorem first_loop_executes (d : Bool) :
    step (engine model) (firstLoop d) (preloop d).val = some (firstIteration d).val := by
  have hr : FP.Structural.range (engine model) "start_n" (firstStart d) (firstStart d+2) 2
      (firstBody d) (preloop d).val = some (firstIteration d).val := by
    rw [FP.Structural.range]
    simp only [Nat.reduceEqDiff, ↓reduceIte, Nat.lt_add_of_pos_right (by decide : 0 < 2)]
    change (run (engine model) (firstBody d) (firstEntry d)).bind
      (FP.Structural.range (engine model) "start_n" (firstStart d+2) (firstStart d+2) 2 (firstBody d)) = _
    simp only [(firstIteration d).property, Option.bind_some]
    simp [FP.Structural.range]
  simp only [firstLoop, step, evalOp_unfold, (first_bounds d).1, (first_bounds d).2,
    bind, Option.bind]
  exact hr

private def between (d : Bool) : {t : State ℝ //
    run (engine model) (((code d).drop 29).take 6) (firstIteration d).val = some t} := by
  cases d
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [code, original, FPEquivProofs.original, TypedSource.original, stage,
      ComputeKernel.surfaceBody, List.drop_succ_cons, List.drop_zero,
      List.take_succ_cons, List.take_zero]
    simp (config := { maxSteps := 150000 }) [firstIteration, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      State.setReg, TileShape.axisDim, Region.cast]
    try rfl

private def secondEntry (d : Bool) : State ℝ :=
  (between d).val.setReg "start_n" .nat [] (fun _ => 2)
private def secondIteration (d : Bool) : {t : State ℝ //
    run (engine model) (secondBody d) (secondEntry d) = some t} := by
  cases d
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [secondBody, code, original, FPEquivProofs.original, TypedSource.original, stage,
      ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map_some, Option.getD_some,
      Bool.true_eq, ↓reduceIte, beq_self_eq_true]
    simp (config := { maxSteps := 150000 }) [secondEntry, between, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      State.setReg, TileShape.axisDim, Region.cast]
    try rfl

private def secondLoop (d : Bool) : ComputeStmt :=
  .forRangeDyn "start_n" (.ref .nat [] "second_lo") (.ref .nat [] "second_hi")
    (.constNat 2) (secondBody d)
private theorem second_bounds (d : Bool) :
    (between d).val.regs .nat [] "second_lo" = some (fun _ => 2) ∧
    (between d).val.regs .nat [] "second_hi" = some (fun _ => 4) := by
  cases d <;> exact ⟨rfl,rfl⟩

theorem second_loop_executes (d : Bool) :
    step (engine model) (secondLoop d) (between d).val = some (secondIteration d).val := by
  have hr : FP.Structural.range (engine model) "start_n" 2 4 2
      (secondBody d) (between d).val = some (secondIteration d).val := by
    rw [FP.Structural.range]
    change (run (engine model) (secondBody d) (secondEntry d)).bind
      (FP.Structural.range (engine model) "start_n" 4 4 2 (secondBody d)) = _
    simp only [(secondIteration d).property, Option.bind_some]
    simp [FP.Structural.range]
  simp only [secondLoop, step, evalOp_unfold, (second_bounds d).1, (second_bounds d).2,
    bind, Option.bind]
  exact hr

theorem before_normalization (d : Bool) :
    run (engine model) ((code d).take 36) initial = some (secondIteration d).val := by
  have hs : (code d).take 36 = (code d).take 28 ++ [firstLoop d] ++
      ((code d).drop 29).take 6 ++ [secondLoop d] := by cases d <;> rfl
  rw [hs]
  simp only [run_append, (preloop d).property, Option.bind_some, run,
    first_loop_executes, (between d).property, second_loop_executes, bind, Option.bind]

theorem denominator_after_helpers (d : Bool) :
    (secondIteration d).val.regs .real [2] "l_i" = some (fun _ => 3) := by
  cases d <;> simp [secondIteration, between, firstIteration, bop]
  all_goals
    funext i
    change (((0 : ℝ) + (1.0 : ℝ)) * 1 + 1) * 1 + 1 = 3
    norm_num

private def postloop (d optimized : Bool) : {t : State ℝ //
    run (engine model)
      ((if optimized then (reciprocal d).kernel else (original d).kernel).surfaceBody.drop 36)
      (secondIteration d).val = some t} := by
  cases d <;> cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [original, reciprocal, FPEquivProofs.original, FPEquivProofs.reciprocal,
      TypedSource.original, TypedSource.reciprocal, stage, Bool.false_eq_true,
      Bool.true_eq, ↓reduceIte, ComputeKernel.surfaceBody, List.drop_succ_cons, List.drop_zero]
    simp (config := { maxSteps := 150000 }) [secondIteration, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl

theorem source_executes (d optimized : Bool) :
    ∃ t, FP.Structural.exec (engine model)
      (if optimized then (reciprocal d).kernel else (original d).kernel) initial = some t := by
  refine ⟨(postloop d optimized).val, ?_⟩
  have hs : (if optimized then (reciprocal d).kernel else (original d).kernel).surfaceBody.take 36 =
      (code d).take 36 := by cases optimized <;> cases d <;> rfl
  change run (engine model)
    (if optimized then (reciprocal d).kernel else (original d).kernel).surfaceBody initial = _
  calc
    _ = run (engine model)
        ((if optimized then (reciprocal d).kernel else (original d).kernel).surfaceBody.take 36 ++
         (if optimized then (reciprocal d).kernel else (original d).kernel).surfaceBody.drop 36) initial :=
      congrArg (fun c => run (engine model) c initial)
        (List.take_append_drop 36 (if optimized then (reciprocal d).kernel else (original d).kernel).surfaceBody).symm
    _ = _ := by simp only [run_append, hs, before_normalization, Option.bind_some, (postloop d optimized).property]

theorem normalization_domain_inhabited (d : Bool) :
    (FPEquivProofs.divisionSite "Q" "K" "V" "QS" "KS" "Out"
      4 4 1 1 4 4 1 1 4 4 1 1 4 4 1 1 1 1 4 1 2 2 1 1 (stage d) (stage_valid d)).Holds
      model domain initial := by
  have hp : Source.prefixBefore (code d) "acc" 1 = (code d).take 36 := by cases d <;> rfl
  intro t ht a b ha hb
  change run (engine model) (Source.prefixBefore (code d) "acc" 1) initial = some t at ht
  rw [hp, before_normalization] at ht
  obtain rfl := Option.some.inj ht.symm
  simp only [evalOp_unfold, denominator_after_helpers, bind, Option.bind] at hb
  rw [← Option.some.inj hb]
  norm_num [domain]

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) (d : Bool) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) (original d).kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) (reciprocal d).kernel initial = some t) ∧
    (∀ site ∈ (original d).domain, site.Holds model domain initial) := by
  refine ⟨model_satisfies_rules R, source_executes d Bool.false, source_executes d Bool.true, ?_⟩
  intro site hs
  obtain rfl := List.mem_singleton.mp hs
  exact normalization_domain_inhabited d

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.TypedExecution
