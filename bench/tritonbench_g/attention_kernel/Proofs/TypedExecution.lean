import bench.tritonbench_g.attention_kernel.Proofs.FP
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! Nonempty complete executions for both host storage types, using the shared
consistent abstract backend. This witnesses executable bias loads, typed dots,
casts, output stores and the normalization guards together. The dot/reduction
backend is opaque; this is neither GPU emulation nor real attention correctness. -/
namespace VeriTile.Bench.TritonBenchG.AttentionKernel.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

private def initial (storage : TypedSource.Storage) : State ℝ where
  mem := fun _ _ => match storage with
    | .fp16 => .mk .fp16 0
    | .bf16 => .mk .bf16 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def original (storage : TypedSource.Storage) :=
  FPEquivProofs.original storage "Q" "K" "V" "B0" "Out" 1
      4 2 1 4 2 1 4 2 1 4 2 1 8 4
      1 1 2 0 2 4 2 2 2
private def code (storage : TypedSource.Storage) := (original storage).kernel.surfaceBody

private def preloop (storage : TypedSource.Storage) :
    {t : State ℝ // run (engine model) ((code storage).take 19) (initial storage) = some t} := by
  cases storage with
  | fp16 =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.FP16.original, TypedSource.BF16.original,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  | bf16 =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.FP16.original, TypedSource.BF16.original,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
    try rfl

private def loopBody (storage : TypedSource.Storage) : List ComputeStmt :=
  match (code storage)[19]'(by cases storage <;> decide) with
  | .forRangeDyn _ _ _ _ body => body
  | _ => []
private def loopEntry (storage : TypedSource.Storage) : State ℝ :=
  (preloop storage).val.setReg "start_n" .nat [] (fun _ => 0)

private def iteration (storage : TypedSource.Storage) :
    {t : State ℝ // run (engine model) (loopBody storage) (loopEntry storage) = some t} := by
  cases storage with
  | fp16 =>
    refine ⟨?_, ?_⟩
    swap
    simp only [loopBody, code, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.FP16.original, TypedSource.BF16.original,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [loopEntry, preloop, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  | bf16 =>
    refine ⟨?_, ?_⟩
    swap
    simp only [loopBody, code, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.FP16.original, TypedSource.BF16.original,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [loopEntry, preloop, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
    try rfl

theorem denominator_after_iteration (storage : TypedSource.Storage) :
    (iteration storage).val.regs .real [2] "l_i" = some (fun _ => 1) := by
  cases storage <;> simp [iteration, bop]
  all_goals
    funext i
    change (0 : ℝ) * 1 + 1 = 1
    norm_num

theorem accumulator_after_iteration (storage : TypedSource.Storage) :
    (iteration storage).val.regs .real [2, 2] "acc" = some (fun _ => 0) := by
  cases storage <;> simp [iteration, bop]
  all_goals
    funext i
    change (0 : ℝ) * 1 + 0 = 0
    norm_num

theorem before_normalization (storage : TypedSource.Storage) :
    run (engine model) ((code storage).take 20) (initial storage) =
      some (iteration storage).val := by
  have hs : (code storage).take 20 = (code storage).take 19 ++ [(code storage)[19]'(by cases storage <;> decide)] := by
    cases storage <;> rfl
  rw [hs, run_append, (preloop storage).property, Option.bind_some]
  have hloop : (code storage)[19]'(by cases storage <;> decide) = .forRangeDyn "start_n"
      (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat 2) (loopBody storage) := by
    cases storage <;> rfl
  have hlo : (preloop storage).val.regs .nat [] "lo" = some (fun _ => 0) := by
    cases storage <;> rfl
  have hhi : (preloop storage).val.regs .nat [] "hi" = some (fun _ => 2) := by
    cases storage <;> rfl
  have hr : FP.Structural.range (engine model) "start_n" 0 2 2 (loopBody storage)
      (preloop storage).val = some (iteration storage).val := by
    rw [FP.Structural.range]
    change (run (engine model) (loopBody storage) (loopEntry storage)).bind
      (FP.Structural.range (engine model) "start_n" 2 2 2 (loopBody storage)) = _
    simp only [(iteration storage).property, Option.bind_some]
    simp [FP.Structural.range]
  have hstep : step (engine model) ((code storage)[19]'(by cases storage <;> decide))
      (preloop storage).val = some (iteration storage).val := by
    rw [hloop, step]
    simpa only [evalOp_unfold, hlo, hhi, bind, Option.bind] using hr
  simp only [run, hstep, Option.bind_some]
  rfl


private def postloop (storage : TypedSource.Storage) :
    {t : State ℝ // run (engine model) ((code storage).drop 20) (iteration storage).val = some t} := by
  cases storage with
  | fp16 =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.FP16.original, TypedSource.BF16.original,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  | bf16 =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.FP16.original, TypedSource.BF16.original,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
    try rfl

theorem original_executes (storage : TypedSource.Storage) : ∃ t,
    FP.Structural.exec (engine model) (original storage).kernel (initial storage) = some t := by
  refine ⟨(postloop storage).val, ?_⟩
  change run (engine model) (code storage) (initial storage) = _
  calc
    _ = run (engine model) ((code storage).take 20 ++ (code storage).drop 20)
        (initial storage) := congrArg (fun c => run (engine model) c (initial storage))
          (List.take_append_drop 20 (code storage)).symm
    _ = _ := by simp only [run_append, before_normalization, Option.bind_some,
        (postloop storage).property]

theorem normalization_domain_inhabited (storage : TypedSource.Storage) :
    ∀ site ∈ (original storage).domain, site.Holds model domain (initial storage) := by
  have hd : (original storage).domain =
      [.divBroadcast ((code storage).take 20) [2, 2] 2 [1]
        (.ref .real [2, 2] "acc")
        (.expandDim ⟨1, by simp⟩ (.ref .real [2] "l_i"))] := by
    cases storage <;> rfl
  rw [hd]
  intro site member
  obtain rfl := List.mem_singleton.mp member
  intro t ht a b ha hb
  rw [before_normalization] at ht
  obtain rfl := Option.some.inj ht
  simp only [evalOp_unfold, denominator_after_iteration, bind, Option.bind] at hb
  rw [← Option.some.inj hb]
  norm_num [domain]

private def reciprocal (storage : TypedSource.Storage) :=
  FPEquivProofs.reciprocal storage "Q" "K" "V" "B0" "Out" 1
      4 2 1 4 2 1 4 2 1 4 2 1 8 4
      1 1 2 0 2 4 2 2 2

private def reciprocalPost (storage : TypedSource.Storage) :
    {t : State ℝ // run (engine model) ((reciprocal storage).kernel.surfaceBody.drop 20) (iteration storage).val = some t} := by
  cases storage with
  | fp16 =>
    refine ⟨?_, ?_⟩
    swap
    simp only [reciprocal, FPEquivProofs.reciprocal, FPEquivProofs.original, TypedSource.reciprocal,
      TypedSource.FP16.reciprocal, TypedSource.BF16.reciprocal,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  | bf16 =>
    refine ⟨?_, ?_⟩
    swap
    simp only [reciprocal, FPEquivProofs.reciprocal, FPEquivProofs.original, TypedSource.reciprocal,
      TypedSource.FP16.reciprocal, TypedSource.BF16.reciprocal,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
    try rfl

theorem reciprocal_executes (storage : TypedSource.Storage) : ∃ t,
    FP.Structural.exec (engine model) (reciprocal storage).kernel (initial storage) = some t := by
  refine ⟨(reciprocalPost storage).val, ?_⟩
  have hp : (reciprocal storage).kernel.surfaceBody.take 20 = (code storage).take 20 := by
    cases storage <;> rfl
  change run (engine model) (reciprocal storage).kernel.surfaceBody (initial storage) = _
  calc
    _ = run (engine model) ((reciprocal storage).kernel.surfaceBody.take 20 ++
        (reciprocal storage).kernel.surfaceBody.drop 20) (initial storage) :=
      congrArg (fun c => run (engine model) c (initial storage))
        (List.take_append_drop 20 (reciprocal storage).kernel.surfaceBody).symm
    _ = _ := by simp only [run_append, hp, before_normalization, Option.bind_some,
        (reciprocalPost storage).property]

/-- These same nonempty executions and guards coexist with the admitted rules. -/
theorem consistent_execution (R : FP.ScalarArithmetic.Rules) (storage : TypedSource.Storage) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) (original storage).kernel (initial storage) = some t) ∧
    (∃ t, FP.Structural.exec (engine model) (reciprocal storage).kernel (initial storage) = some t) ∧
    (∀ site ∈ (original storage).domain, site.Holds model domain (initial storage)) :=
  ⟨model_satisfies_rules R, original_executes storage, reciprocal_executes storage,
    normalization_domain_inhabited storage⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionKernel.TypedExecution
