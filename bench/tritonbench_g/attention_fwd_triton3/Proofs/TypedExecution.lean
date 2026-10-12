import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! Successful execution and normalization-domain witnesses for the complete
source and reciprocal candidate in FPModel's consistent abstract backend.
The fixture has two query heads sharing one KV head, 15 valid Q/KV rows in
16-lane blocks, a signed sliding window and one nonempty loop iteration.
Both tail masks, the alpha repair, fp16 casts and the END-arm M/Out stores execute.

This demonstrates that the FP theorem's execution and operand guards can hold
together. It is not GPU emulation or independent real attention correctness.
The staged states are computed from the actual source; source splits and the
candidate's shared prefix are checked by Lean, not assumed.
-/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open FPModel (model domain)
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

/- Complete pre-loop execution, including INIT, the Q tail and signed bounds. -/
private def initial : State ℝ where
  mem := fun r _ => if r = "M" ∨ r = "L" then .mk .real 1 else .mk .fp16 0
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 1 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d
private def original := FPEquivProofs.original "Q" "K" "V" "M" "Out" "L" 1
    512 256 16 1 240 240 16 1 240 240 16 1 512 256 16 1
    1 2 1 15 16 15 0 0 16 16 16 1 1 1 0 0 16
private def code := original.kernel.surfaceBody

private def preloop : {t : State ℝ // run (engine model) (code.take 21) initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [code, original, FPEquivProofs.original, TypedSource.original,
    ComputeKernel.surfaceBody, List.take_succ_cons, List.take_zero]
  simp (config := { maxSteps := 300000 }) [run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
  try rfl

/- The exact signed-loop body, with one iteration starting at key zero. -/
private def loopBody : List ComputeStmt := match code[21] with
  | .forRangeInt _ _ _ _ body => body
  | _ => []
private def loopEntry : State ℝ := preloop.val.setReg "start_n" .int [] (fun _ => 0)
private def iteration : {t : State ℝ // run (engine model) loopBody loopEntry = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [loopBody, code, original, FPEquivProofs.original, TypedSource.original,
    ComputeKernel.surfaceBody, List.getElem_cons_zero, List.getElem_cons_succ]
  simp (config := { maxSteps := 300000 }) [loopEntry, preloop, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
  try rfl

theorem denominator_after_iteration :
    iteration.val.regs .real [16] "l_i" = some (fun _ => 2) := by
  simp [iteration, bop]
  funext i
  change ((0 : ℝ) + (1.0 : ℝ)) * 1 + 1 = 2
  norm_num

theorem accumulator_after_iteration :
    iteration.val.regs .real [16, 16] "acc" = some (fun _ => 0) := by
  simp [iteration, bop]
  funext i
  change (0 : ℝ) * 1 + 0 = 0
  norm_num

/- Reassemble the signed loop and preserve the final M/Out stores. -/
theorem lo_value : preloop.val.regs .int [] "lo" = some (fun _ => 0) := by
  simp [preloop]
  rfl
theorem hi_value : preloop.val.regs .int [] "hi" = some (fun _ => 15) := by
  simp [preloop]

theorem loop_step : step (engine model) code[21] preloop.val = some iteration.val := by
  have hr : FP.Structural.rangeInt (engine model) "start_n" 0 15 16 loopBody
      preloop.val = some iteration.val := by
    rw [FP.Structural.rangeInt]
    change (run (engine model) loopBody loopEntry).bind
      (FP.Structural.rangeInt (engine model) "start_n" 16 15 16 loopBody) = some iteration.val
    simp only [iteration.property, Option.bind_some]
    simp [FP.Structural.rangeInt]
  have hs : code[21] = .forRangeInt "start_n" (.ref .int [] "lo")
      (.ref .int [] "hi") (.constNat 16) loopBody := rfl
  rw [hs, step]
  simpa only [evalOp_unfold, lo_value, hi_value, bind, Option.bind] using hr

theorem before_normalization :
    run (engine model) (code.take 22) initial = some iteration.val := by
  have hs : code.take 22 = code.take 21 ++ [code[21]] := rfl
  rw [hs, run_append, preloop.property]
  simp only [Option.bind_some, run, loop_step]
  rfl

private def postloop : {t : State ℝ // run (engine model) (code.drop 22) iteration.val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [code, original, FPEquivProofs.original, TypedSource.original,
    ComputeKernel.surfaceBody, List.drop_succ_cons, List.drop_zero]
  simp (config := { maxSteps := 300000 }) [iteration, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
  try rfl

theorem original_executes : ∃ t, FP.Structural.exec (engine model) original.kernel initial = some t := by
  refine ⟨postloop.val, ?_⟩
  change run (engine model) code initial = some postloop.val
  calc
    run (engine model) code initial =
        run (engine model) (code.take 22 ++ code.drop 22) initial :=
      congrArg (fun c => run (engine model) c initial) (List.take_append_drop 22 code).symm
    _ = _ := by simp only [run_append, before_normalization, Option.bind_some, postloop.property]

/- Check the actual denominator after the END arm's log2 update. -/
private def normalizationBefore : List ComputeStmt := match code[22] with
  | .ifThenElse _ yes _ => Source.prefixBefore yes "acc"
  | _ => []
private def normalizationCondition : Op .bool [] := match code[22] with
  | .ifThenElse cond _ _ => cond
  | _ => .constBool Bool.false

private def normalizationState : { t : State ℝ //
    run (engine model) normalizationBefore iteration.val = some t } := by
  refine ⟨?_, ?_⟩
  swap
  simp only [normalizationBefore, code, original, FPEquivProofs.original,
    TypedSource.original, ComputeKernel.surfaceBody,
    List.getElem_cons_succ, List.getElem_cons_zero, Source.prefixBefore]
  simp (config := { maxSteps := 300000 }) [iteration, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, ComputeDType.eraseDType, numeric, bop, State.setReg, TileShape.axisDim]
  try rfl

theorem normalized_denominator :
    normalizationState.val.regs .real [16] "l_i" = some (fun _ => 2) := by
  calc
    _ = iteration.val.regs .real [16] "l_i" := by rfl
    _ = _ := denominator_after_iteration

theorem domain_def : original.domain =
    [.branch (code.take 22) normalizationCondition Bool.true
      (.divBroadcast normalizationBefore [16, 16] 16 [1]
        (.ref .real [16, 16] "acc")
        (.expandDim ⟨1, by simp⟩ (.ref .real [16] "l_i")))] := rfl

theorem normalization_domain_inhabited :
    ∀ site ∈ original.domain, site.Holds model domain initial := by
  rw [domain_def]
  intro site member
  have he : site = _ := List.mem_singleton.mp member
  subst site
  intro t ht value hv hselected u hu a b ha hb
  rw [before_normalization] at ht
  have he : iteration.val = t := Option.some.inj ht
  subst t
  rw [normalizationState.property] at hu
  have he : normalizationState.val = u := Option.some.inj hu
  subst u
  simp only [evalOp_unfold, normalized_denominator, bind, Option.bind] at hb
  have he := Option.some.inj hb
  rw [← he]
  norm_num [domain]

/- Execute the candidate using its definitionally identical shared prefix. -/
private def reciprocal : Program := FPEquivProofs.reciprocal "Q" "K" "V" "M" "Out" "L" 1
    512 256 16 1 240 240 16 1 240 240 16 1 512 256 16 1
    1 2 1 15 16 15 0 0 16 16 16 1 1 1 0 0 16
private def reciprocalPost : {t : State ℝ // run (engine model) (reciprocal.kernel.surfaceBody.drop 22) iteration.val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [reciprocal, FPEquivProofs.reciprocal, FPEquivProofs.original, TypedSource.reciprocal,
    ComputeKernel.surfaceBody, List.drop_succ_cons, List.drop_zero]
  simp (config := { maxSteps := 300000 }) [iteration, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
    numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
    State.setReg, TileShape.axisDim, Region.cast]
  try rfl

theorem reciprocal_executes : ∃ t,
    FP.Structural.exec (engine model) reciprocal.kernel initial = some t := by
  refine ⟨reciprocalPost.val, ?_⟩
  have hs : reciprocal.kernel.surfaceBody.take 22 = code.take 22 := rfl
  change run (engine model) reciprocal.kernel.surfaceBody initial = some reciprocalPost.val
  calc
    _ = run (engine model)
        (reciprocal.kernel.surfaceBody.take 22 ++ reciprocal.kernel.surfaceBody.drop 22) initial :=
      congrArg (fun c => run (engine model) c initial)
        (List.take_append_drop 22 reciprocal.kernel.surfaceBody).symm
    _ = _ := by simp only [run_append, hs, before_normalization, Option.bind_some, reciprocalPost.property]

/-- The same backend and initial state satisfy the admitted scalar model,
the actual normalization guards and successful execution of both full kernels. -/
theorem admitted_model_is_nonvacuous (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain ∧
      (∀ site ∈ original.domain, site.Holds model domain initial) ∧
      (∃ t, FP.Structural.exec (engine model) original.kernel initial = some t) ∧
      (∃ t, FP.Structural.exec (engine model) reciprocal.kernel initial = some t) :=
  ⟨FPModel.model_satisfies_rules R, normalization_domain_inhabited,
    original_executes, reciprocal_executes⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.TypedExecution
