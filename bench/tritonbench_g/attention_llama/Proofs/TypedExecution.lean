import bench.tritonbench_g.attention_llama.Proofs.FP
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! A successful nonempty execution for each causal flag in the shared,
consistent abstract backend. One of two query/key lanes is masked out, so
explicit zero padding, causal selection, fp16 casts and masked output execute.
This witnesses satisfiable execution and domain conditions, not GPU emulation
or real attention correctness. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

private def initial : State ℝ where
  mem := fun _ _ => .mk .fp16 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def original (causal : Bool) := FPEquivProofs.original "Q" "K" "V" "Out" 1
    4 4 2 1 4 4 2 1 4 4 2 1 4 4 2 1
    1 1 1 0 2 2 2 causal
private def candidate (causal : Bool) := FPEquivProofs.carryFirst "Q" "K" "V" "Out" 1
    4 4 2 1 4 4 2 1 4 4 2 1 4 4 2 1
    1 1 1 0 2 2 2 causal
private def program (causal optimized : Bool) :=
  if optimized then candidate causal else original causal
private def code (causal optimized : Bool) := (program causal optimized).kernel.surfaceBody

private def preloop (causal optimized : Bool) :
    {t : State ℝ // run (engine model) ((code causal optimized).take 19) initial = some t} := by
  cases causal <;> cases optimized
  case false.false =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [ run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case false.true =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [ run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case true.false =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [ run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case true.true =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [ run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl

private def loopBody (causal optimized : Bool) : List ComputeStmt :=
  match (code causal optimized)[19]'(by cases causal <;> cases optimized <;> decide) with
  | .forRangeDyn _ _ _ _ body => body
  | _ => []
private def loopEntry (causal optimized : Bool) : State ℝ :=
  (preloop causal optimized).val.setReg "start_n" .nat [] (fun _ => 0)

private def iteration (causal optimized : Bool) :
    {t : State ℝ // run (engine model) (loopBody causal optimized) (loopEntry causal optimized) = some t} := by
  cases causal <;> cases optimized
  case false.false =>
    refine ⟨?_, ?_⟩
    swap
    simp only [loopBody, code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [loopEntry, preloop,  run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case false.true =>
    refine ⟨?_, ?_⟩
    swap
    simp only [loopBody, code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [loopEntry, preloop,  run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case true.false =>
    refine ⟨?_, ?_⟩
    swap
    simp only [loopBody, code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [loopEntry, preloop,  run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case true.true =>
    refine ⟨?_, ?_⟩
    swap
    simp only [loopBody, code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [loopEntry, preloop,  run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl

theorem loop_stop (causal optimized : Bool) :
    (preloop causal optimized).val.regs .nat [] "block_n_end" =
      some (fun _ => if causal then 2 else 1) := by
  cases causal <;> cases optimized <;> rfl

theorem after_loop (causal optimized : Bool) :
    run (engine model) ((code causal optimized).take 20) initial = some (iteration causal optimized).val := by
  have hs : (code causal optimized).take 20 =
      (code causal optimized).take 19 ++ [(code causal optimized)[19]'(by cases causal <;> cases optimized <;> decide)] := by
    cases causal <;> cases optimized <;> rfl
  rw [hs, run_append, (preloop causal optimized).property, Option.bind_some]
  have hloop : (code causal optimized)[19]'(by cases causal <;> cases optimized <;> decide) =
      .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end")
        (.constNat 2) (loopBody causal optimized) := by cases causal <;> cases optimized <;> rfl
  have hr : FP.Structural.range (engine model) "start_n" 0 (if causal then 2 else 1) 2
      (loopBody causal optimized) (preloop causal optimized).val = some (iteration causal optimized).val := by
    rw [FP.Structural.range]
    have hstop : 0 < (if causal then 2 else 1 : Nat) := by cases causal <;> cases optimized <;> decide
    simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, hstop, ↓reduceDIte]
    change (run (engine model) (loopBody causal optimized) (loopEntry causal optimized)).bind
      (FP.Structural.range (engine model) "start_n" 2 (if causal then 2 else 1) 2
        (loopBody causal optimized)) = _
    simp only [(iteration causal optimized).property, Option.bind_some]
    cases causal <;> simp [FP.Structural.range]
  have hstep : step (engine model) ((code causal optimized)[19]'(by cases causal <;> cases optimized <;> decide))
      (preloop causal optimized).val = some (iteration causal optimized).val := by
    rw [hloop, step]
    simpa only [evalOp_unfold, loop_stop, bind, Option.bind] using hr
  simp only [run, hstep, Option.bind_some]
  rfl

private def postloop (causal optimized : Bool) :
    {t : State ℝ // run (engine model) ((code causal optimized).drop 20) (iteration causal optimized).val = some t} := by
  cases causal <;> cases optimized
  case false.false =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop,  run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case false.true =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop,  run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case true.false =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop,  run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl
  case true.true =>
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.carryFirst,
      FPEquivProofs.original, TypedSource.original, TypedSource.carryFirst, Bool.false_eq_true, ↓reduceIte,
      ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero,
      List.getElem_cons_zero, List.getElem_cons_succ]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop,  run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl

private theorem executes (causal optimized : Bool) : ∃ t,
    FP.Structural.exec (engine model) (program causal optimized).kernel initial = some t := by
  refine ⟨(postloop causal optimized).val, ?_⟩
  change run (engine model) (code causal optimized) initial = _
  calc
    _ = run (engine model) ((code causal optimized).take 20 ++ (code causal optimized).drop 20) initial :=
      congrArg (fun c => run (engine model) c initial)
        (List.take_append_drop 20 (code causal optimized)).symm
    _ = _ := by simp only [run_append, after_loop, Option.bind_some, (postloop causal optimized).property]

theorem addition_domain_inhabited (causal : Bool) :
    ∀ site ∈ (original causal).domain, site.Holds model domain initial := by
  intro site member
  change site ∈ FPEquivProofs.requirements "Q" "K" "V" "Out" 1
    4 4 2 1 4 4 2 1 4 4 2 1 4 4 2 1 1 1 1 0 2 2 2 causal at member
  simp only [FPEquivProofs.requirements, TypedSource.original, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, reduceIte] at member
  obtain rfl := List.mem_singleton.mp member
  intro t ht a b d ha hb hd cur u reachable stride positive v hpref x y hx hy
  exact ⟨fun _ => True.intro, fun _ => True.intro⟩

theorem original_executes (causal : Bool) : ∃ t,
    FP.Structural.exec (engine model) (original causal).kernel initial = some t :=
  executes causal Bool.false

theorem candidate_executes (causal : Bool) : ∃ t,
    FP.Structural.exec (engine model) (candidate causal).kernel initial = some t :=
  executes causal Bool.true

/-- A nonempty execution with these guards coexists with all selected atoms. -/
theorem consistent_execution (R : FP.ScalarArithmetic.Rules) (causal : Bool) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) (original causal).kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) (candidate causal).kernel initial = some t) ∧
    (∀ site ∈ (original causal).domain, site.Holds model domain initial) :=
  ⟨model_satisfies_rules R, original_executes causal, candidate_executes causal,
    addition_domain_inhabited causal⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionLlama.TypedExecution
