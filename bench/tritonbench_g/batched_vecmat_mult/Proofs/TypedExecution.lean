import bench.tritonbench_g.batched_vecmat_mult.Proofs.FP
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! Both complete fp32 sources execute in one consistent model of the admitted
atoms. The fixture has two K iterations and a 2×3 output tile at nonzero program
IDs. It witnesses a nonempty execution/domain; opaque model reductions do not
represent a numerical GPU experiment or the independent real GEMV formula. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 2200000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

private def initial : State ℝ where
  mem := fun _ _ => .mk .fp32 2
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

private def original := FPEquivProofs.originalProgram "A" "B" "Out" 4 6 4 2 3 2
private def candidate := FPEquivProofs.blockFirstProgram "A" "B" "Out" 4 6 4 2 3 2
private def program (optimized : Bool) := if optimized then candidate else original

private def code (optimized : Bool) := (program optimized).kernel.surfaceBody
private def body (optimized : Bool) :=
  (Source.dynamicRange? (code optimized) "k_index").map (·.body) |>.getD []

private def preloop : {t : State ℝ //
    run (engine model) ((code Bool.false).take 5) initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [code, program, original, FPEquivProofs.originalProgram,
    TypedSource.original, Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody,
    List.take_succ_cons, List.take_zero]
  simp [run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, initial, engine, Algebra.withDefaultPrecision, resolvePrecision,
    model, ComputeDType.eraseDType, numeric, integralFloorDiv,
    State.setReg, bop]
  try rfl

private def firstEntry := preloop.val.setReg "k_index" .nat [] (fun _ => 0)
private def firstIteration (optimized : Bool) : {t : State ℝ //
    run (engine model) (body optimized) firstEntry = some t} := by
  cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [body, code, program, original, candidate, FPEquivProofs.originalProgram,
      FPEquivProofs.blockFirstProgram, TypedSource.original, TypedSource.blockFirst,
      Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody, Source.dynamicRange?,
      Option.map_some, Option.getD_some, beq_self_eq_true]
    dsimp [firstEntry, preloop]
    simp (config := { maxSteps := 100000 }) [run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
      numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
      State.setReg, TileShape.axisDim, Region.cast]
    try rfl

private def secondEntry (optimized : Bool) :=
  (firstIteration optimized).val.setReg "k_index" .nat [] (fun _ => 1)
private def secondIteration (optimized : Bool) : {t : State ℝ //
    run (engine model) (body optimized) (secondEntry optimized) = some t} := by
  cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [body, code, program, original, candidate, FPEquivProofs.originalProgram,
      FPEquivProofs.blockFirstProgram, TypedSource.original, TypedSource.blockFirst,
      Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody, Source.dynamicRange?,
      Option.map_some, Option.getD_some, beq_self_eq_true]
    dsimp [secondEntry, firstIteration]
    simp (config := { maxSteps := 100000 }) [run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
      numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
      State.setReg, TileShape.axisDim, Region.cast]
    try rfl

private def loopStmt (optimized : Bool) : ComputeStmt :=
  .forRangeDyn "k_index" (.constNat 0) (.ref .nat [] "k_blocks") (.constNat 1) (body optimized)

private theorem loop_executes (optimized : Bool) :
    step (engine model) (loopStmt optimized) preloop.val = some (secondIteration optimized).val := by
  have hb : preloop.val.regs .nat [] "k_blocks" = some (fun _ => 2) := by
    simp [preloop, integralFloorDiv, FP.Structural.bop]
    funext i
    change (4 : Nat) / 2 = 2
    decide
  simp only [loopStmt, step, evalOp_unfold, hb, bind, Option.bind]
  rw [FP.Structural.range]
  change (run (engine model) (body optimized) firstEntry).bind
    (FP.Structural.range (engine model) "k_index" 1 2 1 (body optimized)) = _
  rw [(firstIteration optimized).property, Option.bind_some, FP.Structural.range]
  change (run (engine model) (body optimized) (secondEntry optimized)).bind
    (FP.Structural.range (engine model) "k_index" 2 2 1 (body optimized)) = _
  simp only [(secondIteration optimized).property, Option.bind_some]
  simp [FP.Structural.range]

private def postloop (optimized : Bool) : {t : State ℝ //
    run (engine model) ((code optimized).drop 6) (secondIteration optimized).val = some t} := by
  cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.originalProgram,
      FPEquivProofs.blockFirstProgram, TypedSource.original, TypedSource.blockFirst,
      Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody,
      List.drop_succ_cons, List.drop_zero]
    dsimp [secondIteration]
    simp (config := { maxSteps := 100000 }) [run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
      numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
      State.setReg, TileShape.axisDim, Region.cast]
    try rfl

theorem source_executes (optimized : Bool) :
    ∃ t, FP.Structural.exec (engine model) (program optimized).kernel initial = some t := by
  refine ⟨(postloop optimized).val, ?_⟩
  have hs : code optimized = (code Bool.false).take 5 ++ [loopStmt optimized] ++
      (code optimized).drop 6 := by cases optimized <;> rfl
  change run (engine model) (code optimized) initial = _
  conv_lhs => rw [hs]
  simp only [run_append, preloop.property, Option.bind_some, run,
    loop_executes, bind, Option.bind, (postloop optimized).property]

theorem domain_inhabited : ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original, FPEquivProofs.originalProgram, FPEquivProofs.requirements,
    TypedSource.original, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Site.Holds, domain]

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) original.kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) candidate.kernel initial = some t) ∧
    (∀ site ∈ original.domain, site.Holds model domain initial) := by
  exact ⟨model_satisfies_rules R, source_executes Bool.false,
    source_executes Bool.true, domain_inhabited⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.TypedExecution
