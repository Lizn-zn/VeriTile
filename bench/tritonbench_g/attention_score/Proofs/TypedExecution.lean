import bench.tritonbench_g.attention_score.Proofs.FP
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! Nonempty full-source executions in a consistent model of the admitted
atoms. These witnesses exercise one two-row block, both source versions and
all four flags, including checked/unchecked loads and signed windows.
They establish satisfiable execution/domains, not concrete GPU behavior or the
independent real column-score formula. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

private def initial : State ℝ where
  mem := fun r _ => if r = "M" then .mk .fp32 0 else
    .mk .fp16 (if r = "Out" then 9 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def original (w c em en : Bool) := FPEquivProofs.original "Q" "K" "M" "Out"
  2 2 1 1 2 2 1 1 2 2 99 1 1 1 1 2 1 2 1 2 0 1 1 w c em en rfl
private def candidate (w c em en : Bool) := FPEquivProofs.scalarLeft "Q" "K" "M" "Out"
  2 2 1 1 2 2 1 1 2 2 99 1 1 1 1 2 1 2 1 2 0 1 1 w c em en rfl
private def program (w c em en optimized : Bool) :=
  if optimized then candidate w c em en else original w c em en
private def code (w c em en optimized : Bool) := (program w c em en optimized).kernel.surfaceBody

private def preloop (en : Bool) :
    {t : State ℝ // run (engine model) ((code Bool.false Bool.false Bool.false en Bool.false).take 16) initial = some t} := by
  cases en
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.scalarLeft,
      FPEquivProofs.original, TypedSource.original, TypedSource.scalarLeft,
      Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody,
      List.take_succ_cons, List.take_zero]
    simp (config := { maxSteps := 300000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric,
      integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      FP.Structural.store, State.setReg, TileShape.axisDim, Region.cast]
    try rfl

private def loopBody (w c em en optimized : Bool) : List ComputeStmt :=
  match ((code w c em en optimized)[16]? : Option ComputeStmt) with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
private def loopEntry (en : Bool) := (preloop en).val.setReg "start_m" .nat [] (fun _ => 0)

private def iteration (w c em en optimized : Bool) :
    {t : State ℝ // run (engine model) (loopBody w c em en optimized) (loopEntry en) = some t} := by
  cases w <;> cases c <;> cases em <;> cases en <;> cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [loopBody, code, program, original, candidate, FPEquivProofs.scalarLeft,
      FPEquivProofs.original, TypedSource.original, TypedSource.scalarLeft,
      Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody,
      List.getElem?_cons_zero, List.getElem?_cons_succ]
    simp (config := { maxSteps := 300000 }) [loopEntry, preloop, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, initial,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
      numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
      State.setReg, TileShape.axisDim, Region.cast]
    try rfl

theorem after_loop (w c em en optimized : Bool) :
    run (engine model) ((code w c em en optimized).take 17) initial =
      some (iteration w c em en optimized).val := by
  have hs : (code w c em en optimized).take 17 =
      (code Bool.false Bool.false Bool.false en Bool.false).take 16 ++
        [.forRangeDyn "start_m" (.ref .nat [] "lo") (.ref .nat [] "hi")
          (.constNat 2) (loopBody w c em en optimized)] := by
    cases optimized <;> rfl
  rw [hs, run_append, (preloop en).property, Option.bind_some]
  have hlo : (preloop en).val.regs .nat [] "lo" = some (fun _ => 0) := by cases en <;> rfl
  have hhi : (preloop en).val.regs .nat [] "hi" = some (fun _ => 2) := by cases en <;> rfl
  have hr : FP.Structural.range (engine model) "start_m" 0 2 2
      (loopBody w c em en optimized) (preloop en).val =
      some (iteration w c em en optimized).val := by
    rw [FP.Structural.range]
    change (run (engine model) (loopBody w c em en optimized) (loopEntry en)).bind
      (FP.Structural.range (engine model) "start_m" 2 2 2 (loopBody w c em en optimized)) = _
    simp only [(iteration w c em en optimized).property, Option.bind_some]
    simp [FP.Structural.range]
  simp only [run, step, evalOp_unfold, hlo, hhi, bind, Option.bind]
  rw [hr]

private def postloop (w c em en optimized : Bool) :
    {t : State ℝ // run (engine model) ((code w c em en optimized).drop 17)
      (iteration w c em en optimized).val = some t} := by
  cases w <;> cases c <;> cases em <;> cases en <;> cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [code, program, original, candidate, FPEquivProofs.scalarLeft,
      FPEquivProofs.original, TypedSource.original, TypedSource.scalarLeft,
      Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody,
      List.drop_succ_cons, List.drop_zero]
    simp (config := { maxSteps := 300000 }) [iteration, loopEntry, preloop, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, initial,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod,
      numericLt, numericLe, numericEq, natLt, bop, FP.Structural.store,
      State.setReg, TileShape.axisDim, Region.cast]
    try rfl

theorem executes (w c em en optimized : Bool) : ∃ t,
    FP.Structural.exec (engine model) (program w c em en optimized).kernel initial = some t := by
  refine ⟨(postloop w c em en optimized).val, ?_⟩
  change run (engine model) (code w c em en optimized) initial = _
  calc
    _ = run (engine model) ((code w c em en optimized).take 17 ++
        (code w c em en optimized).drop 17) initial :=
      congrArg (fun code => run (engine model) code initial)
        (List.take_append_drop 17 (code w c em en optimized)).symm
    _ = _ := by simp only [run_append, after_loop, Option.bind_some,
      (postloop w c em en optimized).property]

theorem domain_inhabited (w c em en : Bool) :
    ∀ site ∈ (original w c em en).domain, site.Holds model domain initial := by
  intro site member
  change site ∈ FPEquivProofs.requirements "Q" "K" "M" "Out"
    2 2 1 1 2 2 1 1 2 2 99 1 1 1 1 2 1 2 1 2 0 1 1 w c em en rfl at member
  simp only [FPEquivProofs.requirements, TypedSource.original, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, reduceIte] at member
  obtain rfl := List.mem_singleton.mp member
  intro t ht a b d ha hb hd cur u reachable stride positive v hpref x y hx hy
  exact ⟨fun _ => True.intro, True.intro⟩

/-- Full nonempty executions in every flag combination coexist with all the
selected scalar atoms and the actual per-iteration finite-operand domain. -/
theorem consistent_execution (R : FP.ScalarArithmetic.Rules) (w c em en : Bool) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) (original w c em en).kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) (candidate w c em en).kernel initial = some t) ∧
    (∀ site ∈ (original w c em en).domain, site.Holds model domain initial) :=
  ⟨model_satisfies_rules R, executes w c em en Bool.false, executes w c em en Bool.true,
    domain_inhabited w c em en⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.TypedExecution
