import bench.tritonbench_g.bmm_optimized.Proofs.FP
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! Complete typed execution in the final shortened CTA group and through
two partial-K blocks. The consistent abstract model returns one per dot;
this witnesses rule-domain consistency and execution, not GPU arithmetic. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def original := FPEquivProofs.original .fp16 Bool.false Bool.false Bool.false "A" "B" "O" 5 5 3 2 4 2 2
def candidate := FPEquivProofs.dotFirst .fp16 Bool.false Bool.false Bool.false "A" "B" "O" 5 5 3 2 4 2 2

def initial : State ℝ where
  mem := fun r _ => .mk .fp16 (if r = "A" then 2 else if r = "B" then 3 else 9)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun ax => if ax = 0 then 3 else 2
  undef := fun d _ _ => defaultValue d

def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "_i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def preloop : {t : State ℝ // run (engine model) parts.before initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
    integralMod, numericLt, numericLe, numericEq, natLt, bop, initial,
    State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
  try rfl

def first : {t : State ℝ // run (engine model) parts.body
    (preloop.val.setReg "_i" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
    integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, preloop,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
  try rfl

def second : {t : State ℝ // run (engine model) parts.body
    (first.val.setReg "_i" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
    integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, preloop, first,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
  try rfl

def afterLoop : List ComputeStmt :=
  ((Source.dynamicRange? original.kernel.surfaceBody "_i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩).after

def finish : {t : State ℝ //
  run (engine model) afterLoop second.val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [afterLoop, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
    integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, preloop, first, second,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, FP.Structural.store]
  try rfl

theorem source_parts :
    original.kernel.surfaceBody = parts.before ++
      [.forRangeDyn "_i" parts.start parts.stop parts.stride parts.body] ++ afterLoop := by
  rfl

theorem loop_executes :
    step (engine model) (.forRangeDyn "_i" parts.start parts.stop parts.stride parts.body)
      preloop.val = some second.val := by
  simp only [step]
  change range (engine model) "_i" 0 2 1 parts.body preloop.val = some second.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, first.property, bind, Option.bind_some]
  change range (engine model) "_i" 1 2 1 parts.body first.val = some second.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, second.property, bind, Option.bind_some]
  change range (engine model) "_i" 2 2 1 parts.body second.val = some second.val
  rw [range]
  simp

theorem original_execution :
    FP.Structural.exec (engine model) original.kernel initial =
      some finish.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  simp only [source_parts, List.append_assoc, run_append, preloop.property,
    Option.bind_some, List.singleton_append]
  rw [run]
  simp only [loop_executes, bind, Option.bind_some, finish.property]

theorem domain_inhabited :
    ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original, FPEquivProofs.original, FPEquivProofs.requirementsFor,
    TypedSource.original, TypedSource.specialize, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Source.prefixBefore, Site.Holds, domain]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) candidate.kernel initial = some finish.val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.dot_first_equiv .fp16 Bool.false Bool.false Bool.false "A" "B" "O" 5 5 3 2 4 2 2 R)
  exact (h ℝ model domain (model_satisfies_rules R) initial domain_inhabited).symm.trans original_execution

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) original.kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) candidate.kernel initial = some t) ∧
    (∀ site ∈ original.domain, site.Holds model domain initial) :=
  ⟨model_satisfies_rules R, ⟨_, original_execution⟩, ⟨_, candidate_execution R⟩, domain_inhabited⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.TypedExecution
