import bench.tritonbench_g.chunk_cumsum_vector.Proofs.FP
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.FPModel

/-! Complete bf16-input/fp16-output executions, nonzero feature/head IDs, three
time chunks and both partial masks. Opaque dot and sum use the explicit
abstract backend. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private def initial : State ℝ where
  mem := fun r _ => if r = "s" then .mk .bf16 2 else .mk .fp16 7
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

private def original := FPEquivProofs.original .bf16 .fp16 "s" "o" 15 3 1 5 3 2 2
private def candidate := FPEquivProofs.sumLeft .bf16 .fp16 "s" "o" 15 3 1 5 3 2 2
private def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "i_t").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

private def entry : {t : State ℝ // run (engine model) parts.before initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initial,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg]
  try rfl

private def first : {t : State ℝ // run (engine model) parts.body (entry.val.setReg "i_t" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initial,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry]
  try rfl

private def second : {t : State ℝ // run (engine model) parts.body (first.val.setReg "i_t" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initial,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, first]
  try rfl

private def third : {t : State ℝ // run (engine model) parts.body (second.val.setReg "i_t" .nat [] (fun _ => 2)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initial,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, first, second]
  try rfl

private theorem source_parts : original.kernel.surfaceBody = parts.before ++
    [.forRangeDyn "i_t" parts.start parts.stop parts.stride parts.body] := rfl

private theorem loop_executes :
    step (engine model) (.forRangeDyn "i_t" parts.start parts.stop parts.stride parts.body)
      entry.val = some third.val := by
  simp only [step]
  change range (engine model) "i_t" 0 3 1 parts.body entry.val = some third.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 3 from by decide,
    ↓reduceIte, first.property, bind, Option.bind_some]
  change range (engine model) "i_t" 1 3 1 parts.body first.val = some third.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 3 from by decide,
    ↓reduceIte, second.property, bind, Option.bind_some]
  change range (engine model) "i_t" 2 3 1 parts.body second.val = some third.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (2 : Nat) < 3 from by decide,
    ↓reduceIte, third.property, bind, Option.bind_some]
  change range (engine model) "i_t" 3 3 1 parts.body third.val = some third.val
  rw [range]
  simp

theorem original_execution :
    FP.Structural.exec (engine model) original.kernel initial = some third.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  simp only [source_parts, run_append, entry.property]
  simp only [Option.bind_some, run, loop_executes, bind]

theorem domain_inhabited :
    ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original, FPEquivProofs.original, FPEquivProofs.requirementsFor,
    TypedSource.original, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Source.prefixBefore, Source.conditional?, Site.Holds, domain]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) candidate.kernel initial = some third.val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.sum_left_equiv .bf16 .fp16 "s" "o" 15 3 1 5 3 2 2 R)
  exact (h ℝ model domain (model_satisfies_rules R) initial domain_inhabited).symm.trans
    original_execution


/-- Partial row and column masks preserve their neighboring cells. -/
theorem output_and_masks :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => ((t.mem "o" 16).read .fp16, (t.mem "o" 17).read .fp16,
        (t.mem "o" 18).read .fp16, (t.mem "o" 20).read .fp16,
        (t.mem "o" 21).read .fp16, (t.mem "o" 23).read .fp16,
        (t.mem "o" 26).read .fp16, (t.mem "o" 29).read .fp16,
        (t.mem "o" 30).read .fp16, (t.mem "o" 32).read .fp16)) =
        some (7, 1, 7, 1, 7, 2, 2, 3, 7, 7) := by
  rw [original_execution]
  have hm : TileShape.allIndices [2, 2] =
      [(0, 0, PUnit.unit), (0, 1, PUnit.unit),
        (1, 0, PUnit.unit), (1, 1, PUnit.unit)] := by decide
  have hv : TileShape.allIndices [2] = [(0, PUnit.unit), (1, PUnit.unit)] := by decide
  norm_num [third, second, first, entry, hm, hv, State.write, Cell.read, initial,
    bop, numeric, TileShape.indexToList, String.reduceEq]
  dsimp only
  repeat' apply And.intro
  all_goals rfl

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) original.kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) candidate.kernel initial = some t) ∧
    (∀ site ∈ original.domain, site.Holds model domain initial) :=
  ⟨model_satisfies_rules R, ⟨_, original_execution⟩,
    ⟨_, candidate_execution R⟩, domain_inhabited⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.TypedExecution
