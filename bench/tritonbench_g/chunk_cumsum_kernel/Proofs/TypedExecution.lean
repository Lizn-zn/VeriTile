import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! A complete, nonempty abstract execution: fp16 inputs, bf16 outputs,
program ID one, three chunks and a masked final lane. The chosen opaque scan
is identity and the sum returns one; these are not GPU numerical claims. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private def initial : State ℝ where
  mem := fun r _ => if r = "s" then .mk .fp16 2 else .mk .bf16 7
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

private def original := FPEquivProofs.original .fp16 .bf16 "s" "o" 5 2
private def candidate := FPEquivProofs.sumLeft .fp16 .bf16 "s" "o" 5 2
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
    Source.prefixBefore, Site.Holds, domain]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) candidate.kernel initial = some third.val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.sum_left_equiv .fp16 .bf16 "s" "o" 5 2 R)
  exact (h ℝ model domain (model_satisfies_rules R) initial domain_inhabited).symm.trans
    original_execution

theorem output_and_mask :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => ((t.mem "o" 4).read .bf16, (t.mem "o" 5).read .bf16,
        (t.mem "o" 6).read .bf16, (t.mem "o" 7).read .bf16,
        (t.mem "o" 8).read .bf16, (t.mem "o" 9).read .bf16,
        (t.mem "o" 10).read .bf16)) = some (7, 2, 2, 3, 3, 4, 7) := by
  rw [original_execution]
  have hv : TileShape.allIndices [2] = [(0, PUnit.unit), (1, PUnit.unit)] := by decide
  norm_num [third, second, first, entry, hv, State.write, Cell.read, initial,
    bop, numeric, TileShape.indexToList, String.reduceEq]
  dsimp only
  repeat' apply And.intro
  all_goals first | rfl | (change (2:ℝ) + 1 = 3; norm_num) | (change (2:ℝ) + 2 = 4; norm_num)

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
end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.TypedExecution
