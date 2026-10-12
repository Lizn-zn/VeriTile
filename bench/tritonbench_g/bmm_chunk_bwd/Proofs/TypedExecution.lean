import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! A nonempty complete execution: batch=1, chunk=1, group=1; chunk length
three, two contraction blocks, a partial column tile and a masked output row.
A is fp16 and Dout is bf16, so the source selects a bf16 dot. The abstract
dot returns one per block. Residuals are three; output padding starts at nine.
This witnesses execution and rule-domain consistency, not GPU arithmetic. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def original (hasResidual : Bool) := FPEquivProofs.original .fp16 .bf16 .bf16 .fp16
  "A" "Dout" "Db" "Res" 7 4 3 2
  100 10 4 1 120 24 16 4 1 200 12 5 1 180 11 6 1 4 2 2 hasResidual

def candidate (hasResidual : Bool) := FPEquivProofs.residualLeft .fp16 .bf16 .bf16 .fp16
  "A" "Dout" "Db" "Res" 7 4 3 2
  100 10 4 1 120 24 16 4 1 200 12 5 1 180 11 6 1 4 2 2 hasResidual

def initial : State ℝ where
  mem := fun r _ => if r = "A" then .mk .fp16 2
    else if r = "Dout" then .mk .bf16 2
    else if r = "Res" then .mk .bf16 3
    else .mk .fp16 9
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 2 then 3 else 1
  numPids := fun _ => 8
  undef := fun d _ _ => defaultValue d

def parts : Source.DynamicRange :=
  (Source.dynamicRange? (original Bool.false).kernel.surfaceBody "cs").getD
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
    (preloop.val.setReg "cs" .nat [] (fun _ => 0)) = some t} := by
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
    (first.val.setReg "cs" .nat [] (fun _ => 1)) = some t} := by
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

def afterLoop (hasResidual : Bool) : List ComputeStmt :=
  ((Source.dynamicRange? (original hasResidual).kernel.surfaceBody "cs").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩).after

def finish (hasResidual : Bool) : {t : State ℝ //
    run (engine model) (afterLoop hasResidual) second.val = some t} := by
  cases hasResidual
  all_goals
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

theorem source_parts (hasResidual : Bool) :
    (original hasResidual).kernel.surfaceBody = parts.before ++
      [.forRangeDyn "cs" parts.start parts.stop parts.stride parts.body] ++ afterLoop hasResidual := by
  cases hasResidual <;> rfl

theorem loop_executes :
    step (engine model) (.forRangeDyn "cs" parts.start parts.stop parts.stride parts.body)
      preloop.val = some second.val := by
  simp only [step]
  change range (engine model) "cs" 0 2 1 parts.body preloop.val = some second.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, first.property, bind, Option.bind_some]
  change range (engine model) "cs" 1 2 1 parts.body first.val = some second.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, second.property, bind, Option.bind_some]
  change range (engine model) "cs" 2 2 1 parts.body second.val = some second.val
  rw [range]
  simp

theorem original_execution (hasResidual : Bool) :
    FP.Structural.exec (engine model) (original hasResidual).kernel initial =
      some (finish hasResidual).val := by
  change run (engine model) (original hasResidual).kernel.surfaceBody initial = _
  simp only [source_parts, List.append_assoc, run_append, preloop.property,
    Option.bind_some, List.singleton_append]
  rw [run]
  simp only [loop_executes, bind, Option.bind_some, (finish hasResidual).property]

theorem domain_inhabited (hasResidual : Bool) :
    ∀ site ∈ (original hasResidual).domain, site.Holds model domain initial := by
  simp [original, FPEquivProofs.original, FPEquivProofs.requirementsFor,
    TypedSource.original, TypedSource.specialize, ComputeKernel.surfaceBody,
    Source.conditional?, Source.prefixBefore, Site.Holds, domain]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) (hasResidual : Bool) :
    FP.Structural.exec (engine model) (candidate hasResidual).kernel initial =
      some (finish hasResidual).val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.residual_equiv .fp16 .bf16 .bf16 .fp16 "A" "Dout" "Db" "Res"
      7 4 3 2 100 10 4 1 120 24 16 4 1 200 12 5 1 180 11 6 1 4 2 2 hasResidual R)
  exact (h ℝ model domain (model_satisfies_rules R) initial
    (domain_inhabited hasResidual)).symm.trans (original_execution hasResidual)

/-- The complete execution computes the in-body limit and visits both blocks. -/
theorem reached_partial_chunk (hasResidual : Bool) :
    (FP.Structural.exec (engine model) (original hasResidual).kernel initial).map
      (fun t => (t.regs .nat [] "chunk_size_limit", t.regs .nat [] "cs")) =
      some (some (fun _ => 3), some (fun _ => 1)) := by
  rw [original_execution]
  cases hasResidual <;> rfl

/-- Live cells receive the two dot contributions and optional residual.
The masked fourth row and the masked fourth column keep their original tags
and values. These addresses use nonzero batch, chunk and group offsets. -/
theorem output_and_mask (hasResidual : Bool) :
    (FP.Structural.exec (engine model) (original hasResidual).kernel initial).map
      (fun t => (t.mem "Db" 255, t.mem "Db" 267, t.mem "Db" 279,
        t.mem "Db" 256, t.mem "Db" 291)) =
      some (.mk .fp16 (if hasResidual then 5 else 2),
        .mk .fp16 (if hasResidual then 5 else 2), .mk .fp16 (if hasResidual then 5 else 2),
        .mk .fp16 9, .mk .fp16 9) := by
  rw [original_execution]
  cases hasResidual
  all_goals
    simp only [Option.map_some, Bool.false_eq_true, ↓reduceIte, Option.some.injEq, Prod.mk.injEq]
    repeat' apply And.intro
    all_goals
      conv_lhs => whnf
      try norm_num [bop, Function.comp_def, numeric, ofFloat, toFloat,
        Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) (hasResidual : Bool) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) (original hasResidual).kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) (candidate hasResidual).kernel initial = some t) ∧
    (∀ site ∈ (original hasResidual).domain, site.Holds model domain initial) :=
  ⟨model_satisfies_rules R, ⟨_, original_execution hasResidual⟩,
    ⟨_, candidate_execution R hasResidual⟩, domain_inhabited hasResidual⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.TypedExecution
