import bench.tritonbench_g.bmm_chunk_fwd.Proofs.FP
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! A nonempty complete execution: batch=1, chunk=1, group=1; chunk length
three, two contraction blocks, a partial column tile and a masked output row.
A is fp16 and B is bf16, so the source selects a bf16 dot. The abstract
dot returns one per block. Signed sequence IDs include -1 and -2; initial output is fp16 nine.
This witnesses execution and rule-domain consistency, not GPU arithmetic. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def original (hasSeq : Bool) := FPEquivProofs.original .fp16 .bf16 .fp16
  "A" "B" "Out" ⟨"Seq"⟩ 7 4 3 2
  100 10 4 1 120 11 5 1 200 24 16 4 1 90 1 4 2 2 Bool.true hasSeq

def candidate (hasSeq : Bool) := FPEquivProofs.dotFirst .fp16 .bf16 .fp16
  "A" "B" "Out" ⟨"Seq"⟩ 7 4 3 2
  100 10 4 1 120 11 5 1 200 24 16 4 1 90 1 4 2 2 Bool.true hasSeq

def initial : State ℝ where
  mem := fun r o => if r = "A" then .mk .fp16 2
    else if r = "B" then .mk .bf16 2
    else if r = "Seq" then .mk .int (if o = 95 then -2 else -1)
    else .mk .fp16 9
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 2 then 3 else 1
  numPids := fun _ => 8
  undef := fun d _ _ => defaultValue d

def outer (hasSeq : Bool) : Source.Conditional :=
  (Source.conditional? (original hasSeq).kernel.surfaceBody
    (fun body => (Source.dynamicRange? body "k").isSome)).getD
    ⟨[], .constBool Bool.false, [], [], []⟩

def parts (hasSeq : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (outer hasSeq).yes "k").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def entry : {t : State ℝ // run (engine model) (outer Bool.true).before initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [outer, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
    Option.map, Option.getD, Option.isSome, Bool.false_eq_true, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
    integralMod, numericLt, numericLe, numericEq, natLt, bop, initial,
    State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
  try rfl

def preloop (hasSeq : Bool) : {t : State ℝ // run (engine model) (parts hasSeq).before entry.val = some t} := by
  cases hasSeq
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [parts, outer, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
      Option.map, Option.getD, Option.isSome, Bool.false_eq_true, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, entry,
      State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl

def first (hasSeq : Bool) : {t : State ℝ // run (engine model) (parts hasSeq).body
    ((preloop hasSeq).val.setReg "k" .nat [] (fun _ => 0)) = some t} := by
  cases hasSeq
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [parts, outer, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
      Option.map, Option.getD, Option.isSome, Bool.false_eq_true, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, entry, preloop,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl

def second (hasSeq : Bool) : {t : State ℝ // run (engine model) (parts hasSeq).body
    ((first hasSeq).val.setReg "k" .nat [] (fun _ => 1)) = some t} := by
  cases hasSeq
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [parts, outer, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
      Option.map, Option.getD, Option.isSome, Bool.false_eq_true, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, entry, preloop, first,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl

def afterLoop (hasSeq : Bool) : List ComputeStmt := (parts hasSeq).after

def finish (hasSeq : Bool) : {t : State ℝ //
    run (engine model) (afterLoop hasSeq) (second hasSeq).val = some t} := by
  cases hasSeq
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [afterLoop, parts, outer, original, FPEquivProofs.original, TypedSource.original,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.conditional?, Source.dynamicRange?,
      Option.map, Option.getD, Option.isSome, Bool.false_eq_true, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, entry, preloop, first, second,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, FP.Structural.store]
    try rfl


theorem source_parts (hasSeq : Bool) :
    (original hasSeq).kernel.surfaceBody = (outer hasSeq).before ++
      [.ifThen (outer hasSeq).condition ((parts hasSeq).before ++
        [.forRangeDyn "k" (parts hasSeq).start (parts hasSeq).stop
          (parts hasSeq).stride (parts hasSeq).body] ++ afterLoop hasSeq)] := by
  cases hasSeq <;> rfl

theorem entry_executes (hasSeq : Bool) :
    run (engine model) (outer hasSeq).before initial = some entry.val := by
  cases hasSeq <;> exact entry.property

theorem loop_executes (hasSeq : Bool) :
    step (engine model) (.forRangeDyn "k" (parts hasSeq).start (parts hasSeq).stop
      (parts hasSeq).stride (parts hasSeq).body) (preloop hasSeq).val = some (second hasSeq).val := by
  simp only [step]
  change range (engine model) "k" 0 2 1 (parts hasSeq).body (preloop hasSeq).val = some (second hasSeq).val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, (first hasSeq).property, bind, Option.bind_some]
  change range (engine model) "k" 1 2 1 (parts hasSeq).body (first hasSeq).val = some (second hasSeq).val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, (second hasSeq).property, bind, Option.bind_some]
  change range (engine model) "k" 2 2 1 (parts hasSeq).body (second hasSeq).val = some (second hasSeq).val
  rw [range]
  simp

theorem active_condition (hasSeq : Bool) :
    FP.Structural.evalOp (engine model) none (outer hasSeq).condition entry.val =
      some (fun _ => Bool.true) := by
  cases hasSeq
  all_goals
    conv_lhs => whnf
    simp only [Option.some.injEq]
    apply funext
    intro i
    cases i
    rfl

theorem original_execution (hasSeq : Bool) :
    FP.Structural.exec (engine model) (original hasSeq).kernel initial =
      some (finish hasSeq).val := by
  change run (engine model) (original hasSeq).kernel.surfaceBody initial = _
  rw [source_parts, run_append, entry_executes]
  simp only [Option.bind_some]
  rw [run, step]
  simp only [active_condition, bind, Option.bind_some, ↓reduceIte]
  simp only [List.append_assoc, run_append, (preloop hasSeq).property, Option.bind_some, List.singleton_append]
  rw [run]
  simp only [loop_executes, bind, Option.bind_some, (finish hasSeq).property, run]

theorem domain_inhabited (hasSeq : Bool) :
    ∀ site ∈ (original hasSeq).domain, site.Holds model domain initial := by
  simp [original, FPEquivProofs.original, FPEquivProofs.requirementsFor,
    TypedSource.original, TypedSource.specialize, ComputeKernel.surfaceBody,
    Source.conditional?, Source.dynamicRange?, Source.prefixBefore, Site.Holds, domain]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) (hasSeq : Bool) :
    FP.Structural.exec (engine model) (candidate hasSeq).kernel initial =
      some (finish hasSeq).val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.dot_first_equiv .fp16 .bf16 .fp16 "A" "B" "Out" ⟨"Seq"⟩
      7 4 3 2 100 10 4 1 120 11 5 1 200 24 16 4 1 90 1 4 2 2 Bool.true hasSeq R)
  exact (h ℝ model domain (model_satisfies_rules R) initial
    (domain_inhabited hasSeq)).symm.trans (original_execution hasSeq)

/-- The actual source visits two blocks and computes the shortened length. -/
theorem reached_partial_chunk (hasSeq : Bool) :
    (FP.Structural.exec (engine model) (original hasSeq).kernel initial).map
      (fun t => (t.regs .nat [] "chunk_size_limit", t.regs .nat [] "k")) =
      some (some (fun _ => 3), some (fun _ => 1)) := by
  rw [original_execution]
  cases hasSeq <;> rfl

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) (hasSeq : Bool) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) (original hasSeq).kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) (candidate hasSeq).kernel initial = some t) ∧
    (∀ site ∈ (original hasSeq).domain, site.Holds model domain initial) :=
  ⟨model_satisfies_rules R, ⟨_, original_execution hasSeq⟩,
    ⟨_, candidate_execution R hasSeq⟩, domain_inhabited hasSeq⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedExecution
