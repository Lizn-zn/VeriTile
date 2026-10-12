import bench.tritonbench_g.chunk_gate_recurrence.Proofs.FP
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! Nonempty typed executions of both complete sources. Three chunks and
nonzero head/K/V program IDs expose the forward exclusive seed and backward
one-chunk pointer offset. Source, gate and seed/incoming storage differ.
The shared abstract model returns one for reductions; this is a consistency
and memory-flow witness, not a real DG formula or a GPU measurement. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def initial : State ℝ where
  mem := fun r off =>
    if r = "S" then .mk .fp16 ((off / 16 - 3 + 1 : Nat) : ℝ)
    else if r = "D" then .mk .bf16 2
    else if r = "Seed" then .mk .fp32 3
    else if r = "DS" then .mk .real ((off / 16 - 3 + 1 : Nat) : ℝ)
    else if r = "DI" then .mk .real 9
    else if r = "DG" then .mk .bf16 9
    else .mk .fp16 9
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def forward (hasSeed : Bool) :=
  FPEquivProofs.forward .fp16 .bf16 .fp32 "S" "D" "O" "Seed" 2 3 4 4 2 2 hasSeed
def backward :=
  FPEquivProofs.backward .fp16 .bf16 .fp32 "S" "D" "DI" "DG" "DL" "DS" 2 3 4 4 2 2

def forwardParts (hasSeed : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (forward hasSeed).kernel.surfaceBody "_i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩
def backwardParts : Source.DynamicRange :=
  (Source.dynamicRange? backward.kernel.surfaceBody "_i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def forwardEntry (hasSeed : Bool) :
    {t : State ℝ // run (engine model) (forwardParts hasSeed).before initial = some t} := by
  cases hasSeed
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [forwardParts, forward, FPEquivProofs.forward, TypedSource.forward, TypedSource.specializeForward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod]
    try rfl

def forwardFirst (hasSeed : Bool) :
    {t : State ℝ // run (engine model) (forwardParts hasSeed).body ((forwardEntry hasSeed).val.setReg "_i" .nat [] (fun _ => 0)) = some t} := by
  cases hasSeed
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [forwardParts, forward, FPEquivProofs.forward, TypedSource.forward, TypedSource.specializeForward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, forwardEntry]
    try rfl

def forwardSecond (hasSeed : Bool) :
    {t : State ℝ // run (engine model) (forwardParts hasSeed).body ((forwardFirst hasSeed).val.setReg "_i" .nat [] (fun _ => 1)) = some t} := by
  cases hasSeed
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [forwardParts, forward, FPEquivProofs.forward, TypedSource.forward, TypedSource.specializeForward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, forwardEntry, forwardFirst]
    try rfl

def forwardFinish (hasSeed : Bool) :
    {t : State ℝ // run (engine model) (forwardParts hasSeed).after (forwardSecond hasSeed).val = some t} := by
  cases hasSeed
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [forwardParts, forward, FPEquivProofs.forward, TypedSource.forward, TypedSource.specializeForward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, forwardEntry, forwardFirst, forwardSecond]
    try rfl

theorem forward_parts (hasSeed : Bool) :
    (forward hasSeed).kernel.surfaceBody = (forwardParts hasSeed).before ++
      [.forRangeDyn "_i" (forwardParts hasSeed).start (forwardParts hasSeed).stop (forwardParts hasSeed).stride (forwardParts hasSeed).body] ++ (forwardParts hasSeed).after := by
  cases hasSeed <;> rfl

theorem forward_loop (hasSeed : Bool) :
    step (engine model) (.forRangeDyn "_i" (forwardParts hasSeed).start (forwardParts hasSeed).stop (forwardParts hasSeed).stride (forwardParts hasSeed).body)
      (forwardEntry hasSeed).val = some (forwardSecond hasSeed).val := by
  simp only [step]
  change range (engine model) "_i" 0 2 1 (forwardParts hasSeed).body (forwardEntry hasSeed).val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, (forwardFirst hasSeed).property, bind, Option.bind_some]
  change range (engine model) "_i" 1 2 1 (forwardParts hasSeed).body (forwardFirst hasSeed).val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, (forwardSecond hasSeed).property, bind, Option.bind_some]
  change range (engine model) "_i" 2 2 1 (forwardParts hasSeed).body (forwardSecond hasSeed).val = _
  rw [range]
  simp

theorem forward_execution (hasSeed : Bool) :
    FP.Structural.exec (engine model) (forward hasSeed).kernel initial = some (forwardFinish hasSeed).val := by
  change run (engine model) (forward hasSeed).kernel.surfaceBody initial = _
  simp only [forward_parts, List.append_assoc, run_append, (forwardEntry hasSeed).property,
    Option.bind_some, List.singleton_append]
  rw [run]
  simp only [forward_loop, bind, Option.bind_some, (forwardFinish hasSeed).property]

theorem forward_domain (hasSeed : Bool) :
    ∀ site ∈ (forward hasSeed).domain, site.Holds model domain initial := by
  simp [forward, FPEquivProofs.forward, FPEquivProofs.forwardRequirements,
    TypedSource.forward, TypedSource.specializeForward, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Source.prefixBefore, Site.Holds, domain]

def backwardEntry :
    {t : State ℝ // run (engine model) (backwardParts).before initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [backwardParts, backward, FPEquivProofs.backward, TypedSource.backward, TypedSource.specializeBackward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod]
  try rfl

def backwardFirst :
    {t : State ℝ // run (engine model) (backwardParts).body ((backwardEntry).val.setReg "_i" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [backwardParts, backward, FPEquivProofs.backward, TypedSource.backward, TypedSource.specializeBackward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, backwardEntry]
  try rfl

def backwardSecond :
    {t : State ℝ // run (engine model) (backwardParts).body ((backwardFirst).val.setReg "_i" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [backwardParts, backward, FPEquivProofs.backward, TypedSource.backward, TypedSource.specializeBackward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, backwardEntry, backwardFirst]
  try rfl

def backwardFinish :
    {t : State ℝ // run (engine model) (backwardParts).after (backwardSecond).val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [backwardParts, backward, FPEquivProofs.backward, TypedSource.backward, TypedSource.specializeBackward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, backwardEntry, backwardFirst, backwardSecond]
  try rfl

theorem backward_parts :
    (backward).kernel.surfaceBody = (backwardParts).before ++
      [.forRangeDyn "_i" (backwardParts).start (backwardParts).stop (backwardParts).stride (backwardParts).body] ++ (backwardParts).after := by
  rfl

theorem backward_loop :
    step (engine model) (.forRangeDyn "_i" (backwardParts).start (backwardParts).stop (backwardParts).stride (backwardParts).body)
      (backwardEntry).val = some (backwardSecond).val := by
  simp only [step]
  change range (engine model) "_i" 0 2 1 (backwardParts).body (backwardEntry).val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, (backwardFirst).property, bind, Option.bind_some]
  change range (engine model) "_i" 1 2 1 (backwardParts).body (backwardFirst).val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, (backwardSecond).property, bind, Option.bind_some]
  change range (engine model) "_i" 2 2 1 (backwardParts).body (backwardSecond).val = _
  rw [range]
  simp

theorem backward_execution :
    FP.Structural.exec (engine model) (backward).kernel initial = some (backwardFinish).val := by
  change run (engine model) (backward).kernel.surfaceBody initial = _
  simp only [backward_parts, List.append_assoc, run_append, (backwardEntry).property,
    Option.bind_some, List.singleton_append]
  rw [run]
  simp only [backward_loop, bind, Option.bind_some, (backwardFinish).property]

theorem backward_domain :
    ∀ site ∈ (backward).domain, site.Holds model domain initial := by
  simp [backward, FPEquivProofs.backward, FPEquivProofs.backwardRequirements,
    TypedSource.backward, TypedSource.specializeBackward, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Source.prefixBefore, Site.Holds, domain]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedExecution
