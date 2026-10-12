import bench.tritonbench_g.chunk_gated_attention.Proofs.FP
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! Successful typed executions with nonzero program IDs, two chunks and
partial time/K/V tails. Dot inputs, gate/history, seed and final output retain
their own storage. All eight gate/seed/final flag combinations execute.
The abstract model interprets exp and dot as one. It witnesses consistency
and typed state flow, not the real recurrence or hardware numerical results. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def initial : State ℝ where
  mem := fun r _ =>
    if r = "k" ∨ r = "v" then .mk .fp16 1
    else if r = "g" then .mk .bf16 0
    else if r = "h" then .mk .bf16 99
    else if r = "h0" then .mk .bf16 3
    else .mk .fp32 0
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def recurrent (gateK hasSeed storeFinal : Bool) :=
  FPEquivProofs.recurrent .fp16 .bf16 .bf16 .bf16 .fp32 "k" "v" "g" "h" "h0" "ht"
    15 3 1 15 3 1 18 3 1 5 3 3 4 2 2 2 gateK hasSeed storeFinal

def parts (gateK hasSeed storeFinal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (recurrent gateK hasSeed storeFinal).kernel.surfaceBody "i_t").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def entry (gateK hasSeed storeFinal : Bool) :
    {t : State ℝ // run (engine model) (parts gateK hasSeed storeFinal).before initial = some t} := by
  cases gateK <;> cases hasSeed <;> cases storeFinal
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [parts, recurrent, FPEquivProofs.recurrent, TypedSource.hOriginal, TypedSource.hSource, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    dsimp [Source.dynamicRange?, Option.map, Option.getD]
    simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod]
    try rfl

def first (gateK hasSeed storeFinal : Bool) :
    {t : State ℝ // run (engine model) (parts gateK hasSeed storeFinal).body ((entry gateK hasSeed storeFinal).val.setReg "i_t" .nat [] (fun _ => 0)) = some t} := by
  cases gateK <;> cases hasSeed <;> cases storeFinal
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [parts, recurrent, FPEquivProofs.recurrent, TypedSource.hOriginal, TypedSource.hSource, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    dsimp [Source.dynamicRange?, Option.map, Option.getD]
    simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, entry]
    try rfl

def second (gateK hasSeed storeFinal : Bool) :
    {t : State ℝ // run (engine model) (parts gateK hasSeed storeFinal).body ((first gateK hasSeed storeFinal).val.setReg "i_t" .nat [] (fun _ => 1)) = some t} := by
  cases gateK <;> cases hasSeed <;> cases storeFinal
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [parts, recurrent, FPEquivProofs.recurrent, TypedSource.hOriginal, TypedSource.hSource, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    dsimp [Source.dynamicRange?, Option.map, Option.getD]
    simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, entry, first]
    try rfl

def finish (gateK hasSeed storeFinal : Bool) :
    {t : State ℝ // run (engine model) (parts gateK hasSeed storeFinal).after (second gateK hasSeed storeFinal).val = some t} := by
  cases gateK <;> cases hasSeed <;> cases storeFinal
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [parts, recurrent, FPEquivProofs.recurrent, TypedSource.hOriginal, TypedSource.hSource, TypedSource.Storage.dtype, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
    dsimp [Source.dynamicRange?, Option.map, Option.getD]
    simp (config := { maxSteps := 150000 }) [
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, bop, initial, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    State.setReg, FP.Structural.store, integralFloorDiv, integralMod, entry, first, second]
    try rfl

theorem source_parts (gateK hasSeed storeFinal : Bool) :
    (recurrent gateK hasSeed storeFinal).kernel.surfaceBody = (parts gateK hasSeed storeFinal).before ++
      [.forRange "i_t" 0 2 1 (parts gateK hasSeed storeFinal).body] ++ (parts gateK hasSeed storeFinal).after := by
  cases gateK <;> cases hasSeed <;> cases storeFinal <;> rfl

theorem loop (gateK hasSeed storeFinal : Bool) :
    step (engine model) (.forRange "i_t" 0 2 1 (parts gateK hasSeed storeFinal).body)
      (entry gateK hasSeed storeFinal).val = some (second gateK hasSeed storeFinal).val := by
  simp only [step]
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, (first gateK hasSeed storeFinal).property, bind, Option.bind_some]
  change range (engine model) "i_t" 1 2 1 (parts gateK hasSeed storeFinal).body (first gateK hasSeed storeFinal).val = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, (second gateK hasSeed storeFinal).property, bind, Option.bind_some]
  change range (engine model) "i_t" 2 2 1 (parts gateK hasSeed storeFinal).body (second gateK hasSeed storeFinal).val = _
  rw [range]
  simp

theorem execution (gateK hasSeed storeFinal : Bool) :
    FP.Structural.exec (engine model) (recurrent gateK hasSeed storeFinal).kernel initial = some (finish gateK hasSeed storeFinal).val := by
  change run (engine model) (recurrent gateK hasSeed storeFinal).kernel.surfaceBody initial = _
  simp only [source_parts, List.append_assoc, run_append, (entry gateK hasSeed storeFinal).property,
    Option.bind_some, List.singleton_append]
  rw [run]
  simp only [loop, bind, Option.bind_some, (finish gateK hasSeed storeFinal).property]

theorem domain_holds (gateK hasSeed storeFinal : Bool) :
    ∀ site ∈ (recurrent gateK hasSeed storeFinal).domain, site.Holds model domain initial := by
  simp [recurrent, FPEquivProofs.recurrent, FPEquivProofs.requirements,
    TypedSource.hOriginal, TypedSource.hSource, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Source.prefixBefore, Site.Holds, domain]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedExecution
