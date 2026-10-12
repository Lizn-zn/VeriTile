import bench.tritonbench_g.chunk_gate_recurrence.Proofs.TypedExecutionSteps

/-! Complete original/candidate executions and typed output checks. The shared
abstract reduction returns one; DG is checked here for storage/addressing,
not for its mathematical reduction value or GPU behavior. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem forward_values (hasSeed : Bool) :
    (FP.Structural.exec (engine model) (forward hasSeed).kernel initial).map
      (fun t => (t.mem "O" 58, t.mem "O" 74, t.mem "O" 90, t.mem "O" 57)) =
      some (.mk .fp16 (if hasSeed then 3 else 0),
        .mk .fp16 (if hasSeed then 7 else 1), .mk .fp16 (if hasSeed then 16 else 4), .mk .fp16 9) := by
  rw [forward_execution]
  cases hasSeed
  all_goals
    simp only [Option.map_some, Bool.false_eq_true, ↓reduceIte, Option.some.injEq, Prod.mk.injEq]
    repeat' apply And.intro
    all_goals
      (conv_lhs => whnf) <;>
      norm_num [bop, Function.comp_def, numeric, ofFloat, toFloat,
        Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        State.write, State.setReg, TileShape.allIndices, List.finRange, List.ofFn_succ,
        String.reduceEq] <;> simp [String.reduceEq, Cell.read] <;> norm_num

/-- DI is shifted by one chunk; DL includes the separate final update.
DG uses its independent split-tile address. The last DI/DG slices and another
program's DL cell keep their original values. -/
theorem backward_values :
    (FP.Structural.exec (engine model) backward.kernel initial).map
      (fun t => (t.mem "DI" 58, t.mem "DI" 74, t.mem "DI" 90,
        t.mem "DL" 26, t.mem "DL" 25, t.mem "DG" 15, t.mem "DG" 19, t.mem "DG" 23)) =
      some (.mk .real 8, .mk .real 3, .mk .real 9,
        .mk .fp16 17, .mk .fp16 9, .mk .bf16 1, .mk .bf16 1, .mk .bf16 9) := by
  rw [backward_execution]
  simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop, Function.comp_def, numeric, ofFloat, toFloat,
      Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        State.write, State.setReg, TileShape.allIndices, List.finRange, List.ofFn_succ,
        String.reduceEq] <;> simp [String.reduceEq, Cell.read] <;> norm_num

theorem forward_candidate_execution (R : FP.ScalarArithmetic.Rules) (hasSeed : Bool) :
    FP.Structural.exec (engine model)
      (FPEquivProofs.forwardSumLeft .fp16 .bf16 .fp32 "S" "D" "O" "Seed" 2 3 4 4 2 2 hasSeed).kernel
      initial = some (forwardFinish hasSeed).val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.forward_equiv .fp16 .bf16 .fp32 "S" "D" "O" "Seed" 2 3 4 4 2 2 hasSeed R)
  exact (h ℝ model domain (model_satisfies_rules R) initial (forward_domain hasSeed)).symm.trans
    (forward_execution hasSeed)

theorem backward_candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model)
      (FPEquivProofs.backwardSumLeft .fp16 .bf16 .fp32 "S" "D" "DI" "DG" "DL" "DS" 2 3 4 4 2 2).kernel
      initial = some backwardFinish.val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.backward_equiv .fp16 .bf16 .fp32 "S" "D" "DI" "DG" "DL" "DS" 2 3 4 4 2 2 R)
  exact (h ℝ model domain (model_satisfies_rules R) initial backward_domain).symm.trans backward_execution

private def singleBackward :=
  FPEquivProofs.backward .fp16 .bf16 .fp32 "S" "D" "DI" "DG" "DL" "DS" 2 1 4 4 2 2

private def singleParts : Source.DynamicRange :=
  (Source.dynamicRange? singleBackward.kernel.surfaceBody "_i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

private def singleEntry : {t : State ℝ //
    run (engine model) singleParts.before initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [singleParts, singleBackward, FPEquivProofs.backward, TypedSource.backward,
    TypedSource.specializeBackward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 150000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initial,
    ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
    FP.Structural.store, integralFloorDiv, integralMod]
  try rfl

private def singleFinish : {t : State ℝ //
    run (engine model) singleParts.after singleEntry.val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [singleParts, singleBackward, FPEquivProofs.backward, TypedSource.backward,
    TypedSource.specializeBackward, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 150000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initial,
    ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
    FP.Structural.store, integralFloorDiv, integralMod, singleEntry]
  try rfl

private theorem single_parts :
    singleBackward.kernel.surfaceBody = singleParts.before ++
      [.forRangeDyn "_i" singleParts.start singleParts.stop singleParts.stride singleParts.body] ++
      singleParts.after := rfl

private theorem single_loop :
    step (engine model) (.forRangeDyn "_i" singleParts.start singleParts.stop singleParts.stride singleParts.body)
      singleEntry.val = some singleEntry.val := by
  simp only [step]
  change range (engine model) "_i" 0 0 1 singleParts.body singleEntry.val = _
  rw [range]
  simp

private def singleResult : {t : State ℝ //
    FP.Structural.exec (engine model) singleBackward.kernel initial = some t} := by
  refine ⟨singleFinish.val, ?_⟩
  change run (engine model) singleBackward.kernel.surfaceBody initial = _
  simp only [single_parts, List.append_assoc, run_append, singleEntry.property,
    Option.bind_some, List.singleton_append]
  rw [run]
  simp only [single_loop, bind, Option.bind_some, singleFinish.property]

/-- N=1 has no reverse iteration. Only DL is written, using DS[0]; negative
relative S/DI/DG positions are never accessed. -/
theorem single_chunk_values :
    (FP.Structural.exec (engine model) singleBackward.kernel initial).map
      (fun t => (t.mem "DL" 26, t.mem "DI" 26, t.mem "DG" 7)) =
      some (.mk .fp16 1, .mk .real 9, .mk .bf16 9) := by
  rw [singleResult.property]
  simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop, Function.comp_def, numeric, ofFloat, toFloat,
      Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
      State.write, State.setReg, TileShape.allIndices, List.finRange, List.ofFn_succ,
      String.reduceEq]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedExecution
