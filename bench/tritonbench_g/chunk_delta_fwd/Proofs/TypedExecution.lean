import bench.tritonbench_g.chunk_delta_fwd.Proofs.FP
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! A consistent abstract execution with two outer chunks and two inner chunks
per update. Mixed fp16/bf16/fp32 storage, both state switches enabled, nonzero head/value
program IDs and time/key/value tails are exercised. Typed dots return one in
this model; the witness is neither GPU behavior nor real matrix correctness. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private def initialState : State ℝ where
  mem := fun r _ =>
    if r = "k" ∨ r = "d" then .mk .fp16 2
    else if r = "v" then .mk .bf16 2
    else if r = "h0" then .mk .bf16 3
    else if r = "h" then .mk .fp16 9
    else if r = "v_new" then .mk .bf16 9
    else .mk .real 9
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 0 else 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

private def original :=
  FPEquivProofs.original .fp16 .bf16 .bf16 .fp32 "k" "v" "d" "v_new" "h" "h0" "ht"
    15 3 1 15 3 1 18 3 1 5 3 3 4 2 4 2 2 Bool.true Bool.true
private def candidate :=
  FPEquivProofs.sumLeft .fp16 .bf16 .bf16 .fp32 "k" "v" "d" "v_new" "h" "h0" "ht"
    15 3 1 15 3 1 18 3 1 5 3 3 4 2 4 2 2 Bool.true Bool.true
private def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "i_t").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

private def innerParts : Source.DynamicRange :=
  (Source.dynamicRange? parts.body "i_c").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

private def entry :
  {t : State ℝ // run (engine model) (parts.before) (initialState) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg]
  try rfl

private def pre0 :
  {t : State ℝ // run (engine model) (innerParts.before) (entry.val.setReg "i_t" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry]
  try rfl

private def inner00 :
  {t : State ℝ // run (engine model) (innerParts.body) (pre0.val.setReg "i_c" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, pre0]
  try rfl

private def inner01 :
  {t : State ℝ // run (engine model) (innerParts.body) (inner00.val.setReg "i_c" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, pre0, inner00]
  try rfl

private def post0 :
  {t : State ℝ // run (engine model) (innerParts.after) (inner01.val) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, pre0, inner00, inner01]
  try rfl

private def pre1 :
  {t : State ℝ // run (engine model) (innerParts.before) (post0.val.setReg "i_t" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, pre0, inner00, inner01, post0]
  try rfl

private def inner10 :
  {t : State ℝ // run (engine model) (innerParts.body) (pre1.val.setReg "i_c" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, pre0, inner00, inner01, post0, pre1]
  try rfl

private def inner11 :
  {t : State ℝ // run (engine model) (innerParts.body) (inner10.val.setReg "i_c" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, pre0, inner00, inner01, post0, pre1, inner10]
  try rfl

private def post1 :
  {t : State ℝ // run (engine model) (innerParts.after) (inner11.val) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, pre0, inner00, inner01, post0, pre1, inner10, inner11]
  try rfl

private def finish :
  {t : State ℝ // run (engine model) (parts.after) (post1.val) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [innerParts, parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, initialState,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, pre0, inner00, inner01, post0, pre1, inner10, inner11, post1]
  try rfl


private theorem body_parts :
    parts.body = innerParts.before ++
      [.forRangeDyn "i_c" (.constNat 0) innerParts.stop
        (.constNat 1) innerParts.body] ++
      innerParts.after := rfl

private theorem outer0_execution :
    run (engine model) parts.body
      (entry.val.setReg "i_t" .nat [] (fun _ => 0)) =
      some post0.val := by
  simp only [body_parts, run_append, pre0.property]
  simp only [Option.bind_some, run, step]
  change ((range (engine model) "i_c" 0 2 1 innerParts.body
    pre0.val).bind (fun t => some t)).bind
      (run (engine model) innerParts.after) = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, inner00.property, bind, Option.bind_some]
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, inner01.property, bind, Option.bind_some]
  rw [range]
  simp only [show ¬ (2 : Nat) < 2 from by decide, ↓reduceIte, Option.bind_some]
  exact post0.property

private theorem outer1_execution :
    run (engine model) parts.body
      (post0.val.setReg "i_t" .nat [] (fun _ => 1)) =
      some post1.val := by
  simp only [body_parts, run_append, pre1.property]
  simp only [Option.bind_some, run, step]
  change ((range (engine model) "i_c" 0 2 1 innerParts.body
    pre1.val).bind (fun t => some t)).bind
      (run (engine model) innerParts.after) = _
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, inner10.property, bind, Option.bind_some]
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, inner11.property, bind, Option.bind_some]
  rw [range]
  simp only [show ¬ (2 : Nat) < 2 from by decide, ↓reduceIte, Option.bind_some]
  exact post1.property

private theorem source_parts :
    original.kernel.surfaceBody = parts.before ++
      [.forRange "i_t" 0 2 1 parts.body] ++
      parts.after := rfl

theorem original_execution :
    FP.Structural.exec (engine model) original.kernel initialState =
      some finish.val := by
  change run (engine model) original.kernel.surfaceBody initialState = _
  simp only [source_parts, run_append, entry.property]
  simp only [Option.bind_some, run, step]
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, outer0_execution, bind, Option.bind_some]
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, outer1_execution, bind, Option.bind_some]
  rw [range]
  simp only [show ¬ (2 : Nat) < 2 from by decide, ↓reduceIte, Option.bind_some]
  exact finish.property

theorem domain_inhabited :
    ∀ site ∈ original.domain, site.Holds model domain initialState := by
  simp [original, FPEquivProofs.original, FPEquivProofs.requirementsFor,
    TypedSource.original, TypedSource.specialize, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Source.prefixBefore, Site.Holds, domain]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) candidate.kernel initialState =
      some finish.val := by
  have h := FP.ProfiledRewrite.sound
    (FPEquivProofs.sum_left_equiv .fp16 .bf16 .bf16 .fp32 "k" "v" "d" "v_new" "h" "h0" "ht"
      15 3 1 15 3 1 18 3 1 5 3 3 4 2 4 2 2 Bool.true Bool.true R)
  exact (h ℝ model domain (model_satisfies_rules R) initialState
    (domain_inhabited )).symm.trans (original_execution )

/-- History is stored before each update; the final state is stored after both.
The corrected-value time tail and neighboring key/value cells stay masked. -/
theorem outputs_and_masks :
    (FP.Structural.exec (engine model) original.kernel initialState).map
      (fun t => [(t.mem "h" 19).read .fp16, (t.mem "h" 20).read .fp16,
        (t.mem "h" 26).read .fp16, (t.mem "h" 29).read .fp16,
        (t.mem "h" 35).read .fp16, (t.mem "h" 36).read .fp16,
        (t.mem "v_new" 16).read .bf16, (t.mem "v_new" 17).read .bf16,
        (t.mem "v_new" 26).read .bf16, (t.mem "v_new" 29).read .bf16,
        (t.mem "v_new" 30).read .bf16, (t.mem "ht" 10).read .real,
        (t.mem "ht" 11).read .real, (t.mem "ht" 17).read .real,
        (t.mem "ht" 18).read .real]) =
      some [9, 3, 3, 5, 5, 9, 9, 1, 1, 1, 9, 9, 7, 7, 9] := by
  rw [original_execution]
  have hm : TileShape.allIndices [4, 2] =
      [(0, 0, PUnit.unit), (0, 1, PUnit.unit), (1, 0, PUnit.unit), (1, 1, PUnit.unit),
       (2, 0, PUnit.unit), (2, 1, PUnit.unit), (3, 0, PUnit.unit), (3, 1, PUnit.unit)] := by decide
  have hi : TileShape.allIndices [2, 2] =
      [(0, 0, PUnit.unit), (0, 1, PUnit.unit), (1, 0, PUnit.unit), (1, 1, PUnit.unit)] := by decide
  norm_num [finish, post1, inner11, inner10, pre1, post0, inner01, inner00, pre0,
    entry, hm, hi, State.write, Cell.read, initialState, bop, numeric, TileShape.indexToList,
    String.reduceEq, Region.cast, Region.mk.injEq, boundaryZero]
  dsimp only
  repeat' apply And.intro
  all_goals first | rfl | (change (2 : ℝ) - 1 = 1; norm_num)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.TypedExecution
