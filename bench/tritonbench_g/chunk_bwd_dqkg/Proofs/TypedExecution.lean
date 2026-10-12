import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.FP
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! A successful complete execution in an abstract backend satisfying the
admitted scalar rules. The fixture has mixed storage, two value iterations,
and partial time, key and value tiles. It witnesses a nonempty FP domain;
opaque dots and reductions here are not a GPU accuracy experiment. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 3200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private def initial : State ℝ where
  mem := fun r _ =>
    if r = "q" ∨ r = "k" ∨ r = "dq" ∨ r = "dk" ∨ r = "dh" then .mk .fp16 2
    else if r = "v" ∨ r = "do" then .mk .bf16 2
    -- Explicit fp32 computations use the .real carrier under the fp32 profile.
    else .mk .real 2
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

private def original := FPEquivProofs.original .fp16 .bf16 .fp32 .fp16
  "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
  9 3 9 3 18 3 2 3 3 3 2 2 2 2

private def candidate := FPEquivProofs.dotLeft .fp16 .bf16 .fp32 .fp16
  "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
  9 3 9 3 18 3 2 3 3 3 2 2 2 2

private def parts : Source.DynamicRange :=
  (Source.dynamicRange? original.kernel.surfaceBody "i_v").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

private def entry : {t : State ℝ //
    run (engine model) parts.before initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, numericEq, natLt, bop,
    initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg]
  try rfl

private def first : {t : State ℝ //
    run (engine model) parts.body (entry.val.setReg "i_v" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, numericEq, natLt, bop,
    initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry]
  try rfl

private def second : {t : State ℝ //
    run (engine model) parts.body (first.val.setReg "i_v" .nat [] (fun _ => 1)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, numericEq, natLt, bop,
    initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, first]
  try rfl

private def finish : {t : State ℝ //
    run (engine model) parts.after second.val = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts, original, FPEquivProofs.original, TypedSource.original,
    TypedSource.specialize, TypedSource.Storage.dtype, ComputeKernel.surfaceBody,
    Source.dynamicRange?, Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, numericEq, natLt, bop,
    initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, entry, first, second, FP.Structural.store]
  try rfl

private theorem source_parts :
    original.kernel.surfaceBody = parts.before ++
      [.forRangeDyn "i_v" parts.start parts.stop parts.stride parts.body] ++ parts.after := by
  rfl

private theorem loop_executes :
    step (engine model) (.forRangeDyn "i_v" parts.start parts.stop parts.stride parts.body)
      entry.val = some second.val := by
  simp only [step]
  change range (engine model) "i_v" 0 2 1 parts.body entry.val = some second.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (0 : Nat) < 2 from by decide,
    ↓reduceIte, first.property, bind, Option.bind_some]
  change range (engine model) "i_v" 1 2 1 parts.body first.val = some second.val
  rw [range]
  simp only [show (1 : Nat) ≠ 0 from by decide, show (1 : Nat) < 2 from by decide,
    ↓reduceIte, second.property, bind, Option.bind_some]
  change range (engine model) "i_v" 2 2 1 parts.body second.val = some second.val
  rw [range]
  simp

private theorem original_execution :
    FP.Structural.exec (engine model) original.kernel initial = some finish.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  rw [source_parts]
  simp only [List.append_assoc, run_append, entry.property, Option.bind_some,
    List.singleton_append]
  rw [run]
  simp only [loop_executes, bind, Option.bind_some, finish.property]

theorem domain_inhabited :
    ∀ site ∈ original.domain, site.Holds model domain initial := by
  intro site hs
  simp only [original, FPEquivProofs.original, FPEquivProofs.requirementsFor,
    List.mem_singleton] at hs
  subst site
  intro t ht a b ha hb
  exact ⟨fun _ => True.intro, fun _ => True.intro⟩

private theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) candidate.kernel initial = some finish.val := by
  have he := FP.ProfiledRewrite.sound
    (FPEquivProofs.dot_left_equiv .fp16 .bf16 .fp32 .fp16
      "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg"
      9 3 9 3 18 3 2 3 3 3 2 2 2 2 R)
  exact (he ℝ model domain (model_satisfies_rules R) initial domain_inhabited).symm.trans
    original_execution

/-- Actual output stores write the active lane and preserve the masked tail.
The gate-gradient value is specific to this opaque execution backend. -/
theorem output_and_mask :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => ((t.mem "dq" 17).read .fp16, (t.mem "dq" 18).read .fp16,
        (t.mem "dk" 17).read .fp16, (t.mem "dk" 18).read .fp16,
        (t.mem "dg" 11).read .real, (t.mem "dg" 12).read .real)) =
      some (0, 2, 0, 2, 4, 2) := by
  rw [original_execution]
  have hm : TileShape.allIndices [2, 2] =
      [(0, 0, PUnit.unit), (0, 1, PUnit.unit),
        (1, 0, PUnit.unit), (1, 1, PUnit.unit)] := by decide
  have hv : TileShape.allIndices [2] = [(0, PUnit.unit), (1, PUnit.unit)] := by decide
  norm_num [finish, second, first, entry, hm, hv, State.write, Cell.read, initial,
    bop, numeric, TileShape.indexToList]
  dsimp only
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem candidate_output_and_mask (R : FP.ScalarArithmetic.Rules) :
    (FP.Structural.exec (engine model) candidate.kernel initial).map
      (fun t => ((t.mem "dq" 17).read .fp16, (t.mem "dq" 18).read .fp16,
        (t.mem "dk" 17).read .fp16, (t.mem "dk" 18).read .fp16,
        (t.mem "dg" 11).read .real, (t.mem "dg" 12).read .real)) =
      some (0, 2, 0, 2, 4, 2) := by
  rw [candidate_execution R, ← original_execution]
  exact output_and_mask

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) original.kernel initial = some t) ∧
    (∃ t, FP.Structural.exec (engine model) candidate.kernel initial = some t) ∧
    (∀ site ∈ original.domain, site.Holds model domain initial) :=
  ⟨model_satisfies_rules R, ⟨finish.val, original_execution⟩,
    ⟨finish.val, candidate_execution R⟩, domain_inhabited⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.TypedExecution
