import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.FP
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! Successful complete-source executions in a consistent abstract FP model.
The fixture uses bf16 input/weights, fp16 output, nonzero batch/slot, K=7/BK=2,
N=3/BN=4 and both split choices. Store has four iterations; atomic addition
has two, with a nonzero split ID and a masked K tail. The final output mask
preserves the fourth lane. This is not a GPU or real GEMV correctness test:
the model's opaque reduction returns one, while its atomic backend adds.
-/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 3200000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

private def initial (atomic : Bool) (slot : Int) : State ℝ where
  mem := fun r _ => if r = "Index" then .mk .int slot
    else if r = "X" ∨ r = "W" then .mk .bf16 2
    else .mk .fp16 2
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then if atomic then 1 else 0 else 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

private def original (atomic : Bool) := FPEquivProofs.original .bf16 .fp16
  "X" "W" "Y" 3 7 ⟨"Index"⟩ 3 9 17 40 9 1 13 2 4 2 (if atomic then 2 else 1)
private def candidate (atomic : Bool) := FPEquivProofs.scalarLeft .bf16 .fp16
  "X" "W" "Y" 3 7 ⟨"Index"⟩ 3 9 17 40 9 1 13 2 4 2 (if atomic then 2 else 1)
private def program (atomic optimized : Bool) := if optimized then candidate atomic else original atomic

private def execution (atomic optimized : Bool) : {t : State ℝ //
    FP.Structural.exec (engine model) (program atomic optimized).kernel (initial atomic 1) = some t} := by
  cases atomic <;> cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (config := { maxSteps := 120000 }) [program, original, candidate,
      FPEquivProofs.original, FPEquivProofs.scalarLeft, TypedSource.original,
      TypedSource.scalarLeft, TypedSource.specialize, ComputeKernel.surfaceBody,
      FP.Structural.exec, run, step, FP.Structural.range,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, numericLt, numericLe, numericEq, natLt, bop, initial,
      FP.Structural.store, FP.Structural.atomicAdd, atomicAddition,
      State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl

theorem source_executes (atomic optimized : Bool) :
    ∃ t, FP.Structural.exec (engine model) (program atomic optimized).kernel (initial atomic 1) = some t :=
  ⟨(execution atomic optimized).val, (execution atomic optimized).property⟩

/-- Readback of actual final memory distinguishes successful atomic updates
from mere equality of failed executions. The fourth output lane is masked. -/
theorem output_and_mask (atomic optimized : Bool) :
    (FP.Structural.exec (engine model) (program atomic optimized).kernel (initial atomic 1)).map
      (fun t => ((t.mem "Y" 13).read .fp16, (t.mem "Y" 15).read .fp16,
        (t.mem "Y" 17).read .fp16, (t.mem "Y" 19).read .fp16)) =
      some (if atomic then (8,8,8,2) else (12,12,12,2)) := by
  rw [(execution atomic optimized).property]
  have hi : TileShape.allIndices [4] =
      [(0,PUnit.unit), (1,PUnit.unit), (2,PUnit.unit), (3,PUnit.unit)] := by decide
  cases atomic <;> cases optimized
  all_goals
    norm_num [execution, hi, State.write, Cell.read, initial]
    dsimp only
    first
    | exact ⟨rfl, rfl, rfl, rfl⟩
    | change (2+6 : ℝ) = 8 ∧ (2+6 : ℝ) = 8 ∧ (2+6 : ℝ) = 8 ∧ (2 : ℝ) = 2
      norm_num

private def skipped (atomic optimized : Bool) : {t : State ℝ //
    FP.Structural.exec (engine model) (program atomic optimized).kernel (initial atomic (-1)) = some t} := by
  cases atomic <;> cases optimized
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [program, original, candidate, FPEquivProofs.original, FPEquivProofs.scalarLeft,
      TypedSource.original, TypedSource.scalarLeft, TypedSource.specialize,
      ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      numericEq, bop, initial, State.setReg, Cell.read, Region.cast]
    try rfl

theorem sentinel_skips (atomic optimized : Bool) :
    ∃ t, FP.Structural.exec (engine model) (program atomic optimized).kernel (initial atomic (-1)) = some t ∧
      t.mem = (initial atomic (-1)).mem := by
  refine ⟨(skipped atomic optimized).val, (skipped atomic optimized).property, ?_⟩
  cases atomic <;> cases optimized <;> rfl

theorem domain_inhabited (atomic : Bool) :
    ∀ site ∈ (original atomic).domain, site.Holds model domain (initial atomic 1) := by
  cases atomic
  all_goals
    simp [original, FPEquivProofs.original, FPEquivProofs.requirementsFor,
      TypedSource.original, TypedSource.specialize, ComputeKernel.surfaceBody,
      Source.conditional?, Source.prefixBefore, Site.Holds, domain]

theorem consistent_execution (R : FP.ScalarArithmetic.Rules) (atomic : Bool) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, FP.Structural.exec (engine model) (original atomic).kernel (initial atomic 1) = some t) ∧
    (∃ t, FP.Structural.exec (engine model) (candidate atomic).kernel (initial atomic 1) = some t) ∧
    (∀ site ∈ (original atomic).domain, site.Holds model domain (initial atomic 1)) := by
  exact ⟨model_satisfies_rules R, source_executes atomic Bool.false,
    source_executes atomic Bool.true, domain_inhabited atomic⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.TypedExecution
