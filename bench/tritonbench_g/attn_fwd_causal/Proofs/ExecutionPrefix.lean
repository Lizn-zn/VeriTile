import bench.tritonbench_g.attn_fwd_causal.Proofs.FP
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! Shared execution fixture and the first helper's successful iteration in
one consistent abstract model. Program ID 1, two-lane query/key tiles and both
legal outer stages are retained. TypedExecution completes the second helper,
checks normalization guards and executes both complete source variants.
This witness is not a GPU or real-formula proof. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 3200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def stage (diagonalFirst : Bool) : Nat := if diagonalFirst then 2 else 3
theorem stage_valid (d : Bool) : stage d = 2 ∨ stage d = 3 := by
  cases d <;> simp [stage]

def initial : State ℝ where
  mem := fun r _ => if r = "QS" ∨ r = "KS" then .mk .fp32 1 else .mk .fp16 0
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original (d : Bool) := FPEquivProofs.original "Q" "K" "V" "QS" "KS" "Out"
  4 4 1 1 4 4 1 1 4 4 1 1 4 4 1 1 1 1 4 1 2 2 1 1 (stage d) (stage_valid d)
def reciprocal (d : Bool) := FPEquivProofs.reciprocal "Q" "K" "V" "QS" "KS" "Out"
  4 4 1 1 4 4 1 1 4 4 1 1 4 4 1 1 1 1 4 1 2 2 1 1 (stage d) (stage_valid d)

def code (d : Bool) := (original d).kernel.surfaceBody
def firstBody (d : Bool) : List ComputeStmt :=
  ((Source.dynamicRange? (code d) "start_n" 0).map (·.body)).getD []
def secondBody (d : Bool) : List ComputeStmt :=
  ((Source.dynamicRange? (code d) "start_n" 1).map (·.body)).getD []

def preloop (d : Bool) : {t : State ℝ // run (engine model) ((code d).take 28) initial = some t} := by
  cases d
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [code, original, FPEquivProofs.original, TypedSource.original, stage,
      ComputeKernel.surfaceBody, List.take_succ_cons, List.take_zero]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, State.setReg,
      TileShape.axisDim, Region.cast]
    try rfl

def firstStart (d : Bool) : Nat := if d then 2 else 0
def firstEntry (d : Bool) : State ℝ :=
  (preloop d).val.setReg "start_n" .nat [] (fun _ => firstStart d)
def firstIteration (d : Bool) : {t : State ℝ //
    run (engine model) (firstBody d) (firstEntry d) = some t} := by
  cases d
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [firstBody, code, original, FPEquivProofs.original, TypedSource.original, stage,
      ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map_some, Option.getD_some,
      Bool.true_eq, ↓reduceIte, beq_self_eq_true]
    simp (config := { maxSteps := 150000 }) [firstEntry, firstStart, preloop, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, initial, ComputeDType.eraseDType,
      numeric, integralFloorDiv, integralMod, numericLt, numericLe, numericEq, natLt, bop,
      State.setReg, TileShape.axisDim, Region.cast]
    try rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.TypedExecution
