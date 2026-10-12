import bench.tritonbench_g.dropout_triton.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Float.Control
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def stored (storage : Storage) (value : ℝ) : Cell ℝ :=
  match storage with
  | .fp32 => .mk .real value
  | .fp16 => .mk .fp16 value
  | .bf16 => .mk .bf16 value

def initial (storage : Storage) (keep : KeepStorage) : State ℝ where
  mem := fun r o => if r="keep" then
    match keep with
    | .int32 => .mk .int (if o=0 then 0 else if o=1 then 2 else -1)
    | .bool => .mk .bool (decide (o≠0))
    else stored storage (-3)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Empty-register typed execution, three active lanes in a four-lane tile.
Integer keep values include zero, an even positive value and a negative value.
The real-carrier model witnesses the abstract semantics, not GPU rounding. -/
def execution (storage : Storage) (keep : KeepStorage) (p : ℝ) :
    {v : State ℝ // FP.Structural.exec (engine model)
      (source storage keep Bool.false "x" "keep" "out" 3 4 p) (initial storage keep)=some v} := by
  cases storage <;> cases keep
  all_goals
    refine ⟨?_,?_⟩
    swap
    dsimp only [source,ComputeKernel.surfaceBody,FP.Structural.exec]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericEq,bop,initial,stored,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
    try rfl

theorem candidate_executes (storage : Storage) (keep : KeepStorage) (p : ℝ) :
    FP.Structural.exec (engine model) (source storage keep Bool.true "x" "keep" "out" 3 4 p)
      (initial storage keep)=some (execution storage keep p).val := by
  have h := FP.ProfiledRewrite.sound (FPProof.equivalent storage keep "x" "keep" "out" 3 4 p)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial storage keep) (by simp [FPProof.program])
  exact h.symm.trans (execution storage keep p).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DropoutTriton.TypedExecution
