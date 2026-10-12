import bench.tritonbench_g.embedding_triton_kernel.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Float.Control
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 25000

def stored (storage : Storage) (value : ℝ) : Cell ℝ :=
  match storage with
  | .fp32 => .mk .real value
  | .fp16 => .mk .fp16 value
  | .bf16 => .mk .bf16 value

def initial (storage : Storage) : State ℝ where
  mem := fun r o => if r="ids" then .mk .int (if o=0 then -1 else if o=1 then 1 else 3)
    else stored storage (-3)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Three iterations from empty registers, a negative invalid ID, one valid ID, masked final row and masked hidden tail.
This witnesses abstract typed execution, not GPU rounding. -/
def execution (storage : Storage) :
    {v : State ℝ // FP.Structural.exec (engine model)
      (source storage Bool.false "weight" "ids" "out" 0 3 2 2 2 1 2 3 1) (initial storage)=some v} := by
  cases storage
  all_goals
    refine ⟨?_,?_⟩
    swap
    dsimp only [source,ComputeKernel.surfaceBody,FP.Structural.exec]
    simp [run,step,range_done,range_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,bop,initial,stored,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write,TileShape.dropInsertedIndex]
    try rfl

theorem candidate_executes (storage : Storage) :
    FP.Structural.exec (engine model)
      (source storage Bool.true "weight" "ids" "out" 0 3 2 2 2 1 2 3 1) (initial storage)=
      some (execution storage).val := by
  have h := FP.ProfiledRewrite.sound
    (FPProof.equivalent storage "weight" "ids" "out" 0 3 2 2 2 1 2 3 1)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial storage) (by simp [FPProof.program])
  exact h.symm.trans (execution storage).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.TypedExecution
