import bench.tritonbench_g.dequantize_matmul.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

def initial : State ℝ where
  mem := fun r _ => if r="b" then .mk .int (-3) else .mk .fp32 2
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original := kernel .fp16 Bool.false "b" "scale" "out" 3 3 3 1 3 1 2 2

/-- Full typed execution with signed integer input and a partially active tile.
The consistent abstract backend witnesses inhabited semantics, not GPU rounding. -/
def execution : {t : State ℝ // FP.Structural.exec (engine model) original initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  dsimp only [original,kernel,ComputeKernel.surfaceBody,FP.Structural.exec]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,bop,initial,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write,
    TileShape.dropInsertedIndex]
  rfl

theorem candidate_executes :
    FP.Structural.exec (engine model) (kernel .fp16 Bool.true "b" "scale" "out" 3 3 3 1 3 1 2 2)
      initial = some execution.val := by
  have h := FP.ProfiledRewrite.sound (FPProof.equivalent .fp16 "b" "scale" "out" 3 3 3 1 3 1 2 2)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    initial (by simp [FPProof.program])
  exact h.symm.trans execution.property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.TypedExecution
