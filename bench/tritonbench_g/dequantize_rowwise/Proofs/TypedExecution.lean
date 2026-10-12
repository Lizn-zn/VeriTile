import bench.tritonbench_g.dequantize_rowwise.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def initial : State ℝ where
  mem := fun r _ => if r="x" then .mk .int (-3) else .mk .fp32 2
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def source := original "x" "scale" "out" (1/127) 6 3 4

/-- Negative integer input and a masked tail with omitted other. This is a
consistent abstract FP execution from empty registers, not GPU measurement. -/
def execution : {t : State ℝ // FP.Structural.exec (engine model) source initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  dsimp only [source,original,ComputeKernel.surfaceBody,FP.Structural.exec]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,bop,initial,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
  rfl

theorem candidate_executes :
    FP.Structural.exec (engine model) (indexCommuted "x" "scale" "out" (1/127) 6 3 4)
      initial = some execution.val := by
  have h := FP.ProfiledRewrite.sound (FPProof.equivalent "x" "scale" "out" (1/127) 6 3 4)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    initial (by simp [FPProof.original])
  exact h.symm.trans execution.property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.TypedExecution
