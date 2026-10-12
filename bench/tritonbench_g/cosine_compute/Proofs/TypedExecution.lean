import bench.tritonbench_g.cosine_compute.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.CosineCompute.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def initial : State ℝ where
  mem := fun r _ => if r = "a" then .mk .fp16 (-2) else .mk .fp16 99
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original := CosineCompute.original .fp16 "a" "b" 3 2
def candidate := CosineCompute.indexCommuted .fp16 "a" "b" 3 2

/-- Complete execution from empty registers, with one active lane and a tail.
This named abstract backend witnesses consistent, nonempty FP semantics. -/
def execution : {t : State ℝ // FP.Structural.exec (engine model) original initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [FP.Structural.exec,original,CosineCompute.original,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,bop,initial,ofFloat,toFloat,Region.cast,
    State.setReg,FP.Structural.store]
  rfl

theorem candidate_executes :
    FP.Structural.exec (engine model) candidate initial = some execution.val := by
  have he := FP.ProfiledRewrite.sound (FPProof.equivalent .fp16 "a" "b" 3 2)
  have hs := he ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e h; cases h) initial
    (by simp [FPProof.original])
  exact hs.symm.trans execution.property

theorem output_cells :
    (FP.Structural.exec (engine model) original initial).map (fun t => (t.mem "b" 2,t.mem "b" 3)) =
      some (.mk .fp16 1,.mk .fp16 99) := by
  rw [execution.property]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CosineCompute.TypedExecution
