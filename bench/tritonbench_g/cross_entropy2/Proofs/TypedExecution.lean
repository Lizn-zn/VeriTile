import bench.tritonbench_g.cross_entropy2.Proofs.ForwardFP
import bench.tritonbench_g.cross_entropy2.Proofs.BackwardFP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.CrossEntropy2.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def initial : State ℝ where
  mem := fun r _ => if r="label" then .mk .int 2
    else if r="logits" then .mk .fp16 1
    else if r="gradient" then .mk .fp16 99
    else .mk .real 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def fwd := forward .fp16 Bool.false "loss" "lse" "z_loss" "logits" ⟨"label"⟩ 1 3 2 (-100) 3 0 3 2 4 2 Bool.true Bool.false

def bwd := backward .fp16 Bool.false "gradient" "dloss" "logits" "lse" ⟨"label"⟩ 1 3 2 (-100) 3 0 3 4 4 1 2 Bool.true

/-- Both complete typed kernels execute from empty registers, at a nonzero row
and column block with a masked tail. The abstract backend demonstrates
nonempty execution premises; it is not a physical GPU interpretation. -/
def forwardExecution : {t : State ℝ // FP.Structural.exec (engine model) fwd initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [FP.Structural.exec,fwd,forward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,bop,initial,ofFloat,toFloat,Region.cast,
    State.setReg,FP.Structural.store]
  rfl

def backwardExecution : {t : State ℝ // FP.Structural.exec (engine model) bwd initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [FP.Structural.exec,bwd,backward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,bop,initial,ofFloat,toFloat,Region.cast,
    State.setReg,FP.Structural.store]
  rfl

theorem forward_candidate_executes :
    FP.Structural.exec (engine model)
      (forward .fp16 Bool.true "loss" "lse" "z_loss" "logits" ⟨"label"⟩ 1 3 2 (-100) 3 0 3 2 4 2 Bool.true Bool.false)
      initial = some forwardExecution.val := by
  have hs := FP.ProfiledRewrite.sound
    (ForwardFP.equivalent .fp16 "loss" "lse" "z_loss" "logits" ⟨"label"⟩ 1 3 2 (-100) 3 0 3 2 4 2 Bool.true Bool.false)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e h; cases h) initial
    (by simp [ForwardFP.original])
  exact hs.symm.trans forwardExecution.property

theorem backward_candidate_executes :
    FP.Structural.exec (engine model)
      (backward .fp16 Bool.true "gradient" "dloss" "logits" "lse" ⟨"label"⟩ 1 3 2 (-100) 3 0 3 4 4 1 2 Bool.true)
      initial = some backwardExecution.val := by
  have hs := FP.ProfiledRewrite.sound
    (BackwardFP.equivalent .fp16 "gradient" "dloss" "logits" "lse" ⟨"label"⟩ 1 3 2 (-100) 3 0 3 4 4 1 2 Bool.true)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e h; cases h) initial
    (by simp [BackwardFP.original])
  exact hs.symm.trans backwardExecution.property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropy2.TypedExecution
