import bench.tritonbench_g.decay_cumsum.Proofs.MaskEquivalence
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.Control
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def initial : State ℝ where
  mem := fun r _ => if r ∈ (["g","go","dg"] : List RegionName) then .mk .bf16 1 else .mk .fp16 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-! These executions witness inhabited model premises for full programs,
with independent data/gate storage, two loop iterations and a partial lane.
The backend is a consistent abstract interpretation, not GPU evaluation. -/

def forwardKernel := forward .bf16 Bool.false "g" "go" 16 3 1 2 2 4 2 2 3 1

def forwardExecution : {t : State ℝ // FP.Structural.exec (engine model) forwardKernel initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  dsimp only [forwardKernel,forward,ComputeKernel.surfaceBody]
  dsimp only [FP.Structural.exec]
  simp [FP.Structural.exec,forwardKernel,forward,ComputeKernel.surfaceBody,
    run,step,range_zero,range_done,range_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,initial,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
  rfl

theorem forward_candidate_executes :
    FP.Structural.exec (engine model) (forward .bf16 Bool.true "g" "go" 16 3 1 2 2 4 2 2 3 1)
      initial = some forwardExecution.val := by
  have hs := FP.ProfiledRewrite.sound
    (MaskEquivalence.forward_equiv .bf16 "g" "go" 16 3 1 2 2 4 2 2 3 1)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e h; cases h) initial
    (by simp [MaskEquivalence.forwardProgram])
  exact hs.symm.trans forwardExecution.property

def prepareKernel := prepare .fp16 .bf16 Bool.false "q" "k" "g" "qg" "kg" 16 3 1 2 2 4 2 2 3 1

def prepareExecution : {t : State ℝ // FP.Structural.exec (engine model) prepareKernel initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  dsimp only [prepareKernel,prepare,ComputeKernel.surfaceBody]
  dsimp only [FP.Structural.exec]
  simp [FP.Structural.exec,prepareKernel,prepare,ComputeKernel.surfaceBody,
    run,step,range_zero,range_done,range_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,initial,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
  rfl

theorem prepare_candidate_executes :
    FP.Structural.exec (engine model) (prepare .fp16 .bf16 Bool.true "q" "k" "g" "qg" "kg" 16 3 1 2 2 4 2 2 3 1)
      initial = some prepareExecution.val := by
  have hs := FP.ProfiledRewrite.sound
    (MaskEquivalence.prepare_equiv .fp16 .bf16 "q" "k" "g" "qg" "kg" 16 3 1 2 2 4 2 2 3 1)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e h; cases h) initial
    (by simp [MaskEquivalence.prepareProgram])
  exact hs.symm.trans prepareExecution.property

def backwardKernel := backward .fp16 .bf16 Bool.false "dq_inner" "dq_inter" "dk_inner" "dk_inter" "q" "k" "g" "dg" 16 3 1 2 2 4 2 2 3 1

def backwardExecution : {t : State ℝ // FP.Structural.exec (engine model) backwardKernel initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  dsimp only [backwardKernel,backward,ComputeKernel.surfaceBody]
  dsimp only [FP.Structural.exec]
  simp [FP.Structural.exec,backwardKernel,backward,ComputeKernel.surfaceBody,
    run,step,range_zero,range_done,range_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,initial,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
  rfl

theorem backward_candidate_executes :
    FP.Structural.exec (engine model) (backward .fp16 .bf16 Bool.true "dq_inner" "dq_inter" "dk_inner" "dk_inter" "q" "k" "g" "dg" 16 3 1 2 2 4 2 2 3 1)
      initial = some backwardExecution.val := by
  have hs := FP.ProfiledRewrite.sound
    (MaskEquivalence.backward_equiv .fp16 .bf16 "dq_inner" "dq_inter" "dk_inner" "dk_inter" "q" "k" "g" "dg" 16 3 1 2 2 4 2 2 3 1)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e h; cases h) initial
    (by simp [MaskEquivalence.backwardProgram])
  exact hs.symm.trans backwardExecution.property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DecayCumsum.TypedExecution
