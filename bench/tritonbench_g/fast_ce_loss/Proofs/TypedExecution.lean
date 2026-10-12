import bench.tritonbench_g.fast_ce_loss.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def storageType : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def initial (storage : Storage) (ignore later : Bool) : State ℝ where
  mem := fun r _ => if r="label" then .mk .int (if ignore then -100 else 1)
    else if r="logits" then .mk (storageType storage).toTileDType (ofFloat (storageType storage) 2)
    else .mk .real 1
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 1 else if later then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-- Complete typed executions from empty registers, across all flags and input
storage choices. This backend is a consistency witness, not device numerics. -/
theorem forward_executes (storage : Storage) (cap scale ignore : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (forward storage cap scale Bool.false "logits" "loss" "lse" ⟨"label"⟩ 3 4 4 1 2)
      (initial storage ignore Bool.false)=some v := by
  cases storage <;> cases cap <;> cases scale <;> cases ignore
  all_goals
    simp [FP.Structural.exec,forward,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices,List.finRange_succ,State.write]

theorem chunked_executes (storage : Storage) (cap scale ignore later : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (chunked storage cap scale Bool.false "logits" "loss" "lse" ⟨"label"⟩ 3 2 4 2 1 2)
      (initial storage ignore later)=some v := by
  cases storage <;> cases cap <;> cases scale <;> cases ignore <;> cases later
  all_goals
    simp [FP.Structural.exec,chunked,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices,List.finRange_succ,State.write]

theorem backward_executes (storage : Storage) (cap scale ignore later : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (backward storage cap scale Bool.false "logits" "dloss" "lse" ⟨"label"⟩ 3 4 1 2 1 2)
      (initial storage ignore later)=some v := by
  cases storage <;> cases cap <;> cases scale <;> cases ignore <;> cases later
  all_goals
    simp [FP.Structural.exec,backward,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices,List.finRange_succ,State.write]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.TypedExecution
