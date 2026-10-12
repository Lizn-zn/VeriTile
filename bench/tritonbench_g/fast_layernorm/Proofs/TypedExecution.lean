import bench.tritonbench_g.fast_layernorm.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def storageType : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def initial (input weight bias : Storage) : State ℝ where
  mem := fun r _ => if r="x" ∨ r="y" then .mk (storageType input).toTileDType (ofFloat (storageType input) 2)
    else if r="w" then .mk (storageType weight).toTileDType (ofFloat (storageType weight) 3)
    else if r="b" then .mk (storageType bias).toTileDType (ofFloat (storageType bias) 4)
    else .mk .real 1
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-- Both complete typed bodies start from empty registers and execute with a
padded row and all independently selected storage combinations. This algebra
is a consistency witness, not a hardware rounding model. -/
theorem forward_executes (input weight bias : Storage) :
    ∃ v, FP.Structural.exec (engine model)
      (forward input weight bias Bool.false "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)
      (initial input weight bias)=some v := by
  cases input <;> cases weight <;> cases bias
  all_goals
    simp [FP.Structural.exec,forward,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices,List.finRange_succ,State.write]

theorem backward_executes (input weight bias : Storage) :
    ∃ v, FP.Structural.exec (engine model)
      (backward input weight bias Bool.false "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)
      (initial input weight bias)=some v := by
  cases input <;> cases weight <;> cases bias
  all_goals
    simp [FP.Structural.exec,backward,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices,List.finRange_succ,State.write]

theorem forward_candidate_executes (input weight bias : Storage) :
    ∃ v, FP.Structural.exec (engine model)
      (forward input weight bias Bool.true "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)
      (initial input weight bias)=some v := by
  obtain ⟨v,hv⟩ := forward_executes input weight bias
  have h := FP.ProfiledRewrite.sound
    (FPProof.forward_equivalent input weight bias "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial input weight bias) (by simp [FPProof.forwardProgram])
  exact ⟨v,h.symm.trans hv⟩

theorem backward_candidate_executes (input weight bias : Storage) :
    ∃ v, FP.Structural.exec (engine model)
      (backward input weight bias Bool.true "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)
      (initial input weight bias)=some v := by
  obtain ⟨v,hv⟩ := backward_executes input weight bias
  have h := FP.ProfiledRewrite.sound
    (FPProof.backward_equivalent input weight bias "y" "x" "w" "b" "r" "mu" 4 4 3 4 1)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial input weight bias) (by simp [FPProof.backwardProgram])
  exact ⟨v,h.symm.trans hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.TypedExecution
